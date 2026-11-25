"""Training, evaluation, and EV inference for the calibrated spike/drop predictor.

The network, loss, calibration metric, and EV calculator live in src/model.py (which you
implement). This module only orchestrates: data -> temporal split -> scaling -> training
loop -> calibration logging -> checkpoint -> inference.

Temporal split with embargo
---------------------------
Labels look `horizon` days ahead, so a training row dated t "knows" the price at t+h.
To stop that from leaking into validation, training rows must satisfy t + h < val_start:

    |------------- train -------------|-- embargo (h days) --|------ val ------|
                                                             ^ val_start

Usage:
    python -m src.train train                      # defaults: 7d horizon, last 14 days = val
    python -m src.train train --epochs 50 --lr 3e-4 --val-days 21
    python -m src.train predict --top 20           # rank today's cards by post-fee EV
    python -m src.train predict --fee-rate 0.13 --csv picks.csv
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src import db
from src.features import CLASS_NAMES, FEATURE_COLUMNS, build_features, latest_features
from src.model import (
    CalibratedMTGPredictor,
    brier_score_loss,
    compute_flip_ev,
    expected_calibration_error,
)

MODELS_DIR = db.PROJECT_ROOT / "models"
DEFAULT_CKPT = MODELS_DIR / "mtg_predictor.pt"


# ----------------------------------------------------------------------------- data
def temporal_split(df: pd.DataFrame, val_days: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["price_date"]).dt.date
    last = dates.max()
    val_start = last - dt.timedelta(days=val_days - 1)
    train_end = val_start - dt.timedelta(days=horizon + 1)  # embargo
    train, val = df[dates <= train_end], df[dates >= val_start]
    if train.empty or val.empty:
        raise SystemExit(
            f"Not enough history for a split: data spans {dates.min()}..{last}, need "
            f">= {val_days + horizon + 2} labelled days. Backfill more or lower --val-days."
        )
    print(f"Train: {dates[train.index].min()} .. {train_end}  ({len(train):,} rows)")
    print(f"Embargo: {horizon} day(s)")
    print(f"Val:   {val_start} .. {last}  ({len(val):,} rows)")
    return train, val


def to_tensors(df: pd.DataFrame, scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(scaler.transform(df[FEATURE_COLUMNS].to_numpy()), dtype=torch.float32)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    return x, y


def class_balance(y: torch.Tensor) -> np.ndarray:
    return np.bincount(y.numpy(), minlength=3) / len(y)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 8192) -> dict:
    model.eval()
    logits = torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    probs = torch.softmax(logits, dim=1)
    return metrics_from(logits, probs, y)


def metrics_from(logits: torch.Tensor, probs: torch.Tensor, y: torch.Tensor) -> dict:
    pred = probs.argmax(dim=1)
    out = {
        "brier": float(brier_score_loss(logits, y)),
        "ece": expected_calibration_error(probs, y),
        "acc": float((pred == y).float().mean()),
    }
    # Per-class calibration-in-the-large: mean predicted prob vs observed frequency.
    for k, name in enumerate(["drop", "stable", "spike"]):
        out[f"p_{name}"] = float(probs[:, k].mean())
        out[f"freq_{name}"] = float((y == k).float().mean())
    # How good is the spike ranking? Precision among the top 1% most spike-confident rows.
    n_top = max(1, len(y) // 100)
    top = probs[:, 2].topk(n_top).indices
    out["spike_prec@1%"] = float((y[top] == 2).float().mean())
    return out


def fmt(m: dict) -> str:
    return (f"brier={m['brier']:.4f}  ece={m['ece']:.4f}  acc={m['acc']:.3f}  "
            f"spike_prec@1%={m['spike_prec@1%']:.3f}")


# ----------------------------------------------------------------------------- train
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    con = db.connect(args.db, read_only=True)
    df = build_features(con, horizon=args.horizon, min_price=args.min_price,
                        spike=args.threshold, drop=args.threshold)
    con.close()
    if df.empty:
        raise SystemExit("No labelled feature rows. Run `python -m src.backfill` first.")

    train_df, val_df = temporal_split(df, args.val_days, args.horizon)
    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS].to_numpy())
    x_tr, y_tr = to_tensors(train_df, scaler)
    x_va, y_va = to_tensors(val_df, scaler)

    prior = class_balance(y_tr)
    print("Train class balance: " + ", ".join(f"{n}={p:.2%}" for n, p in zip(CLASS_NAMES, prior)))

    # Climatology baseline: always predict the training base rates. Your model needs
    # to beat this Brier score to be learning anything beyond "most cards are stable".
    base_logits = torch.log(torch.tensor(prior, dtype=torch.float32).clamp_min(1e-6)).expand(len(y_va), 3)
    baseline = metrics_from(base_logits, torch.softmax(base_logits, dim=1), y_va)
    print(f"Baseline (base rates)  val: {fmt(baseline)}\n")

    model = CalibratedMTGPredictor(in_features=len(FEATURE_COLUMNS), dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)

    best, best_state, bad_epochs = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = brier_score_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        m = evaluate(model, x_va, y_va)
        flag = ""
        if m["brier"] < best - 1e-5:
            best, best_state, bad_epochs, flag = m["brier"], copy.deepcopy(model.state_dict()), 0, " *"
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d}  train_brier={running / n:.4f}  val: {fmt(m)}{flag}")
        if bad_epochs >= args.patience:
            print(f"Early stopping (no val improvement for {args.patience} epochs).")
            break

    model.load_state_dict(best_state)
    final = evaluate(model, x_va, y_va)
    print("\n=== Best checkpoint (validation) ===")
    print(f"  model:    {fmt(final)}")
    print(f"  baseline: {fmt(baseline)}")
    skill = 1 - final["brier"] / baseline["brier"]
    print(f"  Brier skill score vs baseline: {skill:+.2%}  (>0 means the model adds information)")
    print("  Calibration-in-the-large (mean predicted vs observed):")
    for k in ("drop", "stable", "spike"):
        print(f"    {k:>6}: predicted {final[f'p_{k}']:.3%}  observed {final[f'freq_{k}']:.3%}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "feature_columns": FEATURE_COLUMNS,
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
        "config": {"horizon": args.horizon, "min_price": args.min_price,
                   "threshold": args.threshold, "dropout": args.dropout},
        "val_metrics": final,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, args.out)
    print(f"\nSaved checkpoint -> {args.out}")


# ----------------------------------------------------------------------------- predict
def predict(args: argparse.Namespace) -> None:
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Run `python -m src.train train` first.")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cols, cfg = ckpt["feature_columns"], ckpt["config"]
    if cols != FEATURE_COLUMNS:
        raise SystemExit("Checkpoint feature columns differ from src/features.py; retrain.")

    con = db.connect(args.db, read_only=True)
    df = latest_features(con, min_price=args.min_price or cfg["min_price"], horizon=cfg["horizon"])
    con.close()
    if df.empty:
        raise SystemExit("No cards to score for the latest date.")

    mean, scale = ckpt["scaler_mean"].numpy(), ckpt["scaler_scale"].numpy()
    x = torch.tensor((df[cols].to_numpy() - mean) / scale, dtype=torch.float32)
    model = CalibratedMTGPredictor(in_features=len(cols), dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()

    df["p_drop"], df["p_stable"], df["p_spike"] = probs[:, 0], probs[:, 1], probs[:, 2]
    df["ev_usd"] = [
        compute_flip_ev(p.tolist(), price, args.expected_gain, args.expected_loss, args.fee_rate)
        for p, price in zip(probs, df["usd"].to_numpy())
    ]
    df["ev_pct"] = df["ev_usd"] / df["usd"]
    ranked = df.sort_values("ev_usd", ascending=False)

    as_of = df["price_date"].iloc[0]
    print(f"Predictions for {as_of} | horizon {cfg['horizon']}d | fee {args.fee_rate:.0%} | "
          f"{len(df):,} cards scored | {(df['ev_usd'] > 0).sum():,} with positive EV\n")
    show = ranked.head(args.top)[["name", "set_code", "usd", "p_drop", "p_stable", "p_spike", "ev_usd", "ev_pct"]]
    with pd.option_context("display.width", 140, "display.max_colwidth", 32):
        print(show.to_string(index=False, formatters={
            "usd": "${:,.2f}".format, "ev_usd": "${:+,.2f}".format, "ev_pct": "{:+.1%}".format,
            "p_drop": "{:.1%}".format, "p_stable": "{:.1%}".format, "p_spike": "{:.1%}".format,
        }))
    if args.csv:
        ranked.to_csv(args.csv, index=False)
        print(f"\nWrote {len(ranked):,} rows -> {args.csv}")


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="database path")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train + validate, save checkpoint")
    t.add_argument("--horizon", type=int, default=7, help="label horizon in days")
    t.add_argument("--threshold", type=float, default=0.20, help="spike/drop threshold")
    t.add_argument("--min-price", type=float, default=3.0)
    t.add_argument("--val-days", type=int, default=14)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", type=Path, default=DEFAULT_CKPT)

    q = sub.add_parser("predict", help="score the latest date and rank by post-fee EV")
    q.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    q.add_argument("--top", type=int, default=20)
    q.add_argument("--min-price", type=float, default=None)
    q.add_argument("--expected-gain", type=float, default=0.25)
    q.add_argument("--expected-loss", type=float, default=0.20)
    q.add_argument("--fee-rate", type=float, default=0.14)
    q.add_argument("--csv", type=Path)

    args = p.parse_args(argv)
    try:
        {"train": train, "predict": predict}[args.cmd](args)
    except NotImplementedError as e:
        print(f"\n[src/model.py] {e}\nA function in src/model.py is still a stub. "
              "Implement it (see the guide comments) and re-run.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
