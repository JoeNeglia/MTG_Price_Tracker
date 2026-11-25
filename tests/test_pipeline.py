"""Unit tests for db, ingest, backfill, alerts, features, train helpers, and (once you
implement it) src/model.py.

The model tests at the bottom skip automatically while a function in src/model.py still
raises NotImplementedError, and run for real as soon as you implement it.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src import alerts, backfill, db, features, ingest

D0 = dt.date(2026, 1, 1)


# ============================================================================ fixtures
@pytest.fixture
def con():
    c = db.connect(":memory:")
    yield c
    c.close()


def add_card(con, card_id, name="Card", rarity="rare", reserved=False, edhrec=1000, set_code="tst"):
    db.upsert_cards(con, pd.DataFrame([{
        "card_id": card_id, "oracle_id": None, "name": name, "set_code": set_code,
        "collector_number": "1", "rarity": rarity, "reserved": reserved, "edhrec_rank": edhrec,
    }]))


def add_series(con, card_id, usd, start=D0, eur=None, usd_foil=None, skip=()):
    rows = []
    for i, p in enumerate(usd):
        if i in skip:
            continue
        rows.append({
            "card_id": card_id, "price_date": start + dt.timedelta(days=i), "usd": p,
            "usd_foil": None if usd_foil is None else usd_foil[i],
            "eur": None if eur is None else eur[i], "eur_foil": None, "source": "test",
        })
    db.upsert_prices(con, pd.DataFrame(rows))


def noisy(base, n, sigma=0.005, seed=0):
    rng = np.random.default_rng(seed)
    return list(base * np.exp(rng.normal(0, sigma, n)))


# ============================================================================ db
def test_schema_creates_tables(con):
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"cards", "prices", "ingest_runs"} <= tables


def test_upsert_prices_dedupes_and_merges(con):
    add_card(con, "a")
    db.upsert_prices(con, pd.DataFrame([
        {"card_id": "a", "price_date": D0, "usd": 5.0, "source": "x"},
        {"card_id": "a", "price_date": D0, "usd": 5.0, "source": "x"},  # duplicate in batch
    ]))
    # second write supplies EUR only: USD must be preserved (COALESCE merge)
    db.upsert_prices(con, pd.DataFrame([{"card_id": "a", "price_date": D0, "eur": 4.0, "source": "y"}]))
    rows = con.execute("SELECT usd, eur, source FROM prices").fetchall()
    assert rows == [(5.0, 4.0, "y")]


def test_upsert_cards_keeps_existing_values_on_null(con):
    add_card(con, "a", edhrec=123)
    db.upsert_cards(con, pd.DataFrame([{"card_id": "a", "name": "Renamed", "edhrec_rank": None}]))
    assert con.execute("SELECT name, edhrec_rank FROM cards").fetchone() == ("Renamed", 123)


def test_summary_and_latest_date(con):
    add_card(con, "a")
    add_series(con, "a", [1.0, 2.0, 3.0])
    assert db.latest_price_date(con) == D0 + dt.timedelta(days=2)
    s = db.summary(con)
    assert s["cards"] == 1 and s["price_rows"] == 3 and s["days"] == 3


# ============================================================================ ingest
SCRYFALL_SAMPLE = [
    {"id": "s1", "oracle_id": "o1", "name": "Sol Ring", "set": "c21", "collector_number": "263",
     "rarity": "uncommon", "reserved": False, "edhrec_rank": 1, "games": ["paper"],
     "prices": {"usd": "1.50", "usd_foil": None, "usd_etched": "9.99", "eur": "1.20", "eur_foil": None}},
    {"id": "s2", "name": "Arena Only", "set": "y22", "games": ["arena"], "prices": {"usd": None}},
    {"id": "s3", "name": "No Price", "set": "abc", "games": ["paper"], "prices": {"usd": None}},
]


def test_card_to_rows_uses_etched_as_foil():
    _, price = ingest.card_to_rows(SCRYFALL_SAMPLE[0], D0)
    assert price["usd"] == 1.5 and price["usd_foil"] == 9.99 and price["eur"] == 1.2


def test_ingest_stream_skips_digital_and_unpriced(con):
    stream = io.BytesIO(json.dumps(SCRYFALL_SAMPLE).encode())
    n_cards, n_prices = ingest.ingest_stream(con, stream, D0)
    assert (n_cards, n_prices) == (2, 1)
    assert con.execute("SELECT card_id FROM prices").fetchall() == [("s1",)]
    assert con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0] == 1


def test_ingest_stream_jsonl_gz(con):
    raw = "\n".join(json.dumps(c) for c in SCRYFALL_SAMPLE).encode()
    stream = gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(raw)))
    assert ingest.ingest_stream(con, stream, D0, jsonl=True) == (2, 1)


# ============================================================================ backfill
def _write_gz(path, obj):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)
    return path


def test_backfill_parses_mtgjson(tmp_path, con):
    ids = _write_gz(tmp_path / "ids.json.gz", {"meta": {}, "data": {
        "u1": {"name": "Black Lotus", "setCode": "LEA", "number": "232", "rarity": "rare",
               "isReserved": True, "edhrecRank": 5000, "identifiers": {"scryfallId": "sf1"}},
        "u2": {"name": "No Scryfall", "identifiers": {}},
    }})
    prices = _write_gz(tmp_path / "prices.json.gz", {"meta": {}, "data": {
        "u1": {"paper": {
            "tcgplayer": {"retail": {"normal": {"2026-01-01": 100.0, "2025-01-01": 1.0},
                                     "foil": {"2026-01-01": 300.0}}},
            "cardmarket": {"retail": {"normal": {"2026-01-01": 90.0}}},
        }},
        "u2": {"paper": {"tcgplayer": {"retail": {"normal": {"2026-01-01": 1.0}}}}},
    }})
    uuid_map, cards = backfill.load_identifiers(ids)
    assert uuid_map == {"u1": "sf1"}
    assert cards.iloc[0]["set_code"] == "lea" and bool(cards.iloc[0]["reserved"])

    rows = list(backfill.iter_price_rows(prices, uuid_map, min_date="2025-12-01"))
    assert rows == [{"card_id": "sf1", "price_date": "2026-01-01", "usd": 100.0,
                     "usd_foil": 300.0, "eur": 90.0, "eur_foil": None, "source": "mtgjson"}]
    db.upsert_prices(con, pd.DataFrame(rows))
    assert con.execute("SELECT usd FROM prices").fetchone() == (100.0,)


# ============================================================================ alerts
def _movers_db(con):
    n = 40
    add_card(con, "spike", "Spiky Card")
    add_series(con, "spike", noisy(10.0, n - 1, seed=1) + [15.0])        # +50% today, calm history
    add_card(con, "bulk", "Bulk Card")
    add_series(con, "bulk", noisy(1.0, n - 1, seed=2) + [2.0])           # +100% but sub-$3
    add_card(con, "drop", "Droppy Card")
    add_series(con, "drop", noisy(20.0, n - 1, seed=3) + [12.0])         # -40%
    add_card(con, "wild", "Volatile Card")
    wild = noisy(10.0, n - 1, sigma=0.15, seed=4)
    add_series(con, "wild", wild + [wild[-1] * 1.17])                    # +17%, but sigma=15%/day
    add_card(con, "gap", "Gappy Card")
    add_series(con, "gap", noisy(10.0, n - 1, seed=5) + [20.0], skip={n - 2})  # yesterday missing
    return D0 + dt.timedelta(days=n - 1)


def test_detect_movers_flags_real_moves_only(con):
    as_of = _movers_db(con)
    m = alerts.detect_movers(con, as_of)
    flagged_24h = set(m.loc[m.window == "24h", "card_id"])
    assert "spike" in flagged_24h and "drop" in flagged_24h
    assert "bulk" not in set(m.card_id)                 # sub-$3 noise filtered
    assert "wild" not in flagged_24h                    # +17% is normal for a volatile card
    assert "gap" not in flagged_24h                     # no price exactly 1 day ago -> no 24h
    assert "gap" in set(m.loc[m.window == "7d", "card_id"])
    row = m[(m.card_id == "spike") & (m.window == "24h")].iloc[0]
    assert row["direction"] == "spike"
    assert row["pct_change"] == pytest.approx(15.0 / row["price_ref"] - 1)
    assert m[(m.card_id == "drop")].iloc[0].direction == "drop"


def test_detect_movers_empty_db(con):
    assert alerts.detect_movers(con).empty


def test_format_discord_messages_chunks_under_limit():
    movers = pd.DataFrame([{
        "card_id": f"c{i}", "name": "A Very Long Card Name " * 3, "set_code": "abc",
        "collector_number": str(i), "finish": "nonfoil", "window": "24h", "price_ref": 10.0,
        "price": 15.0, "pct_change": 0.5, "z_score": 5.0, "direction": "spike",
    } for i in range(60)])
    msgs = alerts.format_discord_messages(movers, D0, top=60)
    assert len(msgs) > 1
    assert all(len(m) <= alerts.DISCORD_LIMIT for m in msgs)
    assert "Spikes (24h)" in msgs[0] and "+50%" in msgs[0]


def test_format_discord_messages_no_movers():
    msgs = alerts.format_discord_messages(pd.DataFrame(), D0)
    assert len(msgs) == 1 and "No statistically meaningful" in msgs[0]


def test_send_discord_posts_and_skips_without_url(monkeypatch):
    calls = []

    class Resp:
        status_code = 204
        def raise_for_status(self): pass

    monkeypatch.setattr(alerts.requests, "post", lambda url, json, timeout: calls.append(json) or Resp())
    monkeypatch.setattr(alerts.time, "sleep", lambda s: None)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert alerts.send_discord(["hi"]) == 0 and not calls
    assert alerts.send_discord(["a", "b"], webhook_url="https://discord.test/hook") == 2
    assert [c["content"] for c in calls] == ["a", "b"]


# ============================================================================ features
def _features_db(con):
    n = 40
    add_card(con, "a", rarity="mythic", reserved=True, edhrec=None)
    usd = [10.0] * 30 + [13.0] * 10                  # +30% jump on day 30
    add_series(con, "a", usd, eur=[9.0] * n, usd_foil=[20.0] * n)
    add_card(con, "b", rarity="common", edhrec=10)
    add_series(con, "b", [5.0] * n, eur=[4.0] * n)
    add_card(con, "cheap", rarity="common")
    add_series(con, "cheap", [1.0] * n)
    return n


def test_features_columns_and_values(con):
    _features_db(con)
    df = features.build_features(con, horizon=7)
    assert set(features.FEATURE_COLUMNS) <= set(df.columns)
    assert "cheap" not in set(df.card_id)            # min_price filter
    assert not df[features.FEATURE_COLUMNS].isna().any().any()

    a = df[df.card_id == "a"].set_index("price_date")
    day = lambda i: pd.Timestamp(D0 + dt.timedelta(days=i))
    assert a.loc[day(32), "ret_7d"] == pytest.approx(math.log(13 / 10), abs=1e-6)
    assert a.loc[day(32), "ret_1d"] == pytest.approx(0.0, abs=1e-6)
    assert a.loc[day(23), "label"] == 2              # 10 -> 13 within 7 days = +30% spike
    assert a.loc[day(5), "label"] == 1
    assert a.loc[day(5), "foil_ratio"] == pytest.approx(math.log(2.0), abs=1e-6)
    r = a.iloc[0]
    assert (r.reserved, r.rarity_mythic, r.rarity_common, r.edhrec_missing) == (1, 1, 0, 1)
    assert a.loc[day(0), "missing_lookbacks"] == 4


def test_features_us_eu_spread_is_fx_neutral(con):
    _features_db(con)
    df = features.build_features(con)
    # Two cards with EUR prices: spread is measured relative to the daily median,
    # so the two cards' spreads are symmetric around zero.
    per_day = df[df.has_eur == 1].groupby("price_date")["us_eu_spread"].sum()
    assert np.allclose(per_day, 0, atol=1e-5)


def test_features_have_no_lookahead(con):
    _features_db(con)
    before = features.build_features(con, with_labels=False)
    con.execute("UPDATE prices SET usd = usd * 3 WHERE price_date > DATE '2026-01-20'")
    after = features.build_features(con, with_labels=False)
    cutoff = pd.Timestamp("2026-01-20")
    cols = [c for c in features.FEATURE_COLUMNS if c != "us_eu_spread"]
    b = before[before.price_date <= cutoff][cols].reset_index(drop=True)
    a = after[after.price_date <= cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_latest_features_unlabelled(con):
    n = _features_db(con)
    df = features.latest_features(con)
    assert len(df) == 2 and df.label.isna().all()
    assert (df.price_date == pd.Timestamp(D0 + dt.timedelta(days=n - 1))).all()


# ============================================================================ train helpers
def test_temporal_split_has_embargo():
    from src.train import temporal_split
    dates = [D0 + dt.timedelta(days=i) for i in range(60)]
    df = pd.DataFrame({"price_date": pd.to_datetime(dates), "label": 1})
    tr, va = temporal_split(df, val_days=14, horizon=7)
    gap = (va.price_date.min() - tr.price_date.max()).days
    assert gap == 8                                   # horizon + 1: no label overlap
    assert len(va) == 14


# ============================================================================ model.py
# These skip until you implement each piece; then they must pass.
from src import model as M  # noqa: E402


def _todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{getattr(fn, '__name__', fn)} not implemented yet (src/model.py)")


def test_model_output_shape_and_grad():
    net = _todo(M.CalibratedMTGPredictor, in_features=len(features.FEATURE_COLUMNS))
    x = torch.randn(16, len(features.FEATURE_COLUMNS))
    logits = _todo(net, x)
    assert logits.shape == (16, 3) and logits.dtype == torch.float32
    logits.sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_model_architecture():
    net = _todo(M.CalibratedMTGPredictor, in_features=20)
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(20, 64), (64, 32), (32, 3)]
    assert sum(isinstance(m, torch.nn.GELU) for m in net.modules()) == 2
    drops = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(drops) == 1 and drops[0].p == pytest.approx(0.15)


def test_model_eval_is_deterministic():
    net = _todo(M.CalibratedMTGPredictor, in_features=8)
    net.eval()
    x = torch.randn(4, 8)
    assert torch.equal(_todo(net, x), net(x))


def test_brier_perfect_and_uniform():
    targets = torch.tensor([0, 1, 2, 1])
    perfect = torch.nn.functional.one_hot(targets, 3).float() * 50.0
    assert float(_todo(M.brier_score_loss, perfect, targets)) == pytest.approx(0.0, abs=1e-6)
    uniform = torch.zeros(4, 3)
    assert float(M.brier_score_loss(uniform, targets)) == pytest.approx(2 / 3, abs=1e-6)


def test_brier_is_differentiable():
    logits = torch.randn(8, 3, requires_grad=True)
    loss = _todo(M.brier_score_loss, logits, torch.randint(0, 3, (8,)))
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_ece_perfect_and_overconfident():
    targets = torch.tensor([0, 1, 2, 0])
    perfect = torch.nn.functional.one_hot(targets, 3).float()
    assert _todo(M.expected_calibration_error, perfect, targets) == pytest.approx(0.0, abs=1e-6)
    # 90% confident on class 0 every time, right only half the time -> ECE = |0.5 - 0.9| = 0.4
    probs = torch.tensor([[0.9, 0.05, 0.05]] * 10)
    targets = torch.tensor([0] * 5 + [1] * 5)
    ece = M.expected_calibration_error(probs, targets)
    assert isinstance(ece, float) and ece == pytest.approx(0.4, abs=1e-6)


def test_flip_ev():
    assert _todo(M.compute_flip_ev, [0.0, 1.0, 0.0], 10.0) == pytest.approx(-1.40)
    p, P, g, l, f = [0.1, 0.6, 0.3], 20.0, 0.25, 0.20, 0.14
    expected = P * ((1 - f) * (1 + p[2] * g - p[0] * l) - 1)
    assert M.compute_flip_ev(p, P, g, l, f) == pytest.approx(expected)
    # break-even spike probability ≈ 0.651 with default params
    assert M.compute_flip_ev([0, 1 - 0.651, 0.651], 100.0) == pytest.approx(0.0, abs=0.05)
