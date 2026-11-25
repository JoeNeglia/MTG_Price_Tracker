"""One-time 90-day historical backfill from MTGJSON.

MTGJSON publishes ~90 days of daily prices in AllPrices.json.gz, keyed by MTGJSON's own
card UUID. AllIdentifiers.json.gz maps each MTGJSON UUID to a Scryfall printing ID, which
is the primary key used everywhere in this project (so the daily Scryfall ingest lines up).

Price sources used (to match what Scryfall reports day-to-day):
    usd / usd_foil  <- paper.tcgplayer.retail.{normal,foil}
    eur / eur_foil  <- paper.cardmarket.retail.{normal,foil}

Both files are multi-GB once decompressed, so they are streamed with ijson rather than
loaded into memory.

Usage:
    python -m src.backfill                 # download + load last 90 days
    python -m src.backfill --days 60
    python -m src.backfill --if-empty      # no-op when the DB already has prices (used in CI)
    python -m src.backfill --prices-file path/AllPrices.json.gz --identifiers-file path/AllIdentifiers.json.gz
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import time
from pathlib import Path

import ijson
import pandas as pd
import requests

from src import db

MTGJSON_BASE = "https://mtgjson.com/api/v5"
PRICES_URL = f"{MTGJSON_BASE}/AllPrices.json.gz"
IDENTIFIERS_URL = f"{MTGJSON_BASE}/AllIdentifiers.json.gz"
USER_AGENT = "mtg-price-tracker/0.1 (personal research project)"
FLUSH_EVERY = 500_000


def download(url: str, dest: Path, max_age_hours: float = 20, force: bool = False) -> Path:
    """Stream a file to disk, reusing a recent local copy if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        age_h = (time.time() - dest.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            print(f"  using cached {dest.name} ({age_h:.1f}h old)")
            return dest
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (50 << 20) < (1 << 20):
                    print(f"    {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
    tmp.replace(dest)
    return dest


def _open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def load_identifiers(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Stream AllIdentifiers -> (mtgjson_uuid -> scryfall_id map, cards DataFrame)."""
    uuid_to_scryfall: dict[str, str] = {}
    cards: list[dict] = []
    with _open(path) as f:
        for uuid, card in ijson.kvitems(f, "data", use_float=True):
            ids = card.get("identifiers") or {}
            sid = ids.get("scryfallId")
            if not sid:
                continue
            uuid_to_scryfall[uuid] = sid
            cards.append({
                "card_id": sid,
                "oracle_id": ids.get("scryfallOracleId"),
                "name": card.get("name"),
                "set_code": (card.get("setCode") or "").lower() or None,
                "collector_number": card.get("number"),
                "rarity": card.get("rarity"),
                "reserved": bool(card.get("isReserved", False)),
                "edhrec_rank": card.get("edhrecRank"),
            })
    return uuid_to_scryfall, pd.DataFrame(cards)


def iter_price_rows(path: Path, uuid_to_scryfall: dict[str, str], min_date: str):
    """Yield one dict per (scryfall_id, date) from AllPrices, restricted to dates >= min_date."""
    with _open(path) as f:
        for uuid, formats in ijson.kvitems(f, "data", use_float=True):
            sid = uuid_to_scryfall.get(uuid)
            paper = (formats or {}).get("paper") if sid else None
            if not paper:
                continue
            tcg = ((paper.get("tcgplayer") or {}).get("retail")) or {}
            cm = ((paper.get("cardmarket") or {}).get("retail")) or {}
            series = {
                "usd": tcg.get("normal") or {},
                "usd_foil": tcg.get("foil") or {},
                "eur": cm.get("normal") or {},
                "eur_foil": cm.get("foil") or {},
            }
            dates = set().union(*(s.keys() for s in series.values()))
            for d in dates:
                if d < min_date:  # ISO dates compare correctly as strings
                    continue
                yield {
                    "card_id": sid,
                    "price_date": d,
                    **{k: _num(s.get(d)) for k, s in series.items()},
                    "source": "mtgjson",
                }


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def run(days: int = 90, prices_file: Path | None = None, identifiers_file: Path | None = None,
        force_download: bool = False, if_empty: bool = False, db_file: str | None = None) -> None:
    con = db.connect(db_file)
    if if_empty and not db.is_empty(con):
        print("Database already has prices; skipping backfill (--if-empty).")
        return

    print("[1/3] Fetching MTGJSON files")
    identifiers_file = identifiers_file or download(
        IDENTIFIERS_URL, db.RAW_DIR / "AllIdentifiers.json.gz", force=force_download)
    prices_file = prices_file or download(
        PRICES_URL, db.RAW_DIR / "AllPrices.json.gz", force=force_download)

    print("[2/3] Parsing AllIdentifiers (streaming)")
    uuid_map, cards_df = load_identifiers(Path(identifiers_file))
    n_cards = db.upsert_cards(con, cards_df)
    print(f"  {len(uuid_map):,} MTGJSON uuids -> Scryfall ids; upserted {n_cards:,} card rows")
    del cards_df

    min_date = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    print(f"[3/3] Parsing AllPrices (streaming), keeping dates >= {min_date}")
    batch: list[dict] = []
    total = 0
    for row in iter_price_rows(Path(prices_file), uuid_map, min_date):
        batch.append(row)
        if len(batch) >= FLUSH_EVERY:
            total += db.upsert_prices(con, pd.DataFrame(batch))
            print(f"  {total:,} price rows loaded")
            batch.clear()
    if batch:
        total += db.upsert_prices(con, pd.DataFrame(batch))

    db.log_run(con, "mtgjson_backfill", db.latest_price_date(con), n_cards, total)
    print(f"Done. {total:,} price rows processed.")
    for k, v in db.summary(con).items():
        print(f"  {k:>10}: {v}")
    con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--prices-file", type=Path)
    p.add_argument("--identifiers-file", type=Path)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--if-empty", action="store_true", help="skip if the prices table already has data")
    p.add_argument("--db", help="database path (default: data/mtg_prices.duckdb)")
    a = p.parse_args()
    run(a.days, a.prices_file, a.identifiers_file, a.force_download, a.if_empty, a.db)


if __name__ == "__main__":
    main()
