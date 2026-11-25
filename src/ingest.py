"""Daily Scryfall bulk snapshot ingestion.

Scryfall regenerates its `default_cards` bulk file every ~24h (one object per English-or-only
printing, with today's USD / USD-foil / EUR prices). The bulk metadata currently advertises a
gzipped JSON-Lines file (`jsonl_download_uri`, ~80 MB compressed / ~500 MB raw); older
responses used a plain JSON array (`download_uri`). Both are supported and both are streamed
straight from the HTTP response (gunzip -> line-by-line, or ijson for the array), so the
snapshot is never held in memory as a whole.

Usage:
    python -m src.ingest                     # fetch today's snapshot
    python -m src.ingest --file cards.jsonl.gz   # ingest a local copy (.json/.jsonl, optionally .gz)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
from pathlib import Path
from typing import IO, Iterator

import ijson
import pandas as pd
import requests

from src import db

BULK_META_URL = "https://api.scryfall.com/bulk-data/default-cards"
# Scryfall requires a descriptive User-Agent and an Accept header on API requests.
HEADERS = {"User-Agent": "mtg-price-tracker/0.1", "Accept": "application/json;q=0.9,*/*;q=0.8"}


def fetch_bulk_meta() -> dict:
    r = requests.get(BULK_META_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _price(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_cards(stream: IO[bytes], jsonl: bool = False) -> Iterator[dict]:
    """Yield raw Scryfall card objects from a JSON-Lines stream or a streaming JSON array."""
    if jsonl:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)
    else:
        yield from ijson.items(stream, "item", use_float=True)


def card_to_rows(card: dict, price_date: dt.date) -> tuple[dict, dict | None]:
    """Split one Scryfall card into (cards row, prices row or None)."""
    card_row = {
        "card_id": card["id"],
        "oracle_id": card.get("oracle_id"),
        "name": card.get("name"),
        "set_code": card.get("set"),
        "collector_number": card.get("collector_number"),
        "rarity": card.get("rarity"),
        "reserved": bool(card.get("reserved", False)),
        "edhrec_rank": card.get("edhrec_rank"),
    }
    p = card.get("prices") or {}
    usd, eur = _price(p.get("usd")), _price(p.get("eur"))
    # Etched-only printings have no usd_foil; treat etched as the foil price.
    usd_foil = _price(p.get("usd_foil")) or _price(p.get("usd_etched"))
    eur_foil = _price(p.get("eur_foil")) or _price(p.get("eur_etched"))
    if all(v is None for v in (usd, usd_foil, eur, eur_foil)):
        return card_row, None
    price_row = {"card_id": card["id"], "price_date": price_date, "usd": usd,
                 "usd_foil": usd_foil, "eur": eur, "eur_foil": eur_foil, "source": "scryfall"}
    return card_row, price_row


def ingest_stream(con, stream: IO[bytes], price_date: dt.date, jsonl: bool = False) -> tuple[int, int]:
    cards, prices = [], []
    for card in parse_cards(stream, jsonl):
        if "paper" not in (card.get("games") or ["paper"]):
            continue  # skip digital-only (Arena/MTGO) printings
        c, p = card_to_rows(card, price_date)
        cards.append(c)
        if p:
            prices.append(p)
    n_cards = db.upsert_cards(con, pd.DataFrame(cards))
    n_prices = db.upsert_prices(con, pd.DataFrame(prices))
    db.log_run(con, "scryfall_daily", price_date, n_cards, n_prices)
    return n_cards, n_prices


def run(file: Path | None = None, price_date: dt.date | None = None, db_file: str | None = None) -> None:
    con = db.connect(db_file)
    if file:
        price_date = price_date or dt.date.today()
        print(f"Ingesting local file {file} as {price_date}")
        opener = gzip.open if file.suffix == ".gz" else open
        with opener(file, "rb") as f:
            n_cards, n_prices = ingest_stream(con, f, price_date, jsonl=".jsonl" in file.name)
    else:
        meta = fetch_bulk_meta()
        # Prices in the snapshot are as-of the file's generation time (UTC).
        updated = dt.datetime.fromisoformat(meta["updated_at"].replace("Z", "+00:00"))
        price_date = price_date or updated.astimezone(dt.timezone.utc).date()
        url = meta.get("jsonl_download_uri") or meta["download_uri"]
        jsonl = ".jsonl" in url
        size = meta.get("compressed_size") or meta.get("size") or 0
        print(f"Scryfall default_cards updated {meta['updated_at']} ({size / 1e6:,.0f} MB) "
              f"-> price_date {price_date}\n  {url}")
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            r.raw.decode_content = True  # undo any gzip *transfer* encoding
            stream = gzip.GzipFile(fileobj=r.raw) if url.endswith(".gz") else r.raw
            n_cards, n_prices = ingest_stream(con, stream, price_date, jsonl=jsonl)
    print(f"Upserted {n_cards:,} cards and {n_prices:,} price rows for {price_date}.")
    for k, v in db.summary(con).items():
        print(f"  {k:>10}: {v}")
    con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, help="local default_cards JSON (optionally .gz)")
    p.add_argument("--date", type=dt.date.fromisoformat, help="override price date (YYYY-MM-DD)")
    p.add_argument("--db", help="database path")
    a = p.parse_args()
    run(a.file, a.date, a.db)


if __name__ == "__main__":
    main()
