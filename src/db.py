"""DuckDB schema setup and connection helpers.

Schema
------
cards        one row per Scryfall printing (card_id = Scryfall UUID)
prices       one row per (card_id, price_date); USD = TCGplayer market, EUR = Cardmarket trend
ingest_runs  audit log of every backfill / daily ingest
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_DB_PATH = DATA_DIR / "mtg_prices.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    card_id          VARCHAR PRIMARY KEY,   -- Scryfall printing UUID
    oracle_id        VARCHAR,
    name             VARCHAR NOT NULL,
    set_code         VARCHAR,
    collector_number VARCHAR,
    rarity           VARCHAR,               -- common / uncommon / rare / mythic / special / bonus
    reserved         BOOLEAN DEFAULT FALSE, -- on the Reserved List
    edhrec_rank      INTEGER,               -- lower = more played in Commander; NULL = unranked
    updated_at       TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS prices (
    card_id    VARCHAR NOT NULL,
    price_date DATE    NOT NULL,
    usd        DOUBLE,
    usd_foil   DOUBLE,
    eur        DOUBLE,
    eur_foil   DOUBLE,
    source     VARCHAR,                     -- 'mtgjson' or 'scryfall'
    PRIMARY KEY (card_id, price_date)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_at     TIMESTAMP DEFAULT current_timestamp,
    source     VARCHAR,
    price_date DATE,
    n_cards    BIGINT,
    n_prices   BIGINT
);
"""

CARD_COLUMNS = [
    "card_id", "oracle_id", "name", "set_code", "collector_number",
    "rarity", "reserved", "edhrec_rank",
]
PRICE_COLUMNS = ["card_id", "price_date", "usd", "usd_foil", "eur", "eur_foil", "source"]


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency). Never overrides real env vars."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def db_path() -> Path:
    load_env()
    return Path(os.environ.get("MTG_DB_PATH") or DEFAULT_DB_PATH)


def connect(path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (and initialise) the DuckDB database. Pass ':memory:' for tests."""
    path = str(path) if path is not None else str(db_path())
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path, read_only=read_only)
    if not read_only:
        init_schema(con)
    return con


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def upsert_cards(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert or update card metadata. Non-null incoming values win; NULLs keep the old value."""
    if df.empty:
        return 0
    df = df.reindex(columns=CARD_COLUMNS)
    con.register("_cards_in", df)
    con.execute(
        """
        INSERT INTO cards (card_id, oracle_id, name, set_code, collector_number,
                           rarity, reserved, edhrec_rank, updated_at)
        SELECT CAST(card_id AS VARCHAR), CAST(oracle_id AS VARCHAR), CAST(name AS VARCHAR),
               CAST(set_code AS VARCHAR), CAST(collector_number AS VARCHAR), CAST(rarity AS VARCHAR),
               COALESCE(TRY_CAST(reserved AS BOOLEAN), FALSE), TRY_CAST(edhrec_rank AS INTEGER),
               current_timestamp
        FROM _cards_in
        WHERE card_id IS NOT NULL AND name IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY card_id ORDER BY edhrec_rank NULLS LAST) = 1
        ON CONFLICT (card_id) DO UPDATE SET
            oracle_id        = COALESCE(EXCLUDED.oracle_id, cards.oracle_id),
            name             = EXCLUDED.name,
            set_code         = COALESCE(EXCLUDED.set_code, cards.set_code),
            collector_number = COALESCE(EXCLUDED.collector_number, cards.collector_number),
            rarity           = COALESCE(EXCLUDED.rarity, cards.rarity),
            reserved         = EXCLUDED.reserved,
            edhrec_rank      = COALESCE(EXCLUDED.edhrec_rank, cards.edhrec_rank),
            updated_at       = EXCLUDED.updated_at
        """
    )
    con.unregister("_cards_in")
    return len(df)


def upsert_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert or merge daily prices. Duplicate keys inside `df` are collapsed first
    (DuckDB rejects ON CONFLICT batches that hit the same key twice)."""
    if df.empty:
        return 0
    df = df.reindex(columns=PRICE_COLUMNS)
    con.register("_prices_in", df)
    con.execute(
        """
        INSERT INTO prices
        SELECT card_id, CAST(price_date AS DATE),
               max(usd), max(usd_foil), max(eur), max(eur_foil), any_value(source)
        FROM _prices_in
        WHERE card_id IS NOT NULL AND price_date IS NOT NULL
          AND (usd IS NOT NULL OR usd_foil IS NOT NULL OR eur IS NOT NULL OR eur_foil IS NOT NULL)
        GROUP BY card_id, CAST(price_date AS DATE)
        ON CONFLICT (card_id, price_date) DO UPDATE SET
            usd      = COALESCE(EXCLUDED.usd, prices.usd),
            usd_foil = COALESCE(EXCLUDED.usd_foil, prices.usd_foil),
            eur      = COALESCE(EXCLUDED.eur, prices.eur),
            eur_foil = COALESCE(EXCLUDED.eur_foil, prices.eur_foil),
            source   = EXCLUDED.source
        """
    )
    con.unregister("_prices_in")
    return len(df)


def log_run(con, source: str, price_date, n_cards: int, n_prices: int) -> None:
    con.execute(
        "INSERT INTO ingest_runs (source, price_date, n_cards, n_prices) VALUES (?, ?, ?, ?)",
        [source, price_date, n_cards, n_prices],
    )


def latest_price_date(con):
    return con.execute("SELECT max(price_date) FROM prices").fetchone()[0]


def is_empty(con) -> bool:
    return con.execute("SELECT count(*) FROM prices").fetchone()[0] == 0


def summary(con) -> dict:
    n_cards, = con.execute("SELECT count(*) FROM cards").fetchone()
    n_prices, d0, d1, n_days = con.execute(
        "SELECT count(*), min(price_date), max(price_date), count(DISTINCT price_date) FROM prices"
    ).fetchone()
    return {"cards": n_cards, "price_rows": n_prices, "first_date": d0, "last_date": d1, "days": n_days}


if __name__ == "__main__":
    with connect() as con:
        print(f"Database: {db_path()}")
        for k, v in summary(con).items():
            print(f"  {k:>10}: {v}")
