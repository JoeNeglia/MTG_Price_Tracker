"""DuckDB SQL feature engineering.

One row per (card_id, price_date) for non-foil printings priced >= MIN_PRICE. Every feature
uses information available at the close of `price_date`; only the label looks forward.

Time-series features (from non-foil USD)
    log_price          ln(usd)
    ret_1d..ret_14d    log returns  ln(usd_t / usd_{t-n})   (0 when the lookback day is missing)
    missing_lookbacks  how many of the 4 lookbacks were missing (lets the model discount zeros)
    vol_14d            stdev of daily log returns over the previous 14 observations
    ma7_gap            ln(usd_t / mean(usd over last 7 obs))   — momentum vs mean reversion
Market-structure features
    us_eu_spread       ln(usd/eur) minus that day's cross-sectional median ln(usd/eur).
                       Subtracting the median cancels the EUR/USD exchange rate, leaving the
                       card-specific US premium (US demand often leads EU prices).
    foil_ratio         ln(usd_foil / usd)
    has_eur, has_foil  availability flags for the two features above
Card fundamentals
    log_edhrec         ln(1 + edhrec_rank), unranked -> ln(1 + 50000) with edhrec_missing = 1
    reserved           1 if on the Reserved List (supply can never increase)
    rarity_*           one-hot: common / uncommon / rare / mythic / other
Label (only when with_labels=True)
    fwd_ret            usd_{t+h} / usd_t - 1
    label              0 = Drop (fwd_ret <= -drop), 1 = Stable, 2 = Spike (fwd_ret >= spike)

Usage:
    python -m src.features                    # class balance + feature summary
    python -m src.features --materialize      # also write a `features` table into the DB
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from src import db

CLASS_NAMES = ["Drop >20%", "Stable", "Spike >20%"]

FEATURE_COLUMNS = [
    "log_price", "ret_1d", "ret_3d", "ret_7d", "ret_14d", "missing_lookbacks",
    "vol_14d", "ma7_gap", "us_eu_spread", "has_eur", "foil_ratio", "has_foil",
    "log_edhrec", "edhrec_missing", "reserved",
    "rarity_common", "rarity_uncommon", "rarity_rare", "rarity_mythic", "rarity_other",
]

_LOOKBACK = """first_value(usd) OVER (PARTITION BY card_id ORDER BY price_date
            RANGE BETWEEN INTERVAL {n} DAY PRECEDING AND INTERVAL {n} DAY PRECEDING)"""

FEATURES_SQL = f"""
WITH base AS (
    SELECT p.card_id, p.price_date, p.usd, p.usd_foil, p.eur
    FROM prices p
    WHERE p.usd > 0 AND p.price_date BETWEEN $start AND $end
),
ts AS (
    SELECT *,
        {_LOOKBACK.format(n=1)}  AS usd_1d,
        {_LOOKBACK.format(n=3)}  AS usd_3d,
        {_LOOKBACK.format(n=7)}  AS usd_7d,
        {_LOOKBACK.format(n=14)} AS usd_14d,
        first_value(usd) OVER (PARTITION BY card_id ORDER BY price_date
            RANGE BETWEEN INTERVAL $horizon DAY FOLLOWING AND INTERVAL $horizon DAY FOLLOWING) AS usd_fwd,
        avg(usd) OVER (PARTITION BY card_id ORDER BY price_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7,
        ln(usd / lag(usd) OVER (PARTITION BY card_id ORDER BY price_date)) AS daily_ret,
        CASE WHEN eur > 0 THEN ln(usd / eur) END AS log_us_eu
    FROM base
),
ts2 AS (
    SELECT *,
        stddev_samp(daily_ret) OVER (PARTITION BY card_id ORDER BY price_date
            ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS vol_14d_raw,
        median(log_us_eu) OVER (PARTITION BY price_date) AS log_us_eu_median
    FROM ts
)
SELECT
    t.card_id, t.price_date, c.name, c.set_code, c.rarity, t.usd,
    ln(t.usd)                                   AS log_price,
    coalesce(ln(t.usd / t.usd_1d), 0)           AS ret_1d,
    coalesce(ln(t.usd / t.usd_3d), 0)           AS ret_3d,
    coalesce(ln(t.usd / t.usd_7d), 0)           AS ret_7d,
    coalesce(ln(t.usd / t.usd_14d), 0)          AS ret_14d,
    (t.usd_1d IS NULL)::INT + (t.usd_3d IS NULL)::INT
      + (t.usd_7d IS NULL)::INT + (t.usd_14d IS NULL)::INT AS missing_lookbacks,
    coalesce(t.vol_14d_raw, 0)                  AS vol_14d,
    ln(t.usd / t.ma7)                           AS ma7_gap,
    coalesce(t.log_us_eu - t.log_us_eu_median, 0) AS us_eu_spread,
    (t.log_us_eu IS NOT NULL)::INT              AS has_eur,
    CASE WHEN t.usd_foil > 0 THEN ln(t.usd_foil / t.usd) ELSE 0 END AS foil_ratio,
    coalesce(t.usd_foil > 0, FALSE)::INT       AS has_foil,
    ln(1 + coalesce(c.edhrec_rank, 50000))      AS log_edhrec,
    (c.edhrec_rank IS NULL)::INT                AS edhrec_missing,
    coalesce(c.reserved, FALSE)::INT            AS reserved,
    (c.rarity = 'common')::INT                  AS rarity_common,
    (c.rarity = 'uncommon')::INT                AS rarity_uncommon,
    (c.rarity = 'rare')::INT                    AS rarity_rare,
    (c.rarity = 'mythic')::INT                  AS rarity_mythic,
    (coalesce(c.rarity, 'other') NOT IN ('common','uncommon','rare','mythic'))::INT AS rarity_other,
    t.usd_fwd / t.usd - 1                       AS fwd_ret,
    CASE WHEN t.usd_fwd IS NULL THEN NULL
         WHEN t.usd_fwd / t.usd - 1 <= -$drop THEN 0
         WHEN t.usd_fwd / t.usd - 1 >=  $spike THEN 2
         ELSE 1 END                             AS label
FROM ts2 t
LEFT JOIN cards c USING (card_id)
WHERE t.usd >= $min_price
  AND t.price_date BETWEEN $out_start AND $out_end
"""


def build_features(
    con,
    start: dt.date | None = None,
    end: dt.date | None = None,
    horizon: int = 7,
    min_price: float = 3.0,
    spike: float = 0.20,
    drop: float = 0.20,
    with_labels: bool = True,
) -> pd.DataFrame:
    """Compute the feature matrix for price dates in [start, end].

    with_labels=True drops rows whose forward price (t + horizon) isn't known yet;
    with_labels=False keeps them (use for live inference on the latest date).
    """
    lo, hi = con.execute("SELECT min(price_date), max(price_date) FROM prices").fetchone()
    if lo is None:
        return pd.DataFrame(columns=["card_id", "price_date", *FEATURE_COLUMNS, "fwd_ret", "label"])
    start, end = start or lo, end or hi
    params = {
        "start": start - dt.timedelta(days=21),          # warm-up for 14d lookbacks + vol
        "end": end + dt.timedelta(days=horizon),         # room for forward labels
        "out_start": start, "out_end": end,
        "min_price": min_price, "spike": spike, "drop": drop,
    }
    sql = FEATURES_SQL.replace("$horizon", str(int(horizon)))  # interval literal can't be bound
    df = con.execute(sql, params).df()
    if with_labels:
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")
    return df.sort_values(["price_date", "card_id"]).reset_index(drop=True)


def latest_features(con, min_price: float = 3.0, horizon: int = 7) -> pd.DataFrame:
    """Unlabelled features for the most recent price date (inference input)."""
    latest = db.latest_price_date(con)
    return build_features(con, latest, latest, horizon=horizon, min_price=min_price, with_labels=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--materialize", action="store_true", help="write a `features` table")
    p.add_argument("--db", help="database path")
    a = p.parse_args()

    con = db.connect(a.db)
    df = build_features(con, horizon=a.horizon, min_price=a.min_price)
    print(f"{len(df):,} labelled rows, {df['card_id'].nunique():,} cards, "
          f"{df['price_date'].min()} .. {df['price_date'].max()}")
    if not df.empty:
        counts = df["label"].value_counts().sort_index()
        for k, n in counts.items():
            print(f"  {CLASS_NAMES[k]:>11}: {n:>9,} ({n / len(df):.2%})")
        print(df[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(3))
    if a.materialize:
        con.register("_feat", df)
        con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM _feat")
        print("Wrote table `features`.")
    con.close()


if __name__ == "__main__":
    main()
