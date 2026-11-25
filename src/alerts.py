"""Price mover detection (SQL window functions) + Discord webhook alerts.

A move is flagged only when ALL of these hold:
  1. The card is not bulk:          max(price_now, price_ref) >= MIN_PRICE ($3.00)
  2. The move is large in % terms:  |pct_change| >= pct threshold (24h: 15%, 7d: 25%)
  3. The move is large in $ terms:  |price_now - price_ref| >= MIN_ABS_CHANGE ($1.00)
  4. The move is unusual *for this card*:
         z = ln(price_now / price_ref) / (sigma_daily * sqrt(h)) ,  |z| >= Z_THRESHOLD
     where sigma_daily is the stdev of the card's own daily log-returns over the previous
     30 observations (floored at 1%) and h is the window length in days. Cards with too
     little history (< MIN_HISTORY returns) skip test 4 and rely on tests 1-3.

Lookbacks use `RANGE BETWEEN INTERVAL n DAY PRECEDING AND INTERVAL n DAY PRECEDING`, so a
missing day yields NULL instead of silently comparing against the wrong date.

Usage:
    python -m src.alerts                 # print today's movers
    python -m src.alerts --notify        # ...and post them to DISCORD_WEBHOOK_URL
    python -m src.alerts --date 2026-11-20 --top 25
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass

import pandas as pd
import requests

from src import db


@dataclass(frozen=True)
class AlertConfig:
    min_price: float = 3.00
    min_abs_change: float = 1.00
    pct_24h: float = 0.15
    pct_7d: float = 0.25
    z_threshold: float = 3.0
    min_history: int = 10
    vol_floor: float = 0.01
    vol_lookback: int = 30


MOVERS_SQL = """
WITH long AS (                         -- unpivot non-foil / foil USD into one series per finish
    SELECT card_id, price_date, 'nonfoil' AS finish, usd AS price
    FROM prices WHERE usd > 0 AND price_date BETWEEN $start AND $as_of
    UNION ALL
    SELECT card_id, price_date, 'foil', usd_foil
    FROM prices WHERE usd_foil > 0 AND price_date BETWEEN $start AND $as_of
),
lagged AS (
    SELECT *,
        first_value(price) OVER (PARTITION BY card_id, finish ORDER BY price_date
            RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS p_24h,
        first_value(price) OVER (PARTITION BY card_id, finish ORDER BY price_date
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 7 DAY PRECEDING) AS p_7d,
        ln(price / lag(price) OVER (PARTITION BY card_id, finish ORDER BY price_date)) AS log_ret
    FROM long
),
vol AS (                               -- trailing volatility, EXCLUDING today's return
    SELECT *,
        stddev_samp(log_ret) OVER hist AS sigma_daily,
        count(log_ret)       OVER hist AS n_hist
    FROM lagged
    WINDOW hist AS (PARTITION BY card_id, finish ORDER BY price_date
                    ROWS BETWEEN $vol_lookback PRECEDING AND 1 PRECEDING)
),
today AS (SELECT * FROM vol WHERE price_date = $as_of),
windows AS (
    SELECT card_id, finish, price_date, price, p_24h AS price_ref, '24h' AS window, 1 AS h,
           $pct_24h AS pct_threshold, sigma_daily, n_hist FROM today WHERE p_24h > 0
    UNION ALL
    SELECT card_id, finish, price_date, price, p_7d, '7d', 7,
           $pct_7d, sigma_daily, n_hist FROM today WHERE p_7d > 0
),
scored AS (
    SELECT *,
        price / price_ref - 1                                         AS pct_change,
        price - price_ref                                             AS abs_change,
        ln(price / price_ref)
            / (greatest(coalesce(sigma_daily, 0), $vol_floor) * sqrt(h)) AS z_score
    FROM windows
)
SELECT s.card_id, c.name, c.set_code, c.collector_number, c.rarity, s.finish, s.window,
       s.price_date, s.price_ref, s.price, s.pct_change, s.abs_change, s.z_score, s.n_hist,
       CASE WHEN s.pct_change > 0 THEN 'spike' ELSE 'drop' END AS direction
FROM scored s
LEFT JOIN cards c USING (card_id)
WHERE greatest(s.price, s.price_ref) >= $min_price
  AND abs(s.pct_change) >= s.pct_threshold
  AND abs(s.abs_change) >= $min_abs_change
  AND (s.n_hist < $min_history OR abs(s.z_score) >= $z_threshold)
ORDER BY abs(s.z_score) DESC, abs(s.pct_change) DESC
"""


def detect_movers(con, as_of: dt.date | None = None, cfg: AlertConfig = AlertConfig()) -> pd.DataFrame:
    """Return flagged 24h / 7d spikes and drops for `as_of` (default: latest date in DB)."""
    as_of = as_of or db.latest_price_date(con)
    if as_of is None:
        return pd.DataFrame()
    params = {
        "as_of": as_of,
        "start": as_of - dt.timedelta(days=cfg.vol_lookback * 2 + 8),
        "pct_24h": cfg.pct_24h, "pct_7d": cfg.pct_7d, "min_price": cfg.min_price,
        "min_abs_change": cfg.min_abs_change, "z_threshold": cfg.z_threshold,
        "min_history": cfg.min_history, "vol_floor": cfg.vol_floor,
        "vol_lookback": cfg.vol_lookback,
    }
    # DuckDB can't bind a parameter inside a ROWS frame, so inline that (trusted int) one.
    sql = MOVERS_SQL.replace("$vol_lookback", str(int(cfg.vol_lookback)))
    params.pop("vol_lookback")
    return con.execute(sql, params).df()


# ---------------------------------------------------------------------------- Discord
DISCORD_LIMIT = 2000


def _card_line(r) -> str:
    arrow = "▲" if r.pct_change > 0 else "▼"
    foil = " ✨foil" if r.finish == "foil" else ""
    set_code = (r.set_code or "?").upper()
    if r.set_code and r.collector_number:
        name = f"[{r.name}](<https://scryfall.com/card/{r.set_code}/{r.collector_number}>)"
    else:
        name = f"**{r.name}**"
    return (f"{arrow} {name} ({set_code}{foil}) ${r.price_ref:,.2f} → **${r.price:,.2f}** "
            f"({r.pct_change:+.0%}, z={r.z_score:+.1f})")


def format_discord_messages(movers: pd.DataFrame, as_of, top: int = 15) -> list[str]:
    """Render movers into Discord message bodies, each under the 2000-char limit."""
    if movers.empty:
        return [f"**MTG price movers — {as_of}**\nNo statistically meaningful moves today."]
    lines = [f"**MTG price movers — {as_of}**"]
    for window in ("24h", "7d"):
        for direction, header in (("spike", "📈 Spikes"), ("drop", "📉 Drops")):
            sub = movers[(movers.window == window) & (movers.direction == direction)].head(top)
            if sub.empty:
                continue
            lines.append(f"\n__{header} ({window})__")
            lines.extend(_card_line(r) for r in sub.itertuples())

    messages, current = [], ""
    for line in lines:
        line = line[: DISCORD_LIMIT - 1]
        if len(current) + len(line) + 1 > DISCORD_LIMIT:
            messages.append(current.rstrip())
            current = ""
        current += line + "\n"
    if current.strip():
        messages.append(current.rstrip())
    return messages


def send_discord(messages: list[str], webhook_url: str | None = None, max_retries: int = 3) -> int:
    """POST each message to the webhook, honouring 429 rate limits. Returns messages sent."""
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("DISCORD_WEBHOOK_URL not set; skipping Discord notification.")
        return 0
    sent = 0
    for msg in messages:
        for _ in range(max_retries):
            r = requests.post(url, json={"content": msg, "username": "MTG Price Tracker",
                                         "allowed_mentions": {"parse": []}}, timeout=15)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 1.0)))
                continue
            r.raise_for_status()
            sent += 1
            break
        time.sleep(0.5)
    return sent


def main() -> None:
    db.load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", type=dt.date.fromisoformat)
    p.add_argument("--notify", action="store_true", help="post to DISCORD_WEBHOOK_URL")
    p.add_argument("--top", type=int, default=15, help="max cards per section")
    p.add_argument("--min-price", type=float, default=AlertConfig.min_price)
    p.add_argument("--db", help="database path")
    a = p.parse_args()

    con = db.connect(a.db, read_only=True)
    as_of = a.date or db.latest_price_date(con)
    movers = detect_movers(con, as_of, AlertConfig(min_price=a.min_price))
    con.close()

    messages = format_discord_messages(movers, as_of, a.top)
    print("\n\n".join(messages))
    print(f"\n{len(movers)} movers flagged for {as_of}.")
    if a.notify:
        print(f"Sent {send_discord(messages)} Discord message(s).")


if __name__ == "__main__":
    main()
