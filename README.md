# MTG Price Tracker + Calibrated Spike/Drop Predictor

A zero-cost Magic: The Gathering price pipeline:

1. **Backfill**: 90 days of daily prices from MTGJSON (`AllPrices` + `AllIdentifiers`) into a local DuckDB file.
2. **Daily ingest**: Scryfall's free `default_cards` bulk snapshot (USD, USD-foil, EUR), streamed and upserted.
3. **Alerts**: statistically meaningful 24h / 7d spikes and drops (bulk under $3 ignored), posted to Discord.
4. **Features + training**: SQL feature engineering and a training/eval/inference CLI for a 3-class
   calibrated predictor (`Drop >20%`, `Stable`, `Spike >20%`) and a post-fee flip-EV ranking.
5. **Automation**: a GitHub Actions workflow runs it all daily for free.

> `src/model.py` is a **skeleton** for you to implement (network, Brier loss, ECE, flip EV).
> Everything else is complete.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env               # optional: add DISCORD_WEBHOOK_URL

python -m src.backfill             # ~385 MB download, streams into data/mtg_prices.duckdb
python -m src.ingest               # today's Scryfall snapshot
python -m src.alerts               # print today's movers (add --notify to post to Discord)
python -m src.db                   # database summary
python -m src.features             # class balance + feature stats

pytest                             # model tests skip until you implement src/model.py
```

After implementing `src/model.py`:

```bash
pytest -k model -v
python -m src.train train          # temporal split + embargo, logs Brier / ECE per epoch
python -m src.train predict --top 20
```

## Layout

| File | What it does |
|---|---|
| `src/db.py` | Schema (`cards`, `prices`, `ingest_runs`), connection, COALESCE-merging upserts, `.env` loader |
| `src/backfill.py` | Streams MTGJSON with `ijson`, maps MTGJSON UUID to Scryfall ID, keeps the last N days |
| `src/ingest.py` | Streams Scryfall bulk data (gzipped JSONL, or the older JSON array) and upserts cards and prices |
| `src/alerts.py` | Window-function mover detection plus a Discord formatter/sender (chunked, rate-limit aware) |
| `src/features.py` | DuckDB SQL features and forward-looking labels |
| `src/model.py` | **Your code:** `CalibratedMTGPredictor`, `brier_score_loss`, `expected_calibration_error`, `compute_flip_ev` |
| `src/train.py` | Training loop, embargoed temporal split, baseline comparison, checkpointing, EV ranking |
| `tests/test_pipeline.py` | Unit tests for every module; the model tests turn on automatically as you implement |

## Data model

- `card_id` is the **Scryfall printing UUID** everywhere. MTGJSON rows are translated via `identifiers.scryfallId`.
- `usd`/`usd_foil` are TCGplayer market prices and `eur`/`eur_foil` are Cardmarket. The MTGJSON backfill uses
  `paper.tcgplayer.retail` and `paper.cardmarket.retail`, which are the same sources Scryfall reports.
- Upserts merge with `COALESCE`, so a later partial row never erases a known price.

## Alert logic

A move is flagged when **all** hold (see `AlertConfig` in `src/alerts.py`):

| Test | Default |
|---|---|
| Not bulk: `max(price_now, price_ref) ≥` | $3.00 |
| % move: `abs(pct_change) ≥` | 15% (24h) / 25% (7d) |
| $ move: `abs(price_now − price_ref) ≥` | $1.00 |
| Unusual for this card: `abs(z) ≥`, with `z = ln(now/ref) / (σ_daily·√h)` | 3.0 |

`σ_daily` is the card's own trailing 30-observation stdev of daily log returns (floored at 1%), so a +17% day
on a card that routinely swings 15% is ignored, while +17% on a card that never moves is flagged. Lookbacks use
`RANGE BETWEEN INTERVAL n DAY PRECEDING AND INTERVAL n DAY PRECEDING`, so a missing day gives NULL instead of
comparing against the wrong date. Non-foil and foil USD are checked separately.

## Features

Computed per (card, date) for non-foil cards ≥ $3. All features are backward-looking (a test checks this):

- `log_price`, log returns `ret_1d/3d/7d/14d`, `missing_lookbacks`, `vol_14d`, `ma7_gap`
- `us_eu_spread`: `ln(usd/eur)` minus that day's cross-card median, which cancels the EUR/USD exchange rate
- `foil_ratio`: `ln(usd_foil/usd)`
- `log_edhrec` (+ `edhrec_missing`), `reserved`, one-hot rarity

Label: the price `horizon` days later (default 7). `0` if return ≤ −20%, `2` if ≥ +20%, else `1`.

## Training notes

- **Embargo:** labels look `h` days ahead, so training rows must end `h+1` days before validation starts.
  Otherwise the training labels "see" validation prices.
- **Baseline:** always predicting the training base rates. The *Brier skill score* printed at the end
  is `1 − Brier_model / Brier_baseline`, and it must be > 0 for the model to be useful.
- Spikes are rare, so a model can look accurate and well-calibrated by always saying "Stable".
  Watch `spike_prec@1%` (precision among the 1% most spike-confident predictions).
- 90 days of history gives roughly 60–70 labelled days. Every day the workflow runs adds one more.

## GitHub Actions (free daily automation)

`.github/workflows/daily_mtg.yml` runs at **11:30 UTC** daily (and on demand via *Run workflow*):

1. Restores `data/mtg_prices.duckdb` from the Actions cache. On a miss, it backfills from MTGJSON.
2. Runs the tests, ingests today's Scryfall snapshot, and saves the DB back to the cache.
3. Runs alerts with `--notify` and uploads the DB as a 7-day run artifact, so you can download it.

Setup: push the repo to GitHub, then add the secret **`DISCORD_WEBHOOK_URL`** under
*Settings → Secrets and variables → Actions*. The workflow works without it; alerts just print to the log.

Caveats: Actions caches unused for 7 days are evicted (daily runs keep it warm). GitHub disables scheduled
workflows in public repos after 60 days without commits. If the cache is ever lost, the next run re-backfills
from MTGJSON automatically.

## Etiquette

Scryfall asks for a descriptive `User-Agent` and an `Accept` header, and bulk files are meant to be downloaded
at most once a day. MTGJSON files are cached locally in `data/raw/` and reused for 20 hours.
