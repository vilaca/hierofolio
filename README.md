# Hierofolio

Hierarchical Risk Parity for UCITS ETFs. Build an ETF universe from ISINs,
fetch historical prices into SQLite, and feed the returns into a hierarchical
risk model and portfolio optimizers.

## Setup

Requires Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

The tooling is three small scripts around a shared config/DB:

1. **`etf_config.py`** — build the ETF universe YAML from ISINs (via OpenFIGI).
2. **`etf_fetch.py`** — populate the SQLite price DB (ftgo, with a yfinance fallback).
3. **`etf_analyze.py`** — returns and summary statistics from the DB.

```bash
# 1. Add ETFs by ISIN (auto-resolves name, tickers, exchange, FIGI)
./etf_config.py add IE00BM67HK77
./etf_config.py add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66
./etf_config.py list
./etf_config.py update IE00BM67HK77

# 2. Fetch prices into the SQLite DB
./etf_fetch.py                 # all ETFs in the config
./etf_fetch.py IE00BM67HK77    # a single ISIN
./etf_fetch.py --force         # ignore the cache and re-download

# 3. Analyze the stored data
./etf_analyze.py returns              # returns for all ETFs
./etf_analyze.py returns IE00BM67HK77 # returns for a single ISIN
./etf_analyze.py summary              # annualized stats + correlation
```

Defaults: config `etf_universe.yaml`, database `hierofolio.db`, start date
`2018-01-01`. Override with `--config`, `--db`, and `--start` (see each
script's `--help`).

## Risk model

`RiskModel.py` provides `HRPRiskModel` plus the `ConstrainedMVOOptimizer` and
`RobustOptimizer` portfolio optimizers. Feed it the returns from the DB:

```python
from etf_analyze import read_returns
from RiskModel import HRPRiskModel

returns = read_returns("hierofolio.db")

risk_model = HRPRiskModel(
    returns=returns,
    shrinkage_method="constant_correlation",
    shrinkage_intensity=0.3,
    cluster_mode="full",
    linkage_method="ward",
)
```

The optimizers require `cvxpy` (included in `requirements.txt`).

## Tests

```bash
pytest test_riskmodel.py
```

## FAQ

**If I run `./etf_fetch.py` again, what happens?**
Each ISIN is only re-downloaded when its newest stored date isn't today
(`days_behind > 0`). If it's already current, that ISIN loads from the DB
(cache) with no network call. Otherwise it re-fetches and inserts only the
missing days — existing rows are left untouched.

**What if I run it again after the trading day completes?**
It picks up the new day. Once the newest stored date is behind "today", the
next run re-fetches and the upsert appends the newly published close. (Before
the change to `days_behind > 0`, a 7-day gate meant a plain run could skip new
days for up to a week — `--force` was the workaround.)

**Why is there no data for today's date?**
The source only returns *completed* trading days, so today's close isn't
available until after the market closes. Days with no source data are never
stored as empty/placeholder rows — they simply don't appear.

**What are "data points" / "trading days" — are they days?**
Yes. One row per day the market was open. Weekends and exchange holidays are
absent, so ~252 rows per year, one `close` each.

**What does `Observations: N` mean?**
It's the row count of the *combined* price matrix across all ETFs, not one
fund. Dates are the union of every ETF's trading days, forward-filled across
gaps, then rows still missing any ETF are dropped. Because different exchanges
have different holiday calendars, this union can exceed any single ETF's count.

**Why do I get the same output with and without `--force`?**
The summary reports the final data, which is the same either way. `--force`
only (a) re-fetches even when an ISIN is already current to today, and
(b) overwrites existing rows instead of keeping them — visible only if the
source restated a past close. Same prices in the DB → same output.

## License

MIT — see [LICENSE](LICENSE).
