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
3. **`etf_analyze.py`** — returns, summary statistics, portfolio weights, and backtesting.

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

# 4. Compute portfolio weights
./etf_analyze.py allocate                                  # HRP (default)
./etf_analyze.py allocate --method mvo                     # Mean-Variance
./etf_analyze.py allocate --method robust                  # Robust MVO
./etf_analyze.py allocate --method mvo --max-weight 0.5    # cap 50% per ETF
./etf_analyze.py allocate --method robust --robustness-penalty 50  # stronger diversification

# 5. Rolling-window out-of-sample backtest
./etf_analyze.py backtest                                  # HRP, 3y window, quarterly
./etf_analyze.py backtest --method mvo --max-weight 0.5
./etf_analyze.py backtest --window 5 --step 6             # 5y window, semi-annual
```

Defaults: config `etf_universe.yaml`, database `hierofolio.db`, start date
`2000-01-01` (earlier than any UCITS ETF, so the first fetch returns each
ETF's full history from inception). Override with `--config`, `--db`, and
`--start` (see each script's `--help`).

## Allocator

`etf_analyze.py allocate` feeds the stored returns into one of three optimizers and prints the resulting weights plus in-sample portfolio statistics:

| Method | What it does |
|--------|-------------|
| `hrp` | Hierarchical Risk Parity — splits money down the clustering tree inversely to cluster variance. No return forecast needed. |
| `mvo` | Mean-Variance Optimization — maximises return minus risk, using historical returns as the signal. Concentrates without `--max-weight`. |
| `robust` | Robust MVO — like MVO but penalises uncertainty in the return forecast, spreading weights away from concentrated positions. Tune with `--robustness-penalty` (try 10–100). |

Key flags (all optional):

| Flag | Default | Effect |
|------|---------|--------|
| `--max-weight W` | none | Cap any single ETF at W (e.g. `0.5`). Essential for MVO/Robust to avoid full concentration. |
| `--risk-aversion λ` | 1.0 | Higher → more risk-averse; shifts MVO/Robust toward lower-vol allocations. |
| `--robustness-penalty ρ` | 1.0 | Robust only. Higher → more diversification regardless of the alpha signal. |

## Backtest

`etf_analyze.py backtest` runs a rolling-window out-of-sample evaluation: it fits the optimizer on a trailing window of returns, records what the *next* period actually returned (the optimizer never sees this), and repeats at each rebalance date. The result is an honest equity curve — no look-ahead bias.

| Flag | Default | Effect |
|------|---------|--------|
| `--window YEARS` | 3 | Length of the training window in years. |
| `--step MONTHS` | 3 | Rebalance frequency in months (3 = quarterly). |
| `--method` | hrp | Same methods and flags as `allocate`. |

Output includes a per-period rebalance log (so you can see how weights evolved) and overall out-of-sample annualised return, vol, Sharpe, and max drawdown.

## Risk model

`risk_model.py` provides `HRPRiskModel` plus the `ConstrainedMVOOptimizer` and
`RobustOptimizer` portfolio optimizers. Feed it the returns from the DB:

```python
from etf_analyze import read_returns
from risk_model import HRPRiskModel

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
(cache) with no network call. Otherwise the fetch is *incremental*: it
requests only the range *after* the last stored date (last date + 1 → now)
and inserts the new days, leaving existing rows untouched. If nothing new is
available upstream, the already-stored data is kept.

**Why doesn't a re-run pull the whole history again?**
It used to. Now the download window starts at the ISIN's last stored date, so
run #2 issues a tiny request instead of re-pulling years of data. `--force` is
the exception — it re-downloads the full range from the start date and
overwrites existing rows (use it to pick up restated prices).

**How far back does the data go?**
The default `DEFAULT_START_DATE` is `2000-01-01` (in `etf_common.py`), which is
earlier than any UCITS ETF — so the first fetch simply returns each ETF's full
history from its inception (e.g. IWDA from 2009-09-28), no inception date
needed. `--start` is an optional *cap* for when you want **less** history:
`./etf_fetch.py --start 2015-01-01` fetches only from 2015. Because fetches are
incremental, `--start` only applies on an ISIN's *first* fetch — to re-cut the
range for already-stored data, add `--force`
(`./etf_fetch.py --start 2015-01-01 --force`).

**I fetched full history, so why are there fewer combined `Observations` than
any single ETF has rows?**
The aligned matrix can only start where *all* ETFs have data, i.e. at the
latest inception among them. If your youngest fund launched in 2014, the
combined panel begins in 2014 even though older funds go back further — the
pre-2014 dates are dropped because the young fund is `NaN` there. So the
combined count tracks your newest holding; dropping it pushes the common start
(and the observation count) back.

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

**How is a security resolved — why not just the ticker?**
By **ISIN**, not the ticker. OpenFIGI maps the ISIN to a ticker, but searching
ftgo by that ticker and taking the first match can return the wrong security
(a real case: `CSSPX` matched a Cohen & Steers realty fund instead of the
iShares Core S&P 500). So `etf_fetch.py` searches ftgo by the ISIN, and pins
the chosen listing's `{xid, symbol, currency}` in `currency_metadata.yaml`.
Pinning matters because FT Markets search ordering isn't stable — without it a
later run could silently switch to a different listing/currency.

**Does an ISIN have one currency?**
No. An ISIN identifies one share class with a fixed *NAV/base* currency, but
that security is cross-listed on several exchanges, each quoting in its own
currency (e.g. IE00B5BMR087 trades as `CSPX:LSE:USD`, `CSP1:LSE:GBX`,
`SXR8:GER:EUR`). The **quote** currency depends on the listing, not the ISIN.
We store one listing per ISIN, and prefer the listing whose currency matches
the fund's own base currency (parsed from its name, e.g. "… USD (Acc)" → the
USD line), falling back to the first match. The choice is recorded in
`currency_metadata.yaml`.

**Does HRP favour low-volatility assets?**
Not directly — that's Risk Parity (a different method). HRP operates on *clusters*, not individual assets. At each split in the dendrogram it divides the budget between two branches inversely proportional to each branch's variance: the higher-variance branch gets less. Within a branch, an asset with higher vol than its peers gets a smaller slice of that branch's budget. The net effect is that HRP rewards **diversification value** — an asset that is lowly correlated with everything else gets its own branch and therefore a full share of the budget regardless of its own volatility. A highly correlated, high-vol pair share a branch and together receive less. So the driver is correlation structure first, volatility second.

*Worked example* — three assets: IWDA (vol 16%), CSPX (vol 16%), EMIM (vol 20%). IWDA and CSPX are 97% correlated, so the dendrogram merges them first. The tree has two branches: **{IWDA, CSPX}** and **{EMIM}**.

Step 1 — branch variances. Within {IWDA, CSPX}, equal inverse-variance weights give each 50%; the branch variance works out to ~0.025 (vol ≈ 15.9%) because the 0.97 correlation means the two funds move almost in lockstep. {EMIM} is a single asset: variance = 0.20² = 0.040.

Step 2 — split the total budget between branches, inversely to variance:

```
budget {IWDA, CSPX} = 0.040 / (0.025 + 0.040) = 62%
budget {EMIM}        = 0.025 / (0.025 + 0.040) = 38%
```

Step 3 — split the 62% within {IWDA, CSPX} equally (same vol → same inverse-variance weight):

```
IWDA  31%   CSPX  31%   EMIM  38%
```

EMIM receives **38% despite having the highest individual volatility (20% vs 16%)**. It earns that budget because it sits in its own branch — its low correlation with the equity pair makes it a genuine diversifier. The {IWDA + CSPX} branch, despite each fund being only 16% vol, has *combined* variance almost as high as EMIM's alone because the two funds are nearly perfectly correlated and provide no diversification to each other.

**Can I mix currencies across ETFs in the analysis?**
Not meaningfully. Per-asset daily returns are currency-invariant, but a
covariance/HRP across assets is only coherent if every series is in the same
currency (or FX-converted). `etf_analyze.py` reads `currency_metadata.yaml` and
prints a `⚠` warning if the ISINs in a `returns`/`summary` run don't share a
currency. Preferring each fund's base currency keeps a same-index panel (e.g. a
book of USD-class ETFs) consistent; genuinely mixing currencies needs FX
conversion, which isn't implemented.

## License

MIT — see [LICENSE](LICENSE).
