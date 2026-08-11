# Hierofolio

Hierarchical Risk Parity for UCITS ETFs. Build an ETF universe from ISINs,
fetch historical prices into SQLite, and feed the returns into a hierarchical
risk model and portfolio optimizers.

## Setup

Requires Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `hierofolio` command (and the short alias `hfolio`).

## Workflow

The tool exposes three subcommands around a shared config/DB:

1. **`hierofolio config`** — build the ETF universe YAML from ISINs (via OpenFIGI).
2. **`hierofolio fetch`** — populate the SQLite price DB (ftgo, with a yfinance fallback).
3. **`hierofolio analyze`** — returns, summary statistics, portfolio weights, and backtesting.

```bash
# 1. Add ETFs by ISIN (auto-resolves name, tickers, exchange, FIGI)
hierofolio config add IE00BM67HK77
hierofolio config add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66
hierofolio config list
hierofolio config update IE00BM67HK77

# 2. Fetch prices into the SQLite DB
hierofolio fetch                 # all ETFs in the config
hierofolio fetch IE00BM67HK77    # a single ISIN
hierofolio fetch --force         # ignore the cache and re-download

# 3. Analyze the stored data
hierofolio analyze returns              # returns for all ETFs
hierofolio analyze returns IE00BM67HK77 # returns for a single ISIN
hierofolio analyze summary              # annualized stats + correlation

# 4. Compute portfolio weights
hierofolio analyze allocate                                  # HRP (default)
hierofolio analyze allocate --method schur-hrp --gamma 0.5   # HRP + between-cluster correlations
hierofolio analyze allocate --method mvo                     # Mean-Variance
hierofolio analyze allocate --method robust                  # Robust MVO
hierofolio analyze allocate --method mvo --max-weight 0.5    # cap 50% per ETF
hierofolio analyze allocate --method robust --robustness-penalty 50  # stronger diversification

# 5. Rolling-window out-of-sample backtest
hierofolio analyze backtest                                  # HRP, 3y window, quarterly
hierofolio analyze backtest --method mvo --max-weight 0.5
hierofolio analyze backtest --window 5 --step 6             # 5y window, semi-annual
```

Both `hierofolio` and the shorter `hfolio` invoke the same tool.

Defaults: config `config/etf_universe.yaml`, database `data/hierofolio.db`,
start date `2000-01-01` (earlier than any UCITS ETF, so the first fetch returns
each ETF's full history from inception). Paths resolve against the project root,
so the commands work from any directory. Override with `--config`, `--db`, and
`--start` (see each command's `--help`).

## Allocator

`hierofolio analyze allocate` feeds the stored returns into one of four optimizers and prints the resulting weights plus in-sample portfolio statistics:

| Method | What it does |
|--------|-------------|
| `hrp` | [Hierarchical Risk Parity](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity) — splits money down the [clustering tree](https://en.wikipedia.org/wiki/Hierarchical_clustering) inversely to cluster variance. No return forecast needed. |
| `schur-hrp` | Schur-complementary HRP — plain HRP that *also* uses the correlations **between** clusters (which HRP ignores), folded back in via the [Schur complement](https://en.wikipedia.org/wiki/Schur_complement). A `--gamma` dial runs from plain HRP (0) toward a lower-variance, min-variance-like allocation. No return forecast needed. |
| `mvo` | [Mean-Variance Optimization](https://en.wikipedia.org/wiki/Modern_portfolio_theory) — maximises return minus risk, using historical returns as the signal. Concentrates without `--max-weight`. |
| `robust` | Robust MVO — like MVO but penalises uncertainty in the return forecast, spreading weights away from concentrated positions. Tune with `--robustness-penalty` (try 10–100). |

Key flags (all optional):

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--gamma γ` | 0–1 | 0.5 | Schur-HRP only. `0` = plain HRP; higher folds in more of the between-cluster correlation, pulling toward a min-variance-like allocation. Try `0.25`–`0.5`; very high values can overshoot, so verify with a backtest. |
| `--max-weight W` | 0–1 | none | Cap any single ETF at W (e.g. `0.5`). Essential for MVO/Robust to avoid full concentration. |
| `--risk-aversion λ` | > 0 | 1.0 | Higher → more risk-averse; shifts MVO/Robust toward lower-vol allocations. |
| `--robustness-penalty ρ` | > 0 | 1.0 | Robust only. Higher → more diversification regardless of the alpha signal. Try 10–100. |
| `--shrinkage-method` | `ledoit_wolf`, `constant_correlation`, `identity` | `constant_correlation` | [Covariance shrinkage](https://en.wikipedia.org/wiki/Shrinkage_(statistics)) estimator used to build the risk model. |
| `--shrinkage-intensity α` | 0–1 | 0.3 | Blend between sample and target covariance. Ignored for `ledoit_wolf` (data-driven). |
| `--linkage-method` | `ward`, `average`, `complete`, `single` | `ward` | Hierarchical clustering linkage. `ward` minimises within-cluster variance; `average` is more robust to outliers. |

## Backtest

`hierofolio analyze backtest` runs a rolling-window out-of-sample [backtest](https://en.wikipedia.org/wiki/Backtesting): it fits the optimizer on a trailing window of returns, records what the *next* period actually returned (the optimizer never sees this), and repeats at each rebalance date. The result is an honest equity curve — no look-ahead bias.

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--window YEARS` | > 0 | 3 | Length of the training window in years. |
| `--step MONTHS` | > 0 | 3 | Rebalance frequency in months (3 = quarterly). |
| `--method` | `hrp`, `schur-hrp`, `mvo`, `robust` | `hrp` | Same methods and flags as `allocate`. |
| `--gamma γ` | 0–1 | 0.5 | Schur-HRP only. `0` = plain HRP; higher pulls toward a min-variance-like allocation. |
| `--shrinkage-method` | `ledoit_wolf`, `constant_correlation`, `identity` | `constant_correlation` | Covariance shrinkage estimator. |
| `--shrinkage-intensity α` | 0–1 | 0.3 | Blend between sample and target covariance. Ignored for `ledoit_wolf`. |
| `--linkage-method` | `ward`, `average`, `complete`, `single` | `ward` | Hierarchical clustering linkage. |
| `--broker BROKER` | `xtb`, `degiro`, `traderepublic`, `ibkr` | none | Apply a named broker cost profile (see below). |
| `--portfolio-size EUR` | > 0 | 10000 | Portfolio size in EUR, used to convert flat fees to a fraction of NAV. |
| `--cost-bps BPS` | > 0 | none | Manual round-trip cost override in basis points (overrides `--broker`). |

Output includes a per-period rebalance log, overall out-of-sample statistics versus an equal-weight benchmark, and (when costs are modelled) a total cost drag line.

### Broker cost profiles

Transaction costs are modelled per rebalance as: `max(min_eur, flat_fee_eur + bps_per_side × notional_traded)` per asset. The optional `min_eur` is a per-order commission floor (e.g. IBKR's €1.25) that dominates on small trades. Profiles are stored in `config/broker_profiles.yaml` and can be edited to reflect your actual terms.

| Profile key | Broker | Model | Approx cost on €10k trade |
|-------------|--------|-------|--------------------------|
| `xtb` | XTB | Spread only | ~5 bps round-trip |
| `degiro` | DEGIRO | €1 + 3 bps/side (core ETF list) | ~8 bps round-trip |
| `traderepublic` | Trade Republic | €1 flat + 1.5 bps/side | size-dependent |
| `ibkr` | Interactive Brokers | 5 bps/side (≈ 0.05%, min €1.25) | ~10 bps round-trip |

**Country-specific taxes are not included** in the profiles — add them via `--cost-bps`. For example, Belgian investors pay a 0.35% TOB (35 bps) on ETF purchases:

```bash
hierofolio analyze backtest --broker degiro --cost-bps 35 --portfolio-size 20000
```

## Practical examples

### Get your first allocation

```bash
hierofolio analyze allocate
```

*Why:* This is the starting point. It reads your historical prices and tells you how much of your money to put in each ETF using HRP — the most robust method since it doesn't require predicting future returns. Run this whenever you want to know the current suggested weights.

*How it works:* It converts your stored daily prices into daily percentage returns (e.g. +0.5% on Monday), measures how correlated every pair of ETFs is, and groups ETFs that tend to move together. It then splits your budget down that grouping tree (simplified): at every branch, the side with higher combined volatility gets less money and the calmer side gets more. This repeats until every ETF has a weight. The weights always sum to 100%.

---

### Understand why HRP gave each ETF its weight

```bash
hierofolio analyze allocate --verbose
```

*Why:* HRP's output can be surprising — an ETF with higher volatility can still get a larger allocation than a calmer one. `--verbose` shows the step-by-step calculation: which ETFs cluster together, what the combined risk of each cluster is, and how the budget gets split. Useful when you want to understand (or explain to someone else) why the numbers came out the way they did.

*How it works:* Same calculation as the plain `allocate`, but at each split it prints the two groups being compared, the annualised volatility of each group (how much it would swing per year), and what fraction of the current budget each group receives. Reading it top-to-bottom you can trace exactly where every euro ended up and why.

---

### Check whether your ETFs are actually diversified

```bash
hierofolio analyze summary
```

*Why:* Before running any optimizer, it's worth checking whether your ETFs move together. If the correlation matrix shows values above 0.8 everywhere, your funds are essentially the same bet — no optimizer can create real diversification from that. The summary shows you exactly this. If everything is highly correlated, consider adding a bond or gold ETF to give the model something to work with.

*How it works:* It takes each ETF's average daily return and compounds it over 252 trading days — `(1 + daily average)^252 − 1` — to get an annual return (compounding, not a flat ×252, because gains build on gains). Volatility is the daily standard deviation scaled up by √252. Then it prints the correlation matrix — a grid where each cell shows how closely two ETFs move together: 1.0 means they always move in lockstep, 0 means unrelated, −1 means they always move in opposite directions.

---

### Limit how much goes into any single ETF

```bash
hierofolio analyze allocate --method mvo --max-weight 0.5
```

*Why:* Without a cap, MVO tends to put everything into whichever ETF had the best past performance — which is rarely a good idea going forward. `--max-weight 0.5` forces the optimizer to spread at least some money across other ETFs. A cap of 0.4–0.5 is a sensible starting point for a 3–5 ETF portfolio.

*How it works:* It treats each ETF's average historical return as a forecast of its future return, then hands the problem to a mathematical solver. The solver finds the mix of weights that maximises expected return minus a risk penalty (portfolio variance). `--max-weight 0.5` is a hard rule fed directly into the solver: any solution where a single ETF exceeds 50% is rejected.

---

### Use the correlations *between* clusters, not just within them (Schur-HRP)

```bash
hierofolio analyze allocate --method schur-hrp --gamma 0.5
```

*Why:* Plain HRP has a blind spot. When it splits your budget between two groups of ETFs, it looks only at the risk *inside* each group and ignores how the two groups move relative to each other — so two groups that are actually strongly correlated get treated as if they were independent. Schur-HRP puts that missing information back in. `--gamma` is the dial: `0` gives you exactly plain HRP's weights, and turning it up toward `1` leans more on the full correlation structure, pulling the result toward the minimum-variance portfolio (the mix with the smallest overall wobble). `0.25`–`0.5` is a sensible middle; very high values can overshoot and become less stable, so they're best confirmed with a backtest.

*How it works:* It builds the same cluster tree as HRP and walks it the same way, but before each split it adjusts each group's risk estimate using the correlation *between* the two groups (this adjustment is the "Schur complement", scaled by gamma). If the two groups move together a lot, the adjustment recognises they are partly the same bet and shifts the split to avoid doubling down. At `gamma=0` the adjustment is switched off and you recover HRP's exact numbers; as gamma rises it folds in more of the between-group correlation, moving the weights toward the lowest-variance mix without ever fully abandoning HRP's diversification. Weights still always sum to 100%.

---

### Find the right Schur-HRP gamma for your ETFs

```bash
hierofolio analyze backtest --method schur-hrp --gamma 0.0   # equivalent to plain HRP
hierofolio analyze backtest --method schur-hrp --gamma 0.5
hierofolio analyze backtest --method schur-hrp --gamma 1.0
```

*Why:* How much you should trust the between-cluster correlations depends on your specific ETFs and how noisy those correlations are — there's no universally best gamma. Because `gamma=0` is exactly plain HRP, this sweep doubles as a direct test of whether folding in the extra correlation information actually helped out-of-sample or just added noise. If a middle gamma gets the best out-of-sample [Sharpe](https://en.wikipedia.org/wiki/Sharpe_ratio) with similar stability, that's your sweet spot; if `gamma=0` wins, plain HRP was already enough.

*How it works:* Each run steps through the identical rebalance dates and training windows — the only thing that changes is the gamma handed to the optimizer at each rebalance. Higher gamma folds more between-cluster correlation into every split (see the Schur-HRP allocation example above), so the weights, and therefore the out-of-sample returns, differ between runs. Comparing the resulting Sharpe ratios and drawdowns tells you which gamma survived real, unseen data best — not just which looked best in-sample.

---

### Find out if this actually worked in the past

```bash
hierofolio analyze backtest --method hrp
```

*Why:* The `allocate` output is in-sample — it's computed on the same data it's judged on, so it looks better than reality. The backtest is honest: it repeatedly trains on old data and measures what actually happened next. The out-of-sample [Sharpe](https://en.wikipedia.org/wiki/Sharpe_ratio) and max drawdown are what you'd have experienced as a real investor. It also shows an equal-weight benchmark so you can see whether the optimizer added value at all.

*How it works:* It slides a 3-year window forward through time in quarterly steps. At each step it fits the optimizer on the returns inside that window, then records what those weights would have returned over the *following* quarter — data the optimizer never saw. All those quarterly results are chained together into one equity curve. An equal-weight portfolio (same amount in every ETF, never optimised) is computed in parallel as a baseline.

---

### Compare HRP against MVO

```bash
hierofolio analyze backtest --method hrp
hierofolio analyze backtest --method mvo --max-weight 0.5
```

*Why:* HRP needs no return forecast and is hard to overfit; MVO uses past returns as a signal and can overfit if the window is wrong. Running both on the same data tells you whether MVO's extra complexity actually pays off for your specific universe. In all-equity portfolios, HRP often matches or beats MVO out-of-sample.

*How it works:* Both runs step through the exact same rebalance dates and training windows, so the only difference is which optimizer produces the weights each quarter. The out-of-sample return for each quarter is calculated identically, making the final Sharpe ratios and drawdowns directly comparable.

---

### See how transaction costs affect your returns

```bash
hierofolio analyze backtest --broker degiro --portfolio-size 15000
hierofolio analyze backtest --broker traderepublic --portfolio-size 5000
```

*Why:* Rebalancing costs money. A strategy that looks great before costs can look mediocre after them — especially with small portfolios or flat-fee brokers like Trade Republic where €1 per trade is a large fraction of a small position. This shows you the real cost drag and whether quarterly rebalancing is worth it at your portfolio size.

*How it works:* At each rebalance, the code computes how much each ETF's weight changed, converts that to a euro amount (weight change × portfolio size), and charges `flat fee + basis points × euros traded` per ETF. That cost is subtracted from the first day's return of the new quarter. The flat fee is charged once per ETF traded per rebalance — so a 5-ETF portfolio with DEGIRO pays up to 5 × €1 = €5 at each rebalance.

---

### Check whether rebalancing less often saves costs

```bash
hierofolio analyze backtest --broker degiro --portfolio-size 10000 --step 3   # quarterly
hierofolio analyze backtest --broker degiro --portfolio-size 10000 --step 12  # annually
```

*Why:* More frequent rebalancing keeps your weights accurate but costs more. Annual rebalancing is cheaper but lets the portfolio drift. This comparison tells you where the trade-off lands for your broker and portfolio size — often annual rebalancing is competitive with quarterly once costs are included.

*How it works:* `--step 3` generates a rebalance date every 3 months; `--step 12` every 12 months. Fewer rebalances means the cost formula runs fewer times, so the total cost drag is smaller. It also means the optimizer re-fits its weights less often, so it reacts more slowly to changing market conditions. The two runs let you weigh cheaper trading against staler weights for your specific numbers. (Note: within each hold period the backtest keeps the target weights fixed each day, so it measures the cost and re-fit effects, not real-world weight drift.)

## Risk model

`hierofolio.risk_model` provides `HRPRiskModel` plus the `ConstrainedMVOOptimizer` and
`RobustOptimizer` portfolio optimizers. Feed it the returns from the DB:

```python
from hierofolio.analyze import read_returns
from hierofolio.risk_model import HRPRiskModel

returns = read_returns("data/hierofolio.db")

risk_model = HRPRiskModel(
    returns=returns,
    shrinkage_method="constant_correlation",
    shrinkage_intensity=0.3,
    cluster_mode="full",
    linkage_method="ward",
)
```


## Tests

```bash
pytest                       # whole suite
pytest tests/test_riskmodel.py   # a single module
```

## FAQ

### If I run `hierofolio fetch` again, what happens?
Each ISIN is only re-downloaded when its newest stored date isn't today
(`days_behind > 0`). If it's already current, that ISIN loads from the DB
(cache) with no network call. Otherwise the fetch is *incremental*: it
requests only the range *after* the last stored date (last date + 1 → now)
and inserts the new days, leaving existing rows untouched. If nothing new is
available upstream, the already-stored data is kept.

### Why doesn't a re-run pull the whole history again?
It used to. Now the download window starts at the ISIN's last stored date, so
run #2 issues a tiny request instead of re-pulling years of data. `--force` is
the exception — it re-downloads the full range from the start date and
overwrites existing rows (use it to pick up restated prices).

### How far back does the data go?
The default `DEFAULT_START_DATE` is `2000-01-01` (in `etf_common.py`), which is
earlier than any UCITS ETF — so the first fetch simply returns each ETF's full
history from its inception (e.g. IWDA from 2009-09-28), no inception date
needed. `--start` is an optional *cap* for when you want **less** history:
`hierofolio fetch --start 2015-01-01` fetches only from 2015. Because fetches are
incremental, `--start` only applies on an ISIN's *first* fetch — to re-cut the
range for already-stored data, add `--force`
(`hierofolio fetch --start 2015-01-01 --force`).

### I fetched full history, so why are there fewer combined `Observations` than any single ETF has rows?
The aligned matrix can only start where *all* ETFs have data, i.e. at the
latest inception among them. If your youngest fund launched in 2014, the
combined panel begins in 2014 even though older funds go back further — the
pre-2014 dates are dropped because the young fund is `NaN` there. So the
combined count tracks your newest holding; dropping it pushes the common start
(and the observation count) back.

### What if I run it again after the trading day completes?
It picks up the new day. Once the newest stored date is behind "today", the
next run re-fetches and the upsert appends the newly published close. (Before
the change to `days_behind > 0`, a 7-day gate meant a plain run could skip new
days for up to a week — `--force` was the workaround.)

### Why is there no data for today's date?
The source only returns *completed* trading days, so today's close isn't
available until after the market closes. Days with no source data are never
stored as empty/placeholder rows — they simply don't appear.

### What are "data points" / "trading days" — are they days?
Yes. One row per day the market was open. Weekends and exchange holidays are
absent, so ~252 rows per year, one `close` each.

### What does `Observations: N` mean?
It's the row count of the *combined* price matrix across all ETFs, not one
fund. Dates are the union of every ETF's trading days, forward-filled across
gaps, then rows still missing any ETF are dropped. Because different exchanges
have different holiday calendars, this union can exceed any single ETF's count.

### Why do I get the same output with and without `--force`?
The summary reports the final data, which is the same either way. `--force`
only (a) re-fetches even when an ISIN is already current to today, and
(b) overwrites existing rows instead of keeping them — visible only if the
source restated a past close. Same prices in the DB → same output.

### How is a security resolved — why not just the ticker?
By **ISIN**, not the ticker. OpenFIGI maps the ISIN to a ticker, but searching
ftgo by that ticker and taking the first match can return the wrong security
(a real case: `CSSPX` matched a Cohen & Steers realty fund instead of the
iShares Core S&P 500). So `hierofolio fetch` searches ftgo by the ISIN, and pins
the chosen listing's `{xid, symbol, currency}` in `data/currency_metadata.yaml`.
Pinning matters because FT Markets search ordering isn't stable — without it a
later run could silently switch to a different listing/currency.

### Does an ISIN have one currency?
No. An ISIN identifies one share class with a fixed *NAV/base* currency, but
that security is cross-listed on several exchanges, each quoting in its own
currency (e.g. IE00B5BMR087 trades as `CSPX:LSE:USD`, `CSP1:LSE:GBX`,
`SXR8:GER:EUR`). The **quote** currency depends on the listing, not the ISIN.
We store one listing per ISIN, and prefer the listing whose currency matches
the fund's own base currency (parsed from its name, e.g. "… USD (Acc)" → the
USD line), falling back to the first match. The choice is recorded in
`data/currency_metadata.yaml`.

### Does HRP favour low-volatility assets?
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

### Can I mix currencies across ETFs in the analysis?
Not meaningfully. Per-asset daily returns are currency-invariant, but a
covariance/HRP across assets is only coherent if every series is in the same
currency (or FX-converted). `hierofolio analyze` reads `data/currency_metadata.yaml` and
prints a `⚠` warning if the ISINs in a `returns`/`summary` run don't share a
currency. Preferring each fund's base currency keeps a same-index panel (e.g. a
book of USD-class ETFs) consistent; genuinely mixing currencies needs FX
conversion, which isn't implemented.

## License

MIT — see [LICENSE](LICENSE).
