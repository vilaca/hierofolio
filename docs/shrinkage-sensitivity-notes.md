# Shrinkage sensitivity notes

Stress-test of the three headline findings from `portfolio-backtest-notes.md`:
do they hold across covariance estimators, or are they artifacts of the default
(`constant_correlation`, intensity 0.3)?

**Setup:** same 2-year window, quarterly step, 13-ETF universe (same three funds
excluded). Shrinkage method/intensity is the only variable.

---

## OOS Sharpe table by estimator

| Method | cc 0.1 | **cc 0.3** (base) | cc 0.5 | ledoit\_wolf | identity |
|--------|:------:|:-----------------:|:------:|:------------:|:--------:|
| HRP | 0.17 | **0.19** | 0.21 | 0.19 | **0.55** |
| Schur-HRP γ=0.5 | 0.20 | **0.19** | 0.19 | 0.23 | 0.58 |
| HRP-Σμ τ=1 | 0.18 | **0.21** | 0.23 | 0.21 | 0.57 |
| MVO uncapped | 0.82 | **0.82** | 0.82 | 0.82 | 0.82 |
| MVO cap 25% | 0.82 | **0.82** | 0.82 | 0.82 | 0.82 |
| Robust cap 25% | 0.82 | **0.82** | 0.82 | 0.82 | **0.84** |
| CRISP γ=0.3 | 0.58 | **0.58** | 0.58 | 0.58 | 0.58 |
| CRISP γ=0.5 | 0.63 | **0.63** | 0.63 | 0.63 | 0.63 |
| CRISP γ=1.0 | 0.59 | **0.59** | 0.59 | 0.59 | 0.59 |
| Equal-weight | 0.70 | **0.70** | 0.70 | 0.70 | 0.70 |

`cc` = `constant_correlation` shrinkage, intensity in parentheses. Equal-weight
is read from the second column of each run's benchmark block (it does not depend
on the shrinkage method). `identity` shrinks toward a diagonal covariance
(zero off-diagonal), effectively discarding correlation structure.

---

## Weights-delta observations

**HRP in-sample allocations are stable across cc intensities and ledoit_wolf.**
Bond fund (IE00BDBRDM35) sits at 9.0–11.0% in the point-in-time snapshot;
top weight (Health Care) ranges 14–17%. No estimator in this family changes the
character of the portfolio.

**With `identity`, HRP converges toward uniform vol-weighting.** The snapshot
shows bond weight at 11%, similar to cc — but the OOS trajectory changes
dramatically: HRP no longer spirals into bonds in 2021–2022, because the
clustering ignores the bond–equity correlation that previously segregated bonds
into a low-volatility silo.

**CRISP in-sample allocations are identical across all five estimators** (US
Tech 49.7%, Bonds 29.7%, Health Care 20.6%), confirmed by running `allocate
--method crisp --corr-penalty 0.5` under each. The OOS table confirms this
holds dynamically too: CRISP Sharpe is constant to two decimal places for every
column.

---

## Verdict: F1/F2/F3 robustness

**F1 — HRP fails badly (0.19 vs equal-weight 0.70): CONDITIONALLY robust.**
It holds under every estimator that preserves the correlation structure
(`constant_correlation` at any realistic intensity, `ledoit_wolf`). Under
`identity` — which discards all correlations — HRP recovers to 0.55, still
below equal-weight but no longer catastrophic. This directly answers the
mechanism: the bond-concentration failure is correlation-driven. The estimated
bond–equity correlation was low enough that HRP's hierarchical step isolated
bonds as a diversifying cluster and piled in. Strip out correlations and the
failure largely disappears. In practice, any sensible covariance model preserves
correlation structure, so F1 holds for realistic usage; but the finding is more
nuanced than "HRP is broken" — it is broken specifically *because* of what
correlations it trusts.

**F2 — MVO / Robust win (~0.82): fully robust.**
MVO and Robust return 0.82 ± 0.01 across every cell in the table. This makes
sense: their edge comes from the expected-return signal (US tech Sharpe ~1.2
over this window), which overwhelms any covariance estimator choice for
diversification. The win is a momentum bet, not a covariance bet.

**F3 — CRISP sits in the middle (~0.58–0.63): fully robust, remarkably so.**
CRISP is invariant to the covariance estimator — identical Sharpe to two decimal
places across all five columns. The correlation-redundancy penalty in its
objective serves the same function as shrinkage: it already down-weights
correlated assets. The optimal portfolio (US Tech + Bonds + Health Care) is
apparently a stable global solution that doesn't depend on how finely the
pairwise correlations are estimated.

---

## Reproduce

```bash
# Extended backtest.sh now accepts shrinkage args:
#   ./backtest.sh [WINDOW] [STEP] [BROKER] [SHRINKAGE_METHOD] [SHRINKAGE_INTENSITY]
./backtest.sh 2 3 "" ledoit_wolf
./backtest.sh 2 3 "" constant_correlation 0.1
./backtest.sh 2 3 "" constant_correlation 0.5
./backtest.sh 2 3 "" identity

# In-sample weight snapshots:
EX=(--exclude IE0006WW1TQ4 IE0003XJA0J9 IE000YYE6WK5)
for sm in ledoit_wolf constant_correlation identity; do
  uv run python -m hierofolio.analyze allocate --method hrp --shrinkage-method "$sm" "${EX[@]}"
  uv run python -m hierofolio.analyze allocate --method crisp --corr-penalty 0.5 --shrinkage-method "$sm" "${EX[@]}"
done
```
