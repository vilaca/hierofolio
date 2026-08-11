# Portfolio backtest notes

Out-of-sample backtest of a real 16-ETF portfolio using the walk-forward engine.
Reproduce with `analyse.sh` (allocations) or the commands at the bottom.

**Universe:** 13 ETFs with ≥ 2019 history. Three recently-launched funds are
excluded (`--exclude`) because their short history would otherwise drag the
common return window to 2024:

| Excluded ISIN | Fund | History from |
|---|---|---|
| IE0006WW1TQ4 | Xtrackers MSCI World ex-USA | Mar 2024 |
| IE0003XJA0J9 | Amundi Prime All Country World | Jun 2024 |
| IE000YYE6WK5 | VanEck Defense | Apr 2023 |

**Setup:** 2-year training window, quarterly rebalance, walk-forward OOS window
**2021-10 → 2026-08 (~4.8y)**. No costs unless noted.

---

## Method comparison (out-of-sample)

| Method | Sharpe | Ann Ret | Ann Vol | Max DD |
|--------|:------:|:-------:|:-------:|:------:|
| MVO (uncapped) | **0.82** | 17.4% | 21.3% | −27.0% |
| MVO (cap 25%) | 0.82 | 14.9% | 18.2% | −24.8% |
| Robust (cap 25%, ρ=10) | 0.82 | 14.5% | 17.7% | −23.3% |
| **Equal-weight (benchmark)** | **0.70** | 9.9% | 14.1% | −20.5% |
| CRISP γ=0.5 | 0.63 | 9.6% | 15.2% | −19.4% |
| CRISP γ=1.0 | 0.59 | 8.3% | 14.0% | −16.6% |
| CRISP γ=0.3 | 0.58 | 9.9% | 17.0% | −24.1% |
| HRP-Σμ (τ=1) | 0.21 | 3.6% | 17.7% | −16.9% |
| Schur-HRP (γ=0.5) | 0.19 | 3.5% | 18.2% | −17.1% |
| HRP | 0.19 | 3.4% | 18.1% | −16.6% |

Transaction costs (DEGIRO, €10k portfolio) are modest — total drag ~−0.56% over
the period — and do not change the ranking. HRP 0.19→0.16, MVO 0.82→0.81,
CRISP γ=0.5 0.63→0.62.

---

## Findings

**1. HRP failed badly — worse than doing nothing (0.19 vs equal-weight 0.70).**
HRP's inverse-variance weighting put **78–84% into the Global Aggregate Bond
fund in 2021–2022**, because bonds had low trailing volatility during the
zero-rate era. It rode that concentration straight into the 2022 bond crash,
cut bonds to ~5% in 2023 *after* the damage, then piled back to ~76% by 2025 as
the crash rolled out of the trailing window. Textbook buy-high-sell-low, driven
by a volatility-regime shift the risk model can't anticipate. Schur-HRP and
HRP-Σμ inherit the same weakness (they share the inverse-variance core).

**2. MVO / Robust won — but it is a tech-momentum bet.** They concentrated into
US Technology (Sharpe ~1.2 over the window, riding the AI rally) and it kept
working. The cost is the highest drawdowns in the table (−25 to −27%). Classic
MVO behaviour: excellent while the trend persists, exposed when it reverses.
The win says as much about the 2021–2026 regime as about the method.

**3. Naive equal-weight beat HRP, Schur-HRP, HRP-Σμ, and every CRISP variant.**
Only signal-following (MVO/Robust) beat it, and only with materially higher tail
risk. CRISP sits sensibly in the middle: better than the HRP family, lower
drawdown than MVO, tracking near equal-weight.

---

## Caveats — do not over-read this

- **One regime.** 2021–2026 combines the 2022 bond crash and the tech/AI
  melt-up — it maximally punishes bond-holders and rewards tech-momentum. A
  different window could invert the ranking. This is not a general verdict on
  the methods, only on how they behaved in this specific period.
- **Mixed currencies** (USD/EUR/GBX) are not FX-converted — the covariance is
  not strictly coherent (the tool warns).
- **Bond fund data** was weekly in 2019–2020, thinning the common panel (~219
  vs 252 obs/year) and adding noise to the earliest training folds.
- **Short OOS** (~4.8y, ~20 quarterly folds). Not enough to be conclusive.
- Duplicate exposures inflate some raw weights: three S&P 500 trackers, two
  European funds, and two US-tech funds each split budget that economically is
  one position. See the correlation matrix in `hfolio analyze summary`.

---

## Reproduce

```bash
EXCLUDE="--exclude IE0006WW1TQ4 IE0003XJA0J9 IE000YYE6WK5"

# HRP — note the bond-concentration swings in the rebalance log
hfolio analyze backtest --method hrp $EXCLUDE --window 2 --step 3

# MVO — tech-momentum concentration
hfolio analyze backtest --method mvo --max-weight 0.25 $EXCLUDE --window 2 --step 3

# CRISP — the middle ground
hfolio analyze backtest --method crisp --corr-penalty 0.5 $EXCLUDE --window 2 --step 3

# With transaction costs
hfolio analyze backtest --method mvo --max-weight 0.25 $EXCLUDE \
  --window 2 --step 3 --broker degiro --portfolio-size 10000
```

Note: pass the ISINs to `--exclude` literally, not via an unquoted shell
variable — `nargs='+'` needs them as separate words.
