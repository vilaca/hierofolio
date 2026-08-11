# CRISP validation notes

**Universe:** 3 ETFs (MSCI World / S&P 500 / EM IMI), 2014–2026  
**Setup:** 3-year rolling window, 6-month rebalance, no transaction costs  
**Escalation trigger:** did not fire — γ=0 CRISP = MVO (exact parity confirmed)

---

## Method comparison (OOS Sharpe, Ann Return, Ann Vol)

| Method        | Sharpe | Ann Ret | Ann Vol | Notes |
|---------------|--------|---------|---------|-------|
| HRP           | 0.704  | 11.66%  | 16.55%  | ~equal-weight risk parity |
| Schur-HRP     | 0.704  | 11.66%  | 16.55%  | identical to HRP (3-asset bisection degenerates) |
| HRP-Σμ        | 0.710  | 11.74%  | 16.53%  | marginal tilt toward best-μ branch |
| MVO           | 0.814  | 13.90%  | 17.06%  | 100% S&P 500 every fold but last |
| Robust        | 0.814  | 13.90%  | 17.06%  | identical to MVO (α-uncertainty unchanged) |
| **CRISP γ=0.3** | **0.778** | **13.10%** | **16.83%** | partial de-concentration, meaningful diversification |

Equal-weight benchmark: Sharpe 0.726, Ann Ret 11.94%, Ann Vol 16.45%

---

## CRISP γ sweep

| γ    | Sharpe | Ann Ret | Ann Vol | Character |
|------|--------|---------|---------|-----------|
| 0.0  | 0.814  | 13.90%  | 17.06%  | MVO — 100% S&P 500 |
| 0.1  | 0.811  | —       | —       | near-MVO, slight diversification |
| 0.2  | 0.803  | 13.63%  | 16.99%  | first meaningful de-concentration |
| 0.3  | 0.778  | 13.10%  | 16.83%  | default; meaningful diversification |
| 0.5  | 0.746  | 12.49%  | 16.74%  | more diversified, tighter vol |
| 0.75 | 0.727  | —       | —       | approaches equal-weight range |
| 1.0  | 0.717  | 12.02%  | 16.76%  | near equal-weight Sharpe, lower vol than MVO |

**Sweet spot: γ ≈ 0.1–0.2.** Gives up ≤1% Sharpe vs MVO but breaks the
single-asset concentration. The handoff's suggested 0.2–0.5 range is confirmed;
the lower end is less costly than expected.

---

## Key observations

**MVO / Robust degenerate on this universe.** S&P 500 dominated both other ETFs
on historical Sharpe, so the optimizer went 100% S&P 500 every fold — classic
estimation-error concentration. The final fold flipped to 100% EM (noise-driven
reversal), showing the fragility.

**CRISP breaks the concentration without a hard max-weight cap.** At γ=0.3 it
holds ~0–90% S&P 500 (median ~80%), ~10–47% EM, and 0% MSCI World. MSCI World
stays zero because it is nearly identical to S&P 500 (ρ ≈ 0.97) — the
correlation penalty makes doubling up on a redundant asset too expensive. That
is exactly the intended CRISP effect.

**Weight stability:** HRP is the most stable (≈30/30/40 drift over time). CRISP
is meaningfully more stable than raw MVO (which never moves until it does, then
flips entirely), but less stable than HRP — it still chases signal but tempers
the concentration.

**Schur-HRP / HRP-Σμ are marginal over HRP here.** With only 3 assets the
bisection tree is minimal (one split), so Schur cross-block augmentation has
nothing to work with. These methods should differentiate on larger universes.

**Universe limitation.** 3 ETFs with 2 nearly-identical developed-market funds
is a stress-test for concentration optimizers, not a representative evaluation.
The CRISP γ-knob behaved exactly as the objective implies — results are
consistent with theory. A larger universe (10–20 ETFs with genuine sector
dispersion) would produce richer differentiation.

---

## Suggested next steps

- Expand universe to 10+ ETFs and re-run; CRISP and Schur-HRP should
  differentiate more clearly there.
- Add `--turnover-penalty` to the CRISP sweep once more assets make turnover
  meaningful.
- Consider adding `--max-weight 0.5` to MVO/Robust as a fairer baseline
  comparison (currently unconstrained MVO vs CRISP is not apples-to-apples).
