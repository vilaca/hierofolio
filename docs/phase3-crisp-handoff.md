# Handoff — Phase 3: CRISP (correlation-regularized iterative shrinkage)

Implementation handoff for the final optimizer on the roadmap
(HRP → Schur-HRP → HRP-Σμ → **CRISP**). Phases 0–2 are implemented and merged.
This document is self-contained: it can be executed in a fresh session.

CRISP = **C**orrelation-**R**egularized **I**terative **S**hrinkage
**P**ortfolios: signal-aware like MVO, but it penalizes *redundant correlated
bets* on top of variance, so tiny differences in noisy return forecasts don't
create huge concentration. It is the robust-signal-aware rung between HRP-Σμ and
distributionally-robust methods.

## Session settings (required)

| Sub-step | Model / effort | Why |
|---|---|---|
| Steps 1–2 (`CRISPOptimizer` + engine w₀ wiring + tests) | **Opus 4.8, xhigh** | Silent-correctness surface: the γ=0,τ=0≡MVO parity depends on matching the exact `−λ/2` objective convention; the turnover-penalty semantics change engine behavior; PSD handling of both Σ and C is required for a convex program. A wrong sign or a missing ½ degrades the portfolio without erroring. |
| Steps 3–4 (CLI wiring + docs) | **Sonnet 4.6, high** | Mechanical, mirrors the `--gamma`/`--tau` wiring already in place. |

**Escalation trigger:** if the γ=0,τ=0≡MVO parity test won't converge to
`np.allclose`, or if increasing γ does *not* reduce concentration, stop and treat
it as a sign the objective is subtly wrong before adding more code.

`/clear` is appropriate before starting; the only input needed is this document.

## Guiding constraint (read first)

Same rule as Phases 0–2: **no existing public signature changes.** The one
behavior change is additive and CRISP-only (Step 2, engine w₀ wiring), and must
leave every other allocator's output identical. Acceptance gate: the
γ=0,τ=0≡MVO parity test plus the entire existing suite green with zero edits.

## What Phases 0–2 already give you (real, implemented)

- **`PortfolioOptimizer` base** (`src/hfolio/risk_model.py:641`):
  `__init__(risk_model, alpha, current_weights=None)` validates the alpha index
  and stores `current_weights` (`:648-657`). CRISP subclasses this exactly like
  `ConstrainedMVOOptimizer` and `RobustOptimizer` do.
- **MVO objective to match** — `ConstrainedMVOOptimizer` uses
  `alpha@w − risk_aversion·0.5·quad_form(w, Σ_psd)` with `_project_to_psd`
  applied to the covariance, constraints `sum(w)==1, w≥0, w≤max_weight`, and a
  soft turnover as `0.5·‖w−w₀‖₁ ≤ turnover_limit` (`risk_model.py:697-762`).
- **`risk_model.correlation()`** (`risk_model.py:354`) returns the (clipped,
  symmetric) correlation matrix — this is CRISP's penalty matrix **C**.
- **`_project_to_psd`** (`risk_model.py:15`) — reuse for both Σ and C so
  `cp.quad_form` stays convex.
- **Adapter + registry pattern** — `MVOAllocator`/`RobustAllocator`
  (`allocators.py:213`-ish) construct the optimizer and call `.solve()`;
  `ALLOCATORS` dict (`allocators.py:301`) is the single dispatch point used by
  the CLI (`analyze.py:480`) and `run_backtest` (`analyze.py:238`).
- **Engine tracks prior weights** — `WalkForwardEngine` keeps `prev_weights`
  (`backtest.py:176,231`) for cost accounting but currently passes
  `current_weights=None` into `allocate` (`backtest.py:198`). Step 2 connects it.

---

## Step 1 — `CRISPOptimizer`

**The objective** (single convex solve — the "iterative" refinement is a
deliberate follow-up, see Out of scope):

```
max_w   μᵀw − (λ/2)·wᵀΣw − γ·wᵀC w − τ·‖w − w₀‖₁
s.t.    sum(w) == 1,  w ≥ 0,  w ≤ max_weight
```

- **μ** = `alpha` (the signal), same as MVO/Robust.
- **Σ** = `risk_model.covariance()`, PSD-projected.
- **C** = `risk_model.correlation()`, PSD-projected. This term penalizes
  weight placed on mutually-correlated assets — the distinctive CRISP force.
- **λ** = `risk_aversion` (reuse the existing knob).
- **γ** = `corr_penalty` — redundancy-aversion strength.
- **τ** = `turnover_penalty` — soft L1 turnover **penalty** (not the hard
  `turnover_limit` constraint the other optimizers use). Active only when
  `current_weights` (w₀) is provided; with w₀ absent the term drops out.

**The parity convention is load-bearing.** Use `−(λ/2)·quad_form(w, Σ)` — the
*exact* MVO form — so that **γ=0, τ=0 ⟹ CRISP ≡ ConstrainedMVO bit-for-bit**.
That is the hard regression anchor, mirroring τ=0≡HRP (Phase 2) and γ=0≡HRP
(Phase 1). Do **not** drop the ½ or the objective silently diverges from MVO.

**Implementation** in `risk_model.py`, structured like `ConstrainedMVOOptimizer`
(`risk_model.py:672`):

```python
class CRISPOptimizer(PortfolioOptimizer):
    """Correlation-regularized signal-aware optimizer.

    max μᵀw − (λ/2)wᵀΣw − γ·wᵀC w − τ·‖w−w₀‖₁
    γ=0, τ=0 reduces exactly to ConstrainedMVOOptimizer.
    """
    def __init__(self, risk_model, alpha, current_weights=None,
                 risk_aversion=1.0, corr_penalty=0.3,
                 turnover_penalty=0.0, psd_eps=1e-10):
        super().__init__(risk_model, alpha, current_weights)
        ...

    def solve(self, max_weight=0.20, **kwargs) -> pd.Series:
        assets = self.alpha.index.tolist()
        Sigma = _project_to_psd(self.risk_model.covariance()
                                .reindex(index=assets, columns=assets).values, self.psd_eps)
        C = _project_to_psd(self.risk_model.correlation()
                            .reindex(index=assets, columns=assets).values, self.psd_eps)
        w = cp.Variable(len(assets), nonneg=True)
        obj = (self.alpha.values @ w
               - self.risk_aversion * 0.5 * cp.quad_form(w, Sigma)
               - self.corr_penalty * cp.quad_form(w, C))
        if self.turnover_penalty > 0 and self.current_weights is not None:
            w0 = self.current_weights.reindex(assets).fillna(0).values
            obj = obj - self.turnover_penalty * cp.norm(w - w0, 1)
        constraints = [cp.sum(w) == 1.0]
        if max_weight is not None:
            constraints.append(w <= max_weight)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if w.value is None:
            raise RuntimeError("CRISP optimization failed to converge")
        return pd.Series(w.value, index=assets)
```

Note the same asset-ordering reindex the other optimizers use — required for the
order-invariance regression that `test_mvo_is_order_invariant` guards.

> **C-matrix choice:** use `correlation()` as-is (diagonal = 1). Its diagonal
> contributes a mild L2 concentration penalty, which is desirable and keeps C
> PSD. Zeroing the diagonal (pure cross-correlation penalty) is a valid variant
> — leave a one-line comment noting it, but default to as-is.

**Adapter** in `allocators.py`, mirroring `MVOAllocator`:

```python
class CRISPAllocator:
    def allocate(self, risk_model, signal=None, current_weights=None,
                 risk_aversion=1.0, corr_penalty=0.3, turnover_penalty=0.0,
                 max_weight=None, **params) -> pd.Series:
        return CRISPOptimizer(
            risk_model=risk_model, alpha=signal, current_weights=current_weights,
            risk_aversion=risk_aversion, corr_penalty=corr_penalty,
            turnover_penalty=turnover_penalty,
        ).solve(max_weight=max_weight)
```

Register `"crisp": CRISPAllocator()` in `ALLOCATORS` (`allocators.py:301`).

---

## Step 2 — Engine w₀ wiring (CRISP-only behavior change)

For the turnover penalty to mean anything in a backtest, CRISP needs the prior
fold's weights as w₀. The engine already has them as `prev_weights` but passes
`current_weights=None` (`backtest.py:198`). Change that one call to pass
`current_weights=prev_weights`.

**Safety:** HRP, Schur-HRP, and HRP-Σμ ignore `current_weights`; `MVOAllocator`
and `RobustAllocator` do not forward it to their optimizers. So only CRISP
observes this — every other allocator's output is unchanged. Add a test
asserting an HRP backtest is byte-identical before/after the wiring (or simply
rely on the existing `run_backtest` HRP tests staying green).

---

## Step 3 — CLI wiring

Mirror the `--gamma`/`--tau` pattern already in `analyze.py`:

1. Add `crisp` to `--method` choices in both parsers (`analyze.py:294`, `:336`).
2. **New flags** on both `allocate` and `backtest`:
   `--corr-penalty` (γ, default 0.3) and `--turnover-penalty` (τ, default 0.0).
   **Reuse `--risk-aversion` for λ** (it already exists).
   *Do not overload `--gamma` (Schur-HRP) or `--tau` (HRP-Σμ)* — those mean
   different things; distinct names prevent semantic collision even though flags
   are method-scoped.
3. `allocate` dispatch (`analyze.py:478-480`): add `corr_penalty`,
   `turnover_penalty` to `method_params` for `method == "crisp"`.
4. `run_backtest` (`analyze.py:190`): add `corr_penalty` / `turnover_penalty`
   params and a `if method == "crisp":` branch in the `method_params` block
   (near `analyze.py:232`); pass them from the CLI call (near `analyze.py:555`).

---

## Step 4 — Tests

**`tests/test_riskmodel.py`** (mirror the MVO/Robust optimizer tests):
- **γ=0, τ=0 ≡ MVO** — `CRISPOptimizer(rm, alpha, corr_penalty=0).solve(max_weight=0.5)`
  is `np.allclose` to `ConstrainedMVOOptimizer(rm, alpha).solve(max_weight=0.5)`.
  Hard anchor.
- **Concentration falls with γ** — on a synthetic with a correlated high-signal
  cluster, raising γ (0 → 0.5 → 1.0) monotonically lowers the max weight / a
  Herfindahl concentration measure. (Matches the CRISP γ-knob intuition.)
- **Turnover penalty pulls toward w₀** — with `current_weights=w₀` given,
  higher τ yields lower `0.5·‖w−w₀‖₁`.
- **Order-invariance** — shuffled alpha index gives the same per-asset weights.
- Valid weights: sum to 1, non-negative, `≤ max_weight`.

**`tests/test_allocators.py`:** `CRISPAllocator` protocol conformance +
`np.allclose` to a direct `CRISPOptimizer` call.

**`tests/test_backtest.py`:** `run_backtest(method="crisp", ...)` produces a
valid OOS series and a log with weights summing to 1; an HRP backtest is
unchanged by the Step 2 wiring.

All pre-existing tests stay green **unedited**.

---

## Step 5 — Validation (the actual research question)

Once CRISP is in the registry, the walk-forward engine can compare all six
methods on the same folds:

```
hfolio analyze backtest --method {hrp,schur-hrp,hrp-sigma-mu,mvo,robust,crisp}
```

The meaningful question is not in-sample fit but **out-of-sample Sharpe and
weight stability**: does CRISP beat HRP-Σμ / Robust MVO out-of-sample without
higher turnover? Sweep `--corr-penalty` (0.2–0.5 is the expected sweet spot).
This validation is analysis, not code — capture findings in the README or a
notes file, not the test suite.

---

## Definition of done

- `CRISPOptimizer` in `risk_model.py`; `CRISPAllocator` in `allocators.py`,
  registered as `"crisp"`.
- Engine passes `current_weights=prev_weights` (CRISP-only effect).
- `--method crisp`, `--corr-penalty`, `--turnover-penalty` on `allocate` and
  `backtest`; λ via existing `--risk-aversion`.
- New tests pass, including the γ=0,τ=0≡MVO anchor and the γ-concentration test.
- `pytest` fully green, every pre-existing test unedited.

## Out of scope (future)

- **Iterative shrinkage** — the true CRISP alternates solve → shrink correlated
  exposures → repeat (proximal / reweighted). Ship the single convex solve
  first; add the reweighting loop only if the one-shot underperforms in Step 5.
- **Wasserstein-DRO** — the distributionally-robust rung beyond CRISP.
- **Schur-Σμ** — signal tilt on Schur-augmented blocks (noted in Phase 2).
