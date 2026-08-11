# Handoff — Phase 0 (infrastructure) + Schur-HRP

Implementation handoff for the first slice of the optimizer roadmap
(walk-forward engine · signal models · Schur-HRP · HRP-Σμ · CRISP). This
document is self-contained: it can be executed in a fresh session.

## Session settings (required)

This handoff spans **two** model picks. Verify and switch before each phase; do
not start work until the setting for the phase you are in is confirmed.

| Sub-step | Model / effort | Why |
|---|---|---|
| 0a, 0b, 0c | **Sonnet 4.6, high** | Routine, well-specified; guarded by parity tests. |
| Step 1 (Schur-HRP) | **Opus 4.8, xhigh** | Subtle cross-block linear algebra; γ=0/γ=1 invariants are the only guard against silently-wrong output. |

`/clear` is appropriate before starting, since the only input needed is this
document.

## Guiding constraint (read first)

`hrp_weights()` (`src/hierofolio/analyze.py:163`), `run_backtest()`
(`analyze.py:245`), and the `PortfolioOptimizer` classes in
`src/hierofolio/risk_model.py` all have tests pinning their current behavior
(`tests/test_allocator.py`, `tests/test_riskmodel.py`).

**No step may change an existing public signature.** New abstractions wrap the
old code; old entry points become thin shims. The acceptance gate for every
step is a *parity test*: new code reproduces the old numbers via `np.allclose`,
and the entire pre-existing suite stays green **with zero edits**.

## Architecture context

Two families of weight generators exist and currently share no interface:

- **Bisection** — `hrp_weights()`, a free function walking the dendrogram.
- **Convex program** — `ConstrainedMVOOptimizer` / `RobustOptimizer`
  (`PortfolioOptimizer` subclasses using cvxpy).

The CLI branches on method with `if method == 'hrp': ... else: optimizer.solve()`
in both `allocate` and `backtest`. Phase 0 introduces a single `Allocator`
protocol above both families so the engine and CLI have one uniform call, and
extracts the hardcoded alpha signal into a pluggable `SignalModel`.

---

## Step 0a — Signal models

**New file `src/hierofolio/signals.py`:**

```python
from typing import Optional, Protocol
import pandas as pd

class SignalModel(Protocol):
    def signal(self, returns: pd.DataFrame) -> tuple[pd.Series, Optional[pd.Series]]:
        """Return (mu, mu_uncertainty). uncertainty=None → optimizer default."""

class HistoricalMeanSignal:
    """Annualized historical mean — exact current behavior."""
    def signal(self, returns):
        mu = (1 + returns.mean()) ** 252 - 1
        return mu, None   # None keeps RobustOptimizer's sqrt(diag Σ) default
```

Returning `None` for uncertainty is deliberate: `RobustOptimizer` already falls
back to `sqrt(diag(cov))` when uncertainty is absent
(`risk_model.py:812`), so backtest numbers do not move.

**Wire-in:** replace the two hardcoded
`alpha = (1 + ...mean()) ** 252 - 1` lines (`analyze.py:301`, `analyze.py:546`)
with `mu, _ = signal_model.signal(train)`. Default
`signal_model = HistoricalMeanSignal()`.

**Acceptance (`tests/test_signals.py`):** `HistoricalMeanSignal().signal(r)[0]`
is `np.allclose` to the old formula on `make_returns()`.

---

## Step 0b — Allocator protocol

**New file `src/hierofolio/allocators.py`:**

```python
from typing import Optional, Protocol
import pandas as pd
from hierofolio.risk_model import RiskModel

class Allocator(Protocol):
    def allocate(self, risk_model: RiskModel,
                 signal: Optional[pd.Series] = None,
                 current_weights: Optional[pd.Series] = None,
                 **params) -> pd.Series: ...
```

Four implementations, chosen to touch existing code as little as possible:

| Allocator | Body |
|---|---|
| `HRPAllocator` | Bisection core extracted from `hrp_weights`. Ignores `signal`. |
| `SchurHRPAllocator` | See Step 1. |
| `MVOAllocator` | Thin adapter: constructs `ConstrainedMVOOptimizer(risk_model, signal, current_weights, ...)` and calls `.solve(**params)`. |
| `RobustAllocator` | Same, wrapping `RobustOptimizer`. |

The optimizer classes and their tests stay **untouched** — the adapters give
them the uniform call shape without a rewrite.

**Extract, don't move:** lift the bisection loop out of `hrp_weights` into
`HRPAllocator.allocate`, then make `hrp_weights(risk_model, verbose=False)` a
shim that calls it (keeping the verbose printing, which is presentation).
`test_allocator.py`'s `hrp_weights` tests pass unchanged.

**CLI dispatch:** replace the `if method == 'hrp' … else optimizer.solve()`
fork in `allocate` and `backtest` (`analyze.py:543`, `analyze.py:298`) with a
registry:

```python
ALLOCATORS = {"hrp": HRPAllocator(), "schur-hrp": SchurHRPAllocator(),
              "mvo": MVOAllocator(), "robust": RobustAllocator()}
weights = ALLOCATORS[method].allocate(risk_model, signal=mu,
                                      current_weights=None, **method_params)
```

**Acceptance (`tests/test_allocators.py`):** protocol conformance (weights sum
to 1, non-negative, indexed by assets); `HRPAllocator` output `np.allclose` to
`hrp_weights`; `MVOAllocator`/`RobustAllocator` `np.allclose` to direct
optimizer calls.

---

## Step 0c — Walk-forward engine

**New file `src/hierofolio/backtest.py`** with `WalkForwardEngine`,
parameterized by a risk-model factory, `SignalModel`, `Allocator`, and:

- **Window policy** — `rolling` (current:
  `train_start = rebal - window`) vs `anchored`/expanding
  (`train_start = idx[0]`).
- **Purge + embargo** — generalize the existing "exclude `rebal_date` itself"
  (`analyze.py:314`) into explicit `purge_days` / `embargo_days` between train
  end and test start.
- **Nested param selection (optional per run)** — `param_grid` + an inner
  walk-forward on the training window, picking the params with best inner-OOS
  Sharpe. This is what lets Schur-HRP's γ be chosen per fold instead of
  hardcoded. **Highest-correctness-risk sub-step** — it is easy to leak
  look-ahead into the inner loop; its test must assert the inner selection only
  ever sees training-window dates.

**Keep `run_backtest` working:** reimplement it as a wrapper over
`WalkForwardEngine` with `rolling`, `purge_days=1`, `embargo_days=0`, no param
grid — today's exact behavior. The cost model, equal-weight benchmark, and log
format (`analyze.py:320-341`) move into the engine unchanged.

**Acceptance (`tests/test_backtest.py`):**
- Engine-with-default-config reproduces `run_backtest` OOS series exactly.
- Purge/embargo removes the expected dates.
- Anchored vs rolling differ where expected.
- Param selection recovers the known-best γ on a synthetic where one γ dominates.
- **No-lookahead assertion** on the inner selection loop.
- All of `test_allocator.py`'s `run_backtest` tests stay green with **zero edits**.

---

## Step 1 — Schur-HRP  (switch to Opus 4.8, xhigh)

**The math.** Plain HRP's `cluster_var` (`analyze.py:168`) ignores the
covariance *between* the left and right sub-clusters. Schur-HRP (Cotton,
*Schur Complementary Portfolios*, 2022) folds that cross-block back in. At each
split, partition the ordered sub-cluster covariance:

```
Σ = [[A, B],      A = left-left,  D = right-right
     [Bᵀ, D]]     B = left-right cross-block
```

and allocate using **augmented** blocks scaled by γ ∈ [0, 1]:

```
A_aug = A − γ · B D⁻¹ Bᵀ        (Schur complement of D)
D_aug = D − γ · Bᵀ A⁻¹ B        (Schur complement of A)
```

- **γ = 0** → `A_aug = A`, `D_aug = D` → reduces *exactly* to current HRP.
- **γ = 1** → full Schur complement → recursion converges to global
  minimum-variance weights.

Schur-HRP is thus the continuous HRP ↔ min-variance dial, which is why it is the
ideal Phase-1 benchmark rung.

> **Correction (added post-implementation, 2026-08-11).** The "γ = 1 → global
> minimum-variance" claim above is **not exact** and was revised during
> implementation. A reference-faithful recursion (Peter Cotton / skfolio) keeps
> *inverse-variance* weights within clusters at every γ and only augments the
> covariance, so γ = 1 merely *approaches* min-variance (empirically 1e-2–3e-1
> off on clean covariances; on a 3-asset problem no augmentation fires at all,
> so γ = 1 == HRP). It is mathematically impossible for a single within-cluster
> rule to give HRP at γ = 0 **and** exact min-variance at γ = 1 — the two
> endpoints require different rules. The γ = 0 ≡ HRP anchor is unaffected. See
> the revised acceptance criteria below.

**`SchurHRPAllocator.allocate(risk_model, gamma=0.5, ...)`** in `allocators.py`,
reusing `risk_model.quasi_diagonalized_covariance` (already a property,
`risk_model.py:434`) for the ordered matrix. Correctness risks to handle
explicitly:

1. `A⁻¹` / `D⁻¹` stability — regularize via the existing `_project_to_psd`
   (`risk_model.py:15`) before inverting.
2. `A_aug` / `D_aug` can lose PSD — clip eigenvalues the same way.
3. The exact reference-completion detail in Cotton's recursion is subtle —
   **mirror Peter Cotton's reference implementation** (`precise` /
   `schurComplementary`) rather than reinventing it; let the boundary invariants
   below be the proof of correctness.

**Acceptance (extend `tests/test_allocators.py`):** *(γ = 1 criterion revised — see the correction note above)*
- **γ = 0 ≡ HRP** — `np.allclose(SchurHRPAllocator().allocate(rm, gamma=0), hrp_weights(rm))`. Hard regression anchor. **(kept — this one is exact.)**
- ~~**γ = 1 ≡ min-variance** — matches `Σ⁻¹𝟙 / 𝟙ᵀΣ⁻¹𝟙` on a small clean problem.~~ **Revised (exact equality not achievable):** instead assert min-variance as a strict *variance lower bound* — `schur_var(γ) ≥ minvar_var` for all γ — and that γ = 1 *differs* from HRP on an n ≥ 4 problem with a real 2-2 split (proving cross-block info is used).
- Weights sum to 1, non-negative, all assets covered.
- ~~(Optional) concentration increases monotonically in γ toward the min-var solution.~~ **Dropped:** verified not robust — on real shrunk covariances raw γ = 1 can *raise* variance above HRP (this is what skfolio's `keep_monotonic` γ-capping exists to fix; not ported here).

**CLI:** add `schur-hrp` to the `--method` choices (`analyze.py:384`,
`analyze.py:417`) and a `--gamma` flag; validate against HRP through the
walk-forward engine.

---

## Build order & dependency graph

```
0a Signals ──┐
0b Allocators ┼──► 0c Walk-forward engine ──► 1 Schur-HRP (validated via engine)
             ┘
```

0a and 0b are independent and small; 0c depends on both; Schur-HRP depends on 0b
(to plug in) and 0c (to validate). Each step lands with the full existing suite
green as the regression net.

## Definition of done

- New files: `signals.py`, `allocators.py`, `backtest.py`; new tests:
  `test_signals.py`, `test_allocators.py`, `test_backtest.py`.
- `--method schur-hrp` and `--gamma` exposed on `allocate` and `backtest`.
- `pytest` fully green, including every pre-existing test unedited.
- Schur-HRP γ=0 and γ=1 boundary invariants pass.

## Out of scope (later phases)

HRP-Σμ (Phase 2) and CRISP (Phase 3, `μᵀw − λwᵀΣw − γwᵀCw − τ‖w−w₀‖₁`,
correlation-regularized iterative shrinkage) build on this infrastructure but
are not part of this handoff.
