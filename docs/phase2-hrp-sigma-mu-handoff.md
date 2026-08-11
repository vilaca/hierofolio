# Handoff — Phase 2: HRP-Σμ (signal-aware hierarchical allocation)

Implementation handoff for the next optimizer on the roadmap
(HRP → Schur-HRP → **HRP-Σμ** → CRISP). Phase 0 infrastructure and Schur-HRP
are already implemented and merged (commits `2382012`, `b58bc9e`). This document
is self-contained: it can be executed in a fresh session.

## Session settings (required)

| Sub-step | Model / effort | Why |
|---|---|---|
| Steps 1–3 (allocator, wiring, tests) | **Sonnet 4.6, high** | Well-specified below; the τ=0≡HRP invariant is a hard guard. No matrix algebra — simpler than Schur-HRP was. |
| Docs tail (Step 4) | **Sonnet 4.6, high** (or Haiku 4.5, low) | Mechanical README/docstring updates. |

**Escalation trigger:** if the τ=0 parity or the monotonicity test misbehaves in
a way that isn't an obvious typo, bump to **Opus 4.8, xhigh** — that means the
split formula's reduction to HRP is subtly wrong.

`/clear` is appropriate before starting; the only input needed is this document.

## Guiding constraint (read first)

Same rule that governed Phase 0: **no existing public signature changes.**
`HRPAllocator`, `SchurHRPAllocator`, `WalkForwardEngine`, `run_backtest`, and the
CLI keep working exactly as they do now. The acceptance gate is a parity test
(τ=0 reproduces HRP via `np.allclose`) plus the whole existing suite green with
zero edits.

## What Phase 0/1 already gives you (real, implemented)

- **`Allocator` protocol** (`src/hfolio/allocators.py:125`):
  `allocate(risk_model, signal=None, current_weights=None, **params) -> pd.Series`.
- **Bisection reference to mirror** — `HRPAllocator.allocate`
  (`allocators.py:135`): walks `risk_model.leaf_order`, splits each ordered
  cluster at `mid = len(c)//2`, and sets
  `alpha = 1 - lv/(lv+rv)` from inverse-variance cluster variances.
- **Signal already threaded** — `WalkForwardEngine.run` computes
  `mu, _ = signal_model.signal(train)` and calls
  `allocator.allocate(risk_model, signal=mu, ...)` (`backtest.py:196-199`).
  So a signal-aware allocator receives μ for free in every backtest fold.
- **`HistoricalMeanSignal`** (`signals.py:10`) returns annualized μ
  (`(1+mean)**252 - 1`) and `None` uncertainty.
- **Registry** — `ALLOCATORS` dict (`allocators.py:247`) is the single dispatch
  point used by both the CLI (`analyze.py:468`) and `run_backtest`
  (`analyze.py:235`).

HRP and Schur-HRP ignore `signal`. **HRP-Σμ is the first bisection allocator
that consumes it.**

---

## Step 1 — `HRPSigmaMuAllocator`

**The idea.** Plain HRP splits each parent's budget purely by inverse cluster
variance, ignoring expected return. HRP-Σμ keeps HRP's dendrogram structure and
recursive bisection but *tilts each split toward the higher-expected-return
branch*, controlled by a signal-strength knob τ. At τ=0 it is exactly HRP; as τ
grows it leans harder on μ.

**The split formula (recommended: multiplicative / softmax tilt).** At each
split of an ordered cluster into left `L` and right `R`:

```
v_L, v_R : cluster variances (inverse-variance weights — identical to HRPAllocator)
m_L, m_R : cluster expected returns = the SAME inverse-variance weights · μ_cluster

score_L = (1 / v_L) · exp(τ · m_L)
score_R = (1 / v_R) · exp(τ · m_R)
alpha   = score_L / (score_L + score_R)     # left branch's share of the budget
```

Why this form:
- **τ = 0 ⟹ HRP exactly.** `exp(0) = 1`, so
  `alpha = (1/v_L)/((1/v_L)+(1/v_R)) = v_R/(v_L+v_R)` — the current HRP split
  bit-for-bit. This is the hard parity anchor.
- **Always positive / well-defined.** No `max(ε, ·)` guard needed; a large
  negative μ can't produce a negative or blown-up split.
- **Shift-invariant in μ.** Adding a constant to all μ cancels in the ratio, so
  centering μ is unnecessary.
- **Monotone.** Higher `m_L` (holding risk fixed) strictly raises `alpha`.

`m_L`/`m_R` use the *same* within-cluster inverse-variance weights already
computed for `v_L`/`v_R`, so the return and risk aggregations are consistent.

> Alternative linear tilt `(1/v)·max(ε, 1 + τ·m)` is acceptable but needs the ε
> guard and isn't shift-invariant. Prefer the exp form unless a test says
> otherwise.

**Implementation** in `allocators.py`, structured like `HRPAllocator` (operate on
`risk_model.covariance()` and `risk_model.leaf_order`, plain cluster variance —
no Schur augmentation; keep the two orthogonal):

```python
class HRPSigmaMuAllocator:
    """Signal-aware HRP: tilts each budget split toward higher-μ branches.

    tau=0 reproduces HRP exactly. Consumes `signal` (annualized mu); raises if
    it is missing or does not cover the universe.
    """
    def allocate(self, risk_model, signal=None, current_weights=None,
                 tau: float = 1.0, **params) -> pd.Series:
        if signal is None:
            raise ValueError("HRPSigmaMuAllocator requires a signal (mu).")
        cov = risk_model.covariance()
        assets = risk_model.leaf_order
        mu = signal.reindex(assets)
        if mu.isna().any():
            raise ValueError("signal must cover all assets in the risk model.")
        # recursive bisection identical to HRPAllocator, but alpha uses the
        # score_L / (score_L + score_R) tilt above.
        ...
```

Register it: add `"hrp-sigma-mu": HRPSigmaMuAllocator()` to `ALLOCATORS`
(`allocators.py:247`).

**τ scaling note:** μ is annualized (~0.05–0.15), so τ on the order of 1–10 is
where the tilt becomes visible. Default `tau=1.0`.

---

## Step 2 — CLI + `run_backtest` wiring

Mirror exactly how `gamma` was threaded for Schur-HRP:

1. **`--method` choices** — add `hrp-sigma-mu` in both parsers
   (`analyze.py:291` and `analyze.py:329`).
2. **New flag `--tau`** — add to both `allocate` and `backtest` parsers next to
   the existing `--gamma` (`analyze.py:295`, `analyze.py:333`), default `1.0`,
   `metavar='τ'`.
3. **`allocate` dispatch** (`analyze.py:460-469`): μ is already computed
   (`mu, _ = HistoricalMeanSignal().signal(returns)` at `:460`) and passed as
   `signal=mu`. Just add `"tau": args.tau` to `method_params` for this method.
4. **`run_backtest`** (`analyze.py:190`): add a `tau: float = 1.0` parameter and,
   in the `method_params` block (`analyze.py:221-231`), add
   `if method == "hrp-sigma-mu": method_params["tau"] = tau`. The engine already
   passes `signal=mu`, so nothing else in `backtest.py` changes.
5. Pass `tau=args.tau` in the CLI `run_backtest` call (near `analyze.py:542`,
   where `gamma=args.gamma` is passed today).

No change to `backtest.py` or `signals.py` is required.

---

## Step 3 — Tests

**Extend `tests/test_allocators.py`:**
- **τ = 0 ≡ HRP** —
  `np.allclose(HRPSigmaMuAllocator().allocate(rm, signal=mu, tau=0.0), hrp_weights(rm))`.
  Hard regression anchor.
- **Monotone tilt** — on a synthetic where one branch has a clearly higher μ,
  increasing τ (0 → 1 → 5) monotonically increases that branch's total weight.
- **Protocol conformance** — weights sum to 1, non-negative, index == assets.
- **Missing / partial signal raises** `ValueError`.

**Extend `tests/test_backtest.py`:**
- `run_backtest(returns, method="hrp-sigma-mu", ...)` produces a valid OOS series
  and a log whose weights sum to 1 (mirror the existing `test_run_backtest_mvo`).

All pre-existing tests must stay green **unedited**.

---

## Step 4 — Docs (mechanical tail)

- README: add `hrp-sigma-mu` and `--tau` to the allocator/backtest flag tables
  and the method table (alongside the `schur-hrp`/`--gamma` rows).
- Class docstring on `HRPSigmaMuAllocator` stating the τ=0≡HRP property and the
  spectrum position (risk-only HRP → signal-aware HRP-Σμ → MVO).

---

## Definition of done

- `HRPSigmaMuAllocator` in `allocators.py`, registered in `ALLOCATORS`.
- `--method hrp-sigma-mu` and `--tau` on both `allocate` and `backtest`.
- `run_backtest` accepts `tau`; engine unchanged.
- New tests pass, including the τ=0≡HRP anchor and the monotonicity test.
- `pytest` fully green, every pre-existing test unedited.

## Out of scope (Phase 3)

**CRISP** — `μᵀw − λwᵀΣw − γwᵀCw − τ‖w−w₀‖₁`, a `PortfolioOptimizer` subclass
using `covariance()` + `correlation()` (both already exist), built and validated
against HRP / Schur-HRP / HRP-Σμ as benchmarks. Separate handoff.

A "Schur-Σμ" combination (signal tilt on top of Schur-augmented blocks) is a
possible later variant; deliberately kept orthogonal here.
