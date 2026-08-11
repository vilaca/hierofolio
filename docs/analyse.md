# analyse.sh — step-by-step walkthrough

`analyse.sh` runs three subcommands against the same ETF universe, excluding three
short-history funds so the common return window reaches back to 2019:

```
uv run python -m hfolio.analyze summary    --exclude IE0006WW1TQ4 IE0003XJA0J9 IE000YYE6WK5
uv run python -m hfolio.analyze allocate   --method hrp   --exclude ...
uv run python -m hfolio.analyze allocate   --method crisp --corr-penalty 0.5 --exclude ...
```

All three share a common data-loading path; they diverge at the model step.

---

## Shared: data loading

**Source:** `data/hfolio.db` (SQLite).

1. `read_prices()` runs:
   ```sql
   SELECT isin, date, close FROM prices ORDER BY date, isin
   ```
   and pivots the result into a `date × isin` DataFrame of closing prices.

2. `pct_change().dropna()` converts prices to daily simple returns; the first row
   (always NaN) is dropped.  The result is a `T × N` returns matrix.

3. A currency warning is emitted when ISINs quote in different currencies (USD, EUR,
   LSE).  The warning is informational — no FX conversion is applied.

---

## Step 1 — `summary`

Computes and prints descriptive statistics; **no model is fitted**.

| Statistic | Formula |
|-----------|---------|
| Ann Return | `(1 + daily_mean)^252 − 1` |
| Ann Vol | `daily_std × √252` |
| Sharpe | `Ann Return / Ann Vol` |

A full Pearson correlation matrix is also printed.

The "Ready for HRPRiskModel" code block printed at the end is a display snippet,
not live code.

---

## Step 2 — `allocate --method hrp`

### 2a. Risk model — `HRPRiskModel`

**Pearson correlation and HRP distance**

```
corr[i,j] = Pearson(returns_i, returns_j), clipped to [−1, 1]
dist[i,j] = sqrt(0.5 × (1 − corr[i,j]))
```

The distance metric maps perfectly correlated pairs to 0 and uncorrelated pairs to
~0.707; it is always non-negative (zero-floored before the sqrt to guard against
floating-point near-1 values).

**Hierarchical clustering (Ward linkage)**

`scipy.cluster.hierarchy.linkage(condensed_distances, method='ward')` builds a
dendrogram by repeatedly merging the two clusters whose merger minimises the
within-cluster variance increment.  The resulting linkage matrix is the backbone of
the HRP bisection.

**Dendrogram leaf order**

`scipy.cluster.hierarchy.dendrogram(..., no_plot=True)` extracts the quasi-diagonal
leaf order — the order in which assets appear on the x-axis of the dendrogram.
Assets that are highly correlated end up adjacent; HRP exploits this ordering during
bisection so that each split divides genuinely different risk clusters.

**Shrunk covariance (constant-correlation shrinkage, intensity 0.3)**

The sample covariance can be noisy with ~7 years of daily data.  The shrinkage
target is a "constant-correlation" matrix: off-diagonal entries are replaced by
`std_i × std_j × ρ̄`, where `ρ̄` is the average pairwise Pearson correlation across
all asset pairs.  The final estimate blends the two:

```
Σ_shrunk = 0.7 × Σ_sample + 0.3 × Σ_target
```

### 2b. Signal

`HistoricalMeanSignal` computes the annualized compounded mean return per asset:

```
μ_i = (1 + daily_mean_i)^252 − 1
```

(HRP ignores μ; it is computed here for consistency with the other allocators.)

### 2c. HRP recursive bisection

Starting with all assets assigned weight 1.0 and the full dendrogram-ordered list:

1. Split the current cluster at its midpoint into a left half and a right half.
2. Compute the **inverse-variance portfolio variance** for each half:
   - Within each half, weights are proportional to `1 / σ_ii` (diagonal of the
     sub-covariance).
   - Portfolio variance: `w^T Σ_sub w`.
3. Allocate budget between the two halves inversely proportional to their variance:
   ```
   α = 1 − var_left / (var_left + var_right)
   weights[left]  *= α
   weights[right] *= (1 − α)
   ```
   The riskier half gets less budget.
4. Repeat for each sub-cluster until all clusters are singletons.
5. Normalize so weights sum to 1.

The result is a fully-invested, long-only allocation that never solves a quadratic
program — purely graph-based.

### 2d. In-sample statistics

```
port_returns = returns @ w          # daily portfolio return series
ann_return   = (1 + port_returns).prod()^(252/T) − 1
ann_vol      = std(port_returns) × √252
sharpe       = ann_return / ann_vol
```

The SRI class maps annualized vol to the EU PRIIP risk scale (1–7).

---

## Step 3 — `allocate --method crisp --corr-penalty 0.5`

### 3a. Risk model

Identical to HRP: same `HRPRiskModel` with constant-correlation shrinkage at
intensity 0.3.

### 3b. Signal

Same `HistoricalMeanSignal` → annualized mean μ per asset.

### 3c. CRISP optimizer (cvxpy / CLARABEL solver)

The shrunk covariance is projected onto the positive-semidefinite cone (eigenvalues
clipped to ≥ 0) before being passed to the solver.

**Decision variable:** `w ∈ ℝⁿ`, constrained `w ≥ 0`.

**Objective (maximized):**

```
μᵀw  −  (λ/2) wᵀΣw  −  γ wᵀCw
```

| Term | Role |
|------|------|
| `μᵀw` | Expected return (signal-following) |
| `−(λ/2) wᵀΣw` | Variance penalty, `λ = 1.0` |
| `−γ wᵀCw` | Correlation-redundancy penalty, `γ = 0.5` |

`C` is the PSD-projected Pearson correlation matrix.  Because the diagonal of `C`
equals 1, the penalty also acts as a mild L2 concentration term.  Because the
off-diagonal entries equal pairwise correlations, it directly penalises allocating
to two assets that move together — CRISP's core "anti-redundancy" behaviour.

**Constraints:**

```
sum(w) = 1    (fully invested)
w ≥ 0         (long-only)
```

No maximum-weight cap is applied in `analyse.sh`.

CLARABEL solves the resulting quadratic program.  At `γ = 0.5`, the
correlation-redundancy term is strong enough that CRISP concentrates the portfolio
into the three assets with the best risk-adjusted, non-redundant signal — US Tech,
Bonds, and Health Care — zeroing out the rest.

### 3d. In-sample statistics

Same as HRP (Step 2d).

---

## Key files

| File | Role |
|------|------|
| `src/hfolio/analyze.py` | Entry point, subcommand dispatch, data loading, stats, output |
| `src/hfolio/risk_model.py` | `HRPRiskModel` (clustering, shrinkage), `CRISPOptimizer` |
| `src/hfolio/allocators.py` | `HRPAllocator` (bisection), `CRISPAllocator` |
| `src/hfolio/signals.py` | `HistoricalMeanSignal` |
| `data/hfolio.db` | SQLite price database |
| `config/etf_universe.yaml` | ISIN → human-readable name mapping |
| `data/currency_metadata.yaml` | ISIN → quote currency mapping |
