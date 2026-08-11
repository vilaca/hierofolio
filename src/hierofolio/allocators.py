from typing import Optional, Protocol
import numpy as np
import pandas as pd

from hierofolio.risk_model import (
    ConstrainedMVOOptimizer,
    CRISPOptimizer,
    RiskModel,
    RobustOptimizer,
    _project_to_psd,
)


def _is_positive_definite(cov: np.ndarray) -> bool:
    """True if a Cholesky factor exists (symmetric positive-definite)."""
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


def _regularized(cov: np.ndarray) -> np.ndarray:
    """PSD-regularize a covariance block via eigenvalue clipping, only when needed.

    Already-PD blocks are returned unchanged so the gamma=0 HRP parity is exact
    and clean, well-conditioned problems are not perturbed.
    """
    return cov if _is_positive_definite(cov) else _project_to_psd(cov)


def _symmetric_step_up_matrix(n1: int, n2: int) -> np.ndarray:
    """Matrix M (n1 x n2) with M @ ones(n2) == ones(n1).

    This is the "reference completion" from Cotton's construction: it lets the
    augmented block be solved against a plain ``1`` vector while still encoding
    the cross-block ``1`` coupling, which is what drives gamma toward the
    minimum-variance solution.
    """
    assert abs(n1 - n2) <= 1
    if n1 == n2:
        return np.eye(n1)
    if n1 < n2:
        return _symmetric_step_up_matrix(n2, n1).T * n1 / n2
    m = np.zeros((n1, n2))
    j_row = np.ones((1, n2)) / n2
    e = np.eye(n2)
    for j in range(n1):
        m += np.concatenate([e[:j], j_row, e[j:]], axis=0) / n1
    return m


def _schur_augmentation(
    a: np.ndarray, b: np.ndarray, d: np.ndarray, gamma: float
) -> np.ndarray:
    """Schur-complement augmentation of block ``A`` given cross-block ``B`` and ``D``.

    Mirrors Peter Cotton's reference implementation (as ported into skfolio's
    ``SchurComplementary``), including the step-up completion factor ``r``.

    gamma=0 (or a singleton block) returns ``A`` unchanged, so the recursion
    reduces exactly to HRP. As gamma rises toward 1 the cross-block coupling is
    folded in, moving the allocation toward global minimum variance.
    """
    n_a, n_d = a.shape[0], d.shape[0]
    if gamma == 0 or n_a == 1 or n_d == 1:
        return a
    d = _regularized(d)
    a_aug = a - gamma * b @ np.linalg.solve(d, b.T)
    m = _symmetric_step_up_matrix(n_a, n_d)
    b_d_inv = np.linalg.solve(d.T, b.T).T  # B @ inv(D)
    r = np.eye(n_a) - gamma * b_d_inv @ m.T
    a_aug = np.linalg.solve(r, a_aug)
    a_aug = 0.5 * (a_aug + a_aug.T)  # restore symmetry lost to the r-solve
    return _regularized(a_aug)


def _naive_cluster_variance(cov: np.ndarray) -> float:
    """Portfolio variance under inverse-variance weights (the HRP cluster measure)."""
    inv_diag = 1.0 / np.diag(cov)
    w = inv_diag / inv_diag.sum()
    return float(w @ cov @ w)


def _schur_hrp_weights(cov: np.ndarray, gamma: float) -> np.ndarray:
    """Schur-complementary recursive bisection on an order-sorted covariance matrix.

    The bisection (`mid = len(c) // 2`) and the inverse-variance split match
    ``HRPAllocator`` exactly, so gamma=0 reproduces HRP bit-for-bit. Augmented
    diagonal blocks are written back in place so deeper splits see the
    cross-block-adjusted covariance, matching Cotton's reference recursion.
    """
    cov = cov.copy()
    n = len(cov)
    weights = np.ones(n)
    clusters = [np.arange(n)]
    while clusters:
        next_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            next_clusters += [left, right]

            a = cov[np.ix_(left, left)]
            d = cov[np.ix_(right, right)]
            if len(left) <= 1 or len(right) <= 1:
                a_aug, d_aug = a, d
            else:
                b = cov[np.ix_(left, right)]
                a_aug = _schur_augmentation(a, b, d, gamma)
                d_aug = _schur_augmentation(d, b.T, a, gamma)
                cov[np.ix_(left, left)] = a_aug
                cov[np.ix_(right, right)] = d_aug

            lv = _naive_cluster_variance(a_aug)
            rv = _naive_cluster_variance(d_aug)
            alpha = 1 - lv / (lv + rv)
            weights[left] *= alpha
            weights[right] *= 1 - alpha
        clusters = next_clusters
    return weights / weights.sum()


class Allocator(Protocol):
    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        **params,
    ) -> pd.Series: ...


class HRPAllocator:
    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        verbose: bool = False,
        **params,
    ) -> pd.Series:
        cov = risk_model.covariance()
        assets = risk_model.leaf_order

        def cluster_var(cluster):
            sub = cov.loc[cluster, cluster].values
            inv_diag = 1.0 / np.diag(sub)
            w = inv_diag / inv_diag.sum()
            return float(w @ sub @ w)

        if verbose:
            print(f"\nDendrogram leaf order: {' → '.join(assets)}")
            print("(adjacent assets cluster together)\n")
            print("Recursive bisection:")

        weights = pd.Series(1.0, index=assets)
        clusters = [list(assets)]
        step = 0
        while clusters:
            next_clusters = []
            for c in clusters:
                if len(c) > 1:
                    mid = len(c) // 2
                    left, right = c[:mid], c[mid:]
                    lv, rv = cluster_var(left), cluster_var(right)
                    alpha = 1 - lv / (lv + rv)
                    weights[left] *= alpha
                    weights[right] *= 1 - alpha
                    if verbose:
                        step += 1
                        ls = "{" + ", ".join(left) + "}"
                        rs = "{" + ", ".join(right) + "}"
                        print(f"  Step {step}: {ls} vs {rs}")
                        print(
                            f"    {ls:<40} branch vol {np.sqrt(lv * 252)*100:5.1f}%"
                            f"  →  {alpha:.1%} of parent budget"
                        )
                        print(
                            f"    {rs:<40} branch vol {np.sqrt(rv * 252)*100:5.1f}%"
                            f"  →  {1 - alpha:.1%} of parent budget"
                        )
                    next_clusters += [left, right]
            clusters = next_clusters
        return weights / weights.sum()


class SchurHRPAllocator:
    """Schur-complementary HRP: a continuous HRP (gamma=0) -> min-variance (gamma->1) dial.

    At gamma=0 this reproduces plain HRP exactly. As gamma rises the allocation
    folds in cross-block covariance and moves toward the global minimum-variance
    portfolio (approaching, not equalling, it). Ignores ``signal`` /
    ``current_weights`` like ``HRPAllocator``.
    """

    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        gamma: float = 0.5,
        **params,
    ) -> pd.Series:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        ordered_cov = risk_model.quasi_diagonalized_covariance
        weights = _schur_hrp_weights(ordered_cov.values, gamma)
        return pd.Series(weights, index=list(ordered_cov.index))


class HRPSigmaMuAllocator:
    """Signal-aware HRP: tilts each budget split toward higher-μ branches.

    tau=0 reproduces HRP exactly (exp(0)=1 reduces the split to plain
    inverse-variance, α = v_R/(v_L+v_R)). As tau grows the allocation leans
    toward whichever branch has the higher inverse-variance-weighted expected
    return. Requires a signal (annualized μ); raises ValueError if absent or
    if any asset is missing from the signal.

    Spectrum: risk-only HRP (tau=0) → signal-aware HRP-Σμ → MVO (high tau, large signal).
    """

    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        tau: float = 1.0,
        **params,
    ) -> pd.Series:
        if signal is None:
            raise ValueError("HRPSigmaMuAllocator requires a signal (mu).")
        cov = risk_model.covariance()
        assets = risk_model.leaf_order
        mu = signal.reindex(assets)
        if mu.isna().any():
            raise ValueError("signal must cover all assets in the risk model.")

        def cluster_var_and_mu(cluster):
            sub = cov.loc[cluster, cluster].values
            inv_diag = 1.0 / np.diag(sub)
            w = inv_diag / inv_diag.sum()
            return float(w @ sub @ w), float(w @ mu.loc[cluster].values)

        weights = pd.Series(1.0, index=assets)
        clusters = [list(assets)]
        while clusters:
            next_clusters = []
            for c in clusters:
                if len(c) > 1:
                    mid = len(c) // 2
                    left, right = c[:mid], c[mid:]
                    lv, lm = cluster_var_and_mu(left)
                    rv, rm = cluster_var_and_mu(right)
                    score_l = (1.0 / lv) * np.exp(tau * lm)
                    score_r = (1.0 / rv) * np.exp(tau * rm)
                    alpha = score_l / (score_l + score_r)
                    weights[left] *= alpha
                    weights[right] *= 1 - alpha
                    next_clusters += [left, right]
            clusters = next_clusters
        return weights / weights.sum()


class MVOAllocator:
    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        risk_aversion: float = 1.0,
        max_weight: Optional[float] = None,
        **params,
    ) -> pd.Series:
        optimizer = ConstrainedMVOOptimizer(
            risk_model=risk_model, alpha=signal, risk_aversion=risk_aversion,
        )
        return optimizer.solve(max_weight=max_weight)


class RobustAllocator:
    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        risk_aversion: float = 1.0,
        robustness_penalty: float = 1.0,
        max_weight: Optional[float] = None,
        **params,
    ) -> pd.Series:
        optimizer = RobustOptimizer(
            risk_model=risk_model, alpha=signal, risk_aversion=risk_aversion,
            robustness_penalty=robustness_penalty,
        )
        return optimizer.solve(max_weight=max_weight)


class CRISPAllocator:
    """Correlation-regularized signal-aware allocator (CRISP).

    Thin adapter over ``CRISPOptimizer``: signal-aware like MVO, plus a
    redundancy penalty (γ) on correlated bets and a soft turnover penalty (τ)
    toward ``current_weights``. γ=0, τ=0 reduces to ``MVOAllocator`` exactly.
    """

    def allocate(
        self,
        risk_model: RiskModel,
        signal: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        risk_aversion: float = 1.0,
        corr_penalty: float = 0.3,
        turnover_penalty: float = 0.0,
        max_weight: Optional[float] = None,
        **params,
    ) -> pd.Series:
        optimizer = CRISPOptimizer(
            risk_model=risk_model, alpha=signal, current_weights=current_weights,
            risk_aversion=risk_aversion, corr_penalty=corr_penalty,
            turnover_penalty=turnover_penalty,
        )
        return optimizer.solve(max_weight=max_weight)


ALLOCATORS: dict = {
    "hrp": HRPAllocator(),
    "schur-hrp": SchurHRPAllocator(),
    "hrp-sigma-mu": HRPSigmaMuAllocator(),
    "mvo": MVOAllocator(),
    "robust": RobustAllocator(),
    "crisp": CRISPAllocator(),
}
