import numpy as np
import pandas as pd
import pytest

from hierofolio.allocators import (
    HRPAllocator,
    MVOAllocator,
    RobustAllocator,
    SchurHRPAllocator,
)
from hierofolio.analyze import hrp_weights
from hierofolio.risk_model import ConstrainedMVOOptimizer, HRPRiskModel, RobustOptimizer
from hierofolio.signals import HistoricalMeanSignal


def make_returns(n_obs: int = 800, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n_obs)
    f2 = rng.standard_normal(n_obs)
    data = np.empty((n_obs, n_assets))
    for j in range(n_assets):
        factor = f1 if j < n_assets - 1 else f2
        data[:, j] = 0.01 * (0.8 * factor + 0.2 * rng.standard_normal(n_obs))
    return pd.DataFrame(
        data,
        index=pd.date_range("2016-01-01", periods=n_obs, freq="B"),
        columns=[f"A{j}" for j in range(n_assets)],
    )


@pytest.fixture
def returns():
    return make_returns()


@pytest.fixture
def risk_model(returns):
    return HRPRiskModel(returns, cluster_mode="full", linkage_method="ward")


@pytest.fixture
def risk_model4():
    # 4 assets so the dendrogram makes a genuine 2-2 split, which is the
    # smallest case where Schur cross-block augmentation actually fires.
    return HRPRiskModel(
        make_returns(n_assets=4), cluster_mode="full", linkage_method="ward"
    )


def _min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Unconstrained global min-variance weights: Sigma^-1 1 / (1' Sigma^-1 1)."""
    inv1 = np.linalg.solve(cov.values, np.ones(len(cov)))
    return pd.Series(inv1 / inv1.sum(), index=cov.index)


def _portfolio_variance(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.index).values
    return float(w @ cov.values @ w)


# ---------------------------------------------------------------------------
# HRPAllocator — protocol conformance + parity with hrp_weights
# ---------------------------------------------------------------------------

def test_hrp_allocator_sum_to_one(risk_model):
    w = HRPAllocator().allocate(risk_model)
    assert abs(w.sum() - 1.0) < 1e-9


def test_hrp_allocator_nonnegative(risk_model):
    w = HRPAllocator().allocate(risk_model)
    assert (w >= 0).all()


def test_hrp_allocator_covers_all_assets(risk_model, returns):
    w = HRPAllocator().allocate(risk_model)
    assert set(w.index) == set(returns.columns)


def test_hrp_allocator_parity_with_hrp_weights(risk_model):
    w_allocator = HRPAllocator().allocate(risk_model)
    w_legacy = hrp_weights(risk_model)
    assert np.allclose(w_allocator.reindex(w_legacy.index).values, w_legacy.values)


def test_hrp_allocator_verbose_no_crash(risk_model, capsys):
    HRPAllocator().allocate(risk_model, verbose=True)
    out = capsys.readouterr().out
    assert "Dendrogram" in out
    assert "Step" in out


# ---------------------------------------------------------------------------
# MVOAllocator — protocol conformance + parity with ConstrainedMVOOptimizer
# ---------------------------------------------------------------------------

def test_mvo_allocator_sum_to_one(risk_model, returns):
    mu, _ = HistoricalMeanSignal().signal(returns)
    w = MVOAllocator().allocate(risk_model, signal=mu)
    assert abs(w.sum() - 1.0) < 1e-6


def test_mvo_allocator_nonnegative(risk_model, returns):
    mu, _ = HistoricalMeanSignal().signal(returns)
    w = MVOAllocator().allocate(risk_model, signal=mu)
    assert (w >= -1e-9).all()


def test_mvo_allocator_parity_with_direct_optimizer(risk_model, returns):
    mu, _ = HistoricalMeanSignal().signal(returns)
    w_alloc = MVOAllocator().allocate(risk_model, signal=mu, risk_aversion=2.0)
    w_direct = ConstrainedMVOOptimizer(
        risk_model=risk_model, alpha=mu, risk_aversion=2.0
    ).solve(max_weight=None)
    assert np.allclose(
        w_alloc.reindex(w_direct.index).values,
        w_direct.values,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# SchurHRPAllocator — HRP <-> min-variance dial
#
# The handoff proposed an exact "gamma=1 == min-variance" invariant. That is
# not achievable for a reference-faithful implementation: the recursion keeps
# inverse-variance within-cluster weights at every gamma, so gamma=1 only
# *approaches* min-variance (and is provably incompatible with gamma=0 == HRP,
# which needs a different within-cluster rule). We therefore keep gamma=0 == HRP
# as the exact anchor and assert the achievable gamma=1 invariants: valid
# weights, min-variance as a strict variance lower bound, and that gamma=1
# genuinely uses cross-block information (differs from HRP).
# ---------------------------------------------------------------------------

def test_schur_gamma0_equals_hrp(risk_model):
    w_schur = SchurHRPAllocator().allocate(risk_model, gamma=0.0)
    w_hrp = hrp_weights(risk_model)
    assert np.allclose(w_schur.reindex(w_hrp.index).values, w_hrp.values)


def test_schur_gamma0_equals_hrp_4_assets(risk_model4):
    w_schur = SchurHRPAllocator().allocate(risk_model4, gamma=0.0)
    w_hrp = hrp_weights(risk_model4)
    assert np.allclose(w_schur.reindex(w_hrp.index).values, w_hrp.values)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_schur_weights_valid(risk_model4, gamma):
    w = SchurHRPAllocator().allocate(risk_model4, gamma=gamma)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= -1e-12).all()
    assert set(w.index) == set(risk_model4.covariance().index)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_schur_variance_lower_bounded_by_min_variance(risk_model4, gamma):
    # Any weights summing to 1 have variance >= the unconstrained global minimum.
    cov = risk_model4.quasi_diagonalized_covariance
    w = SchurHRPAllocator().allocate(risk_model4, gamma=gamma)
    v_schur = _portfolio_variance(w, cov)
    v_min = _portfolio_variance(_min_variance_weights(cov), cov)
    assert v_schur >= v_min - 1e-12


def test_schur_gamma1_uses_cross_block_info(risk_model4):
    # With a real 2-2 split, gamma=1 must differ from plain HRP: the whole point
    # is that it folds in the cross-block covariance HRP discards.
    w1 = SchurHRPAllocator().allocate(risk_model4, gamma=1.0)
    w_hrp = hrp_weights(risk_model4)
    assert not np.allclose(w1.reindex(w_hrp.index).values, w_hrp.values)


@pytest.mark.parametrize("gamma", [-0.1, 1.5])
def test_schur_gamma_out_of_range_raises(risk_model4, gamma):
    with pytest.raises(ValueError, match="gamma"):
        SchurHRPAllocator().allocate(risk_model4, gamma=gamma)


# ---------------------------------------------------------------------------
# RobustAllocator — protocol conformance + parity with RobustOptimizer
# ---------------------------------------------------------------------------

def test_robust_allocator_sum_to_one(risk_model, returns):
    mu, _ = HistoricalMeanSignal().signal(returns)
    w = RobustAllocator().allocate(risk_model, signal=mu)
    assert abs(w.sum() - 1.0) < 1e-6


def test_robust_allocator_parity_with_direct_optimizer(risk_model, returns):
    mu, _ = HistoricalMeanSignal().signal(returns)
    w_alloc = RobustAllocator().allocate(
        risk_model, signal=mu, risk_aversion=1.0, robustness_penalty=2.0
    )
    w_direct = RobustOptimizer(
        risk_model=risk_model, alpha=mu, risk_aversion=1.0, robustness_penalty=2.0
    ).solve(max_weight=None)
    assert np.allclose(
        w_alloc.reindex(w_direct.index).values,
        w_direct.values,
        atol=1e-6,
    )
