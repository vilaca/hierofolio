"""Unit tests for risk_model.py (HRPRiskModel and the portfolio optimizers)."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from risk_model import (
    HRPRiskModel,
    ConstrainedMVOOptimizer,
    RobustOptimizer,
    _project_to_psd,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ASSETS = [f"A{i}" for i in range(8)]


def make_returns(n_obs: int = 250, assets=ASSETS, seed: int = 0) -> pd.DataFrame:
    """Deterministic correlated return series with a clear block structure."""
    rng = np.random.default_rng(seed)
    n = len(assets)
    # Two latent factors -> two natural blocks, so clustering is non-trivial.
    f1 = rng.standard_normal(n_obs)
    f2 = rng.standard_normal(n_obs)
    data = np.empty((n_obs, n))
    for j in range(n):
        factor = f1 if j < n // 2 else f2
        data[:, j] = 0.8 * factor + 0.2 * rng.standard_normal(n_obs)
    return pd.DataFrame(data, columns=assets)


@pytest.fixture
def returns():
    return make_returns()


@pytest.fixture
def fixed_model(returns):
    return HRPRiskModel(returns, cluster_mode="fixed", n_clusters=3)


@pytest.fixture
def full_model(returns):
    return HRPRiskModel(returns, cluster_mode="full")


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

def test_rejects_nan_returns():
    r = make_returns()
    r.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        HRPRiskModel(r)


def test_rejects_single_asset():
    r = make_returns(assets=["OnlyOne"])
    with pytest.raises(ValueError, match="at least 2 assets"):
        HRPRiskModel(r)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shrinkage_method": "nope"},
        {"cluster_mode": "nope"},
        {"linkage_method": "nope"},
    ],
)
def test_rejects_bad_config(returns, kwargs):
    with pytest.raises(ValueError):
        HRPRiskModel(returns, **kwargs)


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def test_covariance_shape_and_labels(fixed_model):
    cov = fixed_model.covariance()
    assert list(cov.index) == ASSETS
    assert list(cov.columns) == ASSETS
    assert cov.shape == (len(ASSETS), len(ASSETS))
    assert np.allclose(cov.values, cov.values.T)


def test_correlation_diagonal_is_one(fixed_model):
    corr = fixed_model.correlation()
    assert np.allclose(np.diag(corr.values), 1.0)
    assert corr.values.min() >= -1.0 and corr.values.max() <= 1.0


def test_distance_is_symmetric_zero_diagonal(fixed_model):
    dist = fixed_model.distance()
    assert np.allclose(np.diag(dist.values), 0.0)
    assert np.allclose(dist.values, dist.values.T)
    assert (dist.values >= 0).all()


def test_ledoit_wolf_covariance_is_psd(fixed_model):
    eigvals = np.linalg.eigvalsh(fixed_model.covariance().values)
    assert eigvals.min() > -1e-8


# ---------------------------------------------------------------------------
# Shrinkage methods
# ---------------------------------------------------------------------------

def test_identity_shrinkage_zero_intensity_recovers_sample_cov(returns):
    m = HRPRiskModel(returns, shrinkage_method="identity", shrinkage_intensity=0.0)
    assert np.allclose(m.covariance().values, returns.cov().values)


def test_identity_shrinkage_full_intensity_is_diagonal(returns):
    m = HRPRiskModel(returns, shrinkage_method="identity", shrinkage_intensity=1.0)
    cov = m.covariance().values
    off_diag = cov - np.diag(np.diag(cov))
    assert np.allclose(off_diag, 0.0)


def test_constant_correlation_zero_intensity_recovers_sample_cov(returns):
    m = HRPRiskModel(
        returns, shrinkage_method="constant_correlation", shrinkage_intensity=0.0
    )
    assert np.allclose(m.covariance().values, returns.cov().values)


@pytest.mark.parametrize("method", ["ledoit_wolf", "constant_correlation", "identity"])
def test_shrinkage_methods_symmetric(returns, method):
    m = HRPRiskModel(returns, shrinkage_method=method)
    cov = m.covariance().values
    assert np.allclose(cov, cov.T)


# ---------------------------------------------------------------------------
# Fixed-mode clustering
# ---------------------------------------------------------------------------

def test_fixed_leaf_clusters(fixed_model):
    leaves = fixed_model.leaf_clusters()
    assert len(leaves) == len(ASSETS)
    assert set(leaves.keys()) == set(range(len(ASSETS)))
    assert all(len(v) == 1 for v in leaves.values())


def test_fixed_cluster_ids_do_not_collide_with_leaf_ids(fixed_model):
    """Regression: fixed-mode cluster IDs are offset past the leaf ID range."""
    leaf_ids = set(fixed_model.leaf_clusters())
    cluster_ids = set(fixed_model.all_clusters())
    assert leaf_ids.isdisjoint(cluster_ids)


def test_fixed_root_contains_all_assets(fixed_model):
    root_assets = fixed_model.assets_in_cluster(fixed_model.root)
    assert set(root_assets) == set(ASSETS)


def test_fixed_default_n_clusters(returns):
    m = HRPRiskModel(returns, cluster_mode="fixed")  # default sqrt(n)
    expected = int(np.ceil(np.sqrt(len(ASSETS))))
    # internal clusters + any singletons together partition all assets.
    all_assets = [a for assets in m.cut(expected).values() for a in assets]
    assert sorted(all_assets) == sorted(ASSETS)


def test_fixed_cut_is_a_partition(fixed_model):
    partition = fixed_model.cut(3)
    flat = [a for assets in partition.values() for a in assets]
    assert sorted(flat) == sorted(ASSETS)  # each asset exactly once


def test_tree_methods_raise_in_fixed_mode(fixed_model):
    for call in (
        lambda: fixed_model.children(0),
        lambda: fixed_model.parent(0),
        lambda: fixed_model.level(0),
        lambda: fixed_model.path_of(ASSETS[0]),
    ):
        with pytest.raises(ValueError, match="cluster_mode='full'"):
            call()


# ---------------------------------------------------------------------------
# Full-mode clustering / tree traversal
# ---------------------------------------------------------------------------

def test_full_leaf_ids_are_zero_to_n_minus_one(full_model):
    assert set(full_model.leaf_clusters()) == set(range(len(ASSETS)))


def test_full_internal_ids_above_leaf_range(full_model):
    n = len(ASSETS)
    for node_id in full_model.internal_clusters():
        assert node_id >= n


def test_full_root_contains_all_assets(full_model):
    assert set(full_model.assets_under(full_model.root)) == set(ASSETS)


def test_full_children_and_parent_consistent(full_model):
    for node_id, (left, right) in full_model._children_map.items():
        assert full_model.parent(left) == node_id
        assert full_model.parent(right) == node_id
        merged = full_model.assets_under(left) + full_model.assets_under(right)
        assert sorted(merged) == sorted(full_model.assets_under(node_id))


def test_full_is_leaf_is_internal(full_model):
    for i in range(len(ASSETS)):
        assert full_model.is_leaf(i)
        assert not full_model.is_internal(i)
    assert full_model.is_internal(full_model.root)


def test_full_path_root_to_leaf(full_model):
    path = full_model.path_of(ASSETS[0])
    assert path[0] == full_model.root
    assert full_model.is_leaf(path[-1])
    assert full_model.assets_under(path[-1]) == [ASSETS[0]]


def test_full_path_unknown_asset_raises(full_model):
    with pytest.raises(KeyError):
        full_model.path_of("does-not-exist")


def test_full_level_increases_with_depth(full_model):
    assert all(full_model.level(i) == 0 for i in range(len(ASSETS)))
    assert full_model.level(full_model.root) >= 1


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_full_cut_is_a_partition(full_model, level):
    partition = full_model.cut(level)
    flat = [a for assets in partition.values() for a in assets]
    assert sorted(flat) == sorted(ASSETS)


def test_full_cut_level_zero_is_root(full_model):
    partition = full_model.cut(0)
    assert list(partition.keys()) == [full_model.root]


def test_cluster_of_full_mode_warns(full_model):
    with pytest.warns(UserWarning):
        full_model.cluster_of(ASSETS[0])


# ---------------------------------------------------------------------------
# Convenience / diagnostics
# ---------------------------------------------------------------------------

def test_subtree_weights_sum_to_root_total(full_model):
    weights = {a: 1.0 / len(ASSETS) for a in ASSETS}
    agg = full_model.subtree_weights(weights)
    assert agg[full_model.root] == pytest.approx(1.0)


def test_summary_columns(fixed_model):
    df = fixed_model.summary()
    assert {"node_id", "node_type", "n_assets", "cluster_std", "avg_correlation"} <= set(
        df.columns
    )
    assert (df["node_type"] == "root").sum() == 1


# ---------------------------------------------------------------------------
# Hashing & equality (regression for the broken dataclass __eq__)
# ---------------------------------------------------------------------------

def test_equality_does_not_raise_and_matches_config(returns):
    a = HRPRiskModel(returns, cluster_mode="fixed", n_clusters=3)
    b = HRPRiskModel(returns.copy(), cluster_mode="fixed", n_clusters=3)
    assert a == b  # must not raise on DataFrame/ndarray fields
    assert hash(a) == hash(b)


def test_inequality_on_different_config(returns):
    a = HRPRiskModel(returns, shrinkage_intensity=0.5)
    b = HRPRiskModel(returns, shrinkage_intensity=0.9)
    assert a != b


def test_not_equal_to_other_types(fixed_model):
    assert fixed_model != "not a model"


# ---------------------------------------------------------------------------
# PSD helper
# ---------------------------------------------------------------------------

def test_project_to_psd_clips_negative_eigenvalues():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    psd = _project_to_psd(a, eps=1e-10)
    assert np.linalg.eigvalsh(psd).min() >= -1e-8
    assert np.allclose(psd, psd.T)


# ---------------------------------------------------------------------------
# Optimizers (require cvxpy)
# ---------------------------------------------------------------------------


@pytest.fixture
def alpha():
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(len(ASSETS)) * 0.01, index=ASSETS)


def _assert_valid_weights(w, max_weight=None):
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w.values >= -1e-6).all()
    if max_weight is not None:
        assert (w.values <= max_weight + 1e-6).all()


def test_mvo_basic_solution(fixed_model, alpha):
    w = ConstrainedMVOOptimizer(fixed_model, alpha).solve(max_weight=0.5)
    assert list(w.index) == ASSETS
    _assert_valid_weights(w, max_weight=0.5)


def test_mvo_alpha_index_mismatch_raises(fixed_model):
    bad = pd.Series(np.zeros(len(ASSETS)), index=[f"X{i}" for i in range(len(ASSETS))])
    with pytest.raises(ValueError, match="Alpha index"):
        ConstrainedMVOOptimizer(fixed_model, bad)


def test_mvo_is_order_invariant(fixed_model, alpha):
    """Regression: covariance must be aligned to alpha's ordering.

    Solving with a shuffled alpha index must yield the same per-asset weights.
    """
    base = ConstrainedMVOOptimizer(fixed_model, alpha).solve(max_weight=0.5)

    shuffled = alpha.sample(frac=1.0, random_state=1)
    shuffled_w = ConstrainedMVOOptimizer(fixed_model, shuffled).solve(max_weight=0.5)

    aligned = shuffled_w.reindex(base.index)
    assert np.allclose(aligned.values, base.values, atol=1e-4)


def _capped_clusters(model):
    """Multi-asset clusters excluding the whole-universe root."""
    universe = set(a for assets in model.leaf_clusters().values() for a in assets)
    return [v for v in model.internal_clusters().values() if set(v) != universe]


def test_mvo_cluster_exposure_respected(fixed_model, alpha):
    cap = 0.5
    w = ConstrainedMVOOptimizer(fixed_model, alpha).solve(
        max_cluster_exposure=cap, max_weight=1.0
    )
    capped = _capped_clusters(fixed_model)
    assert capped  # ensure the constraint actually applies to something
    for assets_list in capped:
        assert w[assets_list].sum() <= cap + 1e-6


def test_mvo_cluster_exposure_requires_fixed_mode(full_model, alpha):
    with pytest.raises(ValueError, match="cluster_mode='fixed'"):
        ConstrainedMVOOptimizer(full_model, alpha).solve(max_cluster_exposure=0.5)


def test_mvo_turnover_constraint(fixed_model, alpha):
    current = pd.Series(1.0 / len(ASSETS), index=ASSETS)
    w = ConstrainedMVOOptimizer(fixed_model, alpha, current_weights=current).solve(
        turnover_limit=0.1, max_weight=1.0
    )
    turnover = 0.5 * np.abs(w.values - current.values).sum()
    assert turnover <= 0.1 + 1e-4


def test_robust_basic_solution(fixed_model, alpha):
    w = RobustOptimizer(fixed_model, alpha).solve(max_weight=0.5)
    _assert_valid_weights(w, max_weight=0.5)


def test_robust_default_uncertainty_from_covariance(fixed_model, alpha):
    opt = RobustOptimizer(fixed_model, alpha)
    expected = np.sqrt(np.diag(fixed_model.covariance().values))
    assert np.allclose(opt.alpha_uncertainty.reindex(ASSETS).values, expected)


def test_robust_cluster_constraint_fixed_mode_feasible(fixed_model, alpha):
    """Regression: fixed-mode cluster cap must not collapse to one whole-book cluster."""
    w = RobustOptimizer(fixed_model, alpha).solve(
        max_cluster_exposure=0.5, max_weight=1.0
    )
    _assert_valid_weights(w)
    for assets_list in _capped_clusters(fixed_model):
        assert w[assets_list].sum() <= 0.5 + 1e-6


def test_robust_uncertainty_index_mismatch_raises(fixed_model, alpha):
    bad_unc = pd.Series(np.ones(len(ASSETS)), index=[f"Z{i}" for i in range(len(ASSETS))])
    with pytest.raises(ValueError, match="uncertainty index"):
        RobustOptimizer(fixed_model, alpha, alpha_uncertainty=bad_unc)
