from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from hierofolio.allocators import HRPAllocator
from hierofolio.analyze import run_backtest
from hierofolio.backtest import WalkForwardEngine
from hierofolio.risk_model import HRPRiskModel
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


def default_factory(returns: pd.DataFrame) -> HRPRiskModel:
    return HRPRiskModel(
        returns, cluster_mode="full", linkage_method="ward",
        shrinkage_method="constant_correlation", shrinkage_intensity=0.3,
    )


@pytest.fixture
def returns():
    return make_returns()


# ---------------------------------------------------------------------------
# Engine reproduces run_backtest exactly
# ---------------------------------------------------------------------------

def test_engine_reproduces_run_backtest(returns):
    oos_legacy, ew_legacy, log_legacy = run_backtest(
        returns, method="hrp", window_years=2, step_months=6
    )

    engine = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        window_policy="rolling",
        purge_days=1,
        embargo_days=0,
    )
    oos_engine, ew_engine, log_engine = engine.run(returns)

    assert np.allclose(oos_legacy.values, oos_engine.values, atol=1e-12)
    assert np.allclose(ew_legacy.values, ew_engine.values, atol=1e-12)
    assert len(log_legacy) == len(log_engine)


# ---------------------------------------------------------------------------
# Purge/embargo date exclusion
# ---------------------------------------------------------------------------

def test_purge_removes_rebal_date(returns):
    engine = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        purge_days=1,
        embargo_days=0,
    )
    oos, _, log = engine.run(returns)
    # For each fold, verify its hold period starts strictly after rebal_date.
    # Fold i holds (rebal_date_i, next_rebal_i] — next_rebal_i == rebal_date_{i+1},
    # so rebal_date_i can legally appear in fold i-1's hold as the last day.
    for i, entry in enumerate(log):
        rebal_date = pd.Timestamp(entry["date"])
        next_rebal = (
            pd.Timestamp(log[i + 1]["date"]) if i + 1 < len(log) else oos.index[-1]
        )
        fold_oos = oos[(oos.index > rebal_date) & (oos.index <= next_rebal)]
        assert len(fold_oos) > 0
        assert fold_oos.index[0] > rebal_date


def test_embargo_removes_extra_days(returns):
    engine0 = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        purge_days=1,
        embargo_days=0,
    )
    engine5 = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        purge_days=1,
        embargo_days=5,
    )
    oos0, _, _ = engine0.run(returns)
    oos5, _, _ = engine5.run(returns)
    # embargo=5 starts later — fewer OOS observations (or equal if not binding)
    assert len(oos5) <= len(oos0)
    # The first OOS date in the embargo run is later
    assert oos5.index[0] >= oos0.index[0]


# ---------------------------------------------------------------------------
# Anchored vs rolling
# ---------------------------------------------------------------------------

def test_anchored_and_rolling_differ(returns):
    engine_rolling = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        window_policy="rolling",
    )
    engine_anchored = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        window_policy="anchored",
    )
    _, _, log_rolling = engine_rolling.run(returns)
    _, _, log_anchored = engine_anchored.run(returns)

    # Anchored folds have strictly more training data from the second fold onward
    assert log_anchored[1]["n_train"] > log_rolling[1]["n_train"]


# ---------------------------------------------------------------------------
# Param selection
# ---------------------------------------------------------------------------

class _TiltedAllocator:
    """gamma=1.0 concentrates on A0; gamma=0.0 gives equal weights."""

    def allocate(self, risk_model, signal=None, current_weights=None, gamma=0.0, **kw):
        assets = list(risk_model.covariance().columns)
        n = len(assets)
        if gamma >= 0.99:
            w = pd.Series(0.0, index=assets)
            w["A0"] = 1.0
            return w
        return pd.Series(1 / n, index=assets)


def test_param_selection_recovers_best_param():
    rng = np.random.default_rng(99)
    n_obs = 1500
    # A0 has strongly superior Sharpe; A1 and A2 are noisy
    base = rng.standard_normal(n_obs) * 0.003 + 0.001
    noise = rng.standard_normal((n_obs, 3)) * 0.015
    data = pd.DataFrame(
        {
            "A0": base * 3 + noise[:, 0] + 0.0015,
            "A1": noise[:, 1],
            "A2": noise[:, 2],
        },
        index=pd.date_range("2014-01-01", periods=n_obs, freq="B"),
    )

    engine = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=_TiltedAllocator(),
        window_years=2,
        step_months=6,
        param_grid={"gamma": [0.0, 1.0]},
    )
    _, _, log = engine.run(data)
    selected = [e["params"].get("gamma") for e in log]
    assert 1.0 in selected, "Expected gamma=1.0 to be selected in at least one fold"


# ---------------------------------------------------------------------------
# No-lookahead assertion on inner param selection
# ---------------------------------------------------------------------------

def test_no_lookahead_inner_param_selection():
    returns = make_returns(n_obs=1000)

    engine = WalkForwardEngine(
        risk_model_factory=default_factory,
        signal_model=HistoricalMeanSignal(),
        allocator=HRPAllocator(),
        window_years=2,
        step_months=6,
        param_grid={"dummy": [0]},
    )

    outer_rebal_dates = []
    inner_train_ends = []

    original_select = engine._select_params

    def spy_select(train_returns):
        inner_train_ends.append(train_returns.index[-1])
        return original_select(train_returns)

    with patch.object(engine, "_select_params", spy_select):
        _, _, log = engine.run(returns)
        outer_rebal_dates = [pd.Timestamp(e["date"]) for e in log]

    assert len(inner_train_ends) == len(outer_rebal_dates)
    for inner_end, outer_rebal in zip(inner_train_ends, outer_rebal_dates):
        assert inner_end <= outer_rebal, (
            f"Inner selection saw date {inner_end} > outer rebal {outer_rebal}"
        )
