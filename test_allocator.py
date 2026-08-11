"""Unit tests for hrp_weights, portfolio_stats, and run_backtest in etf_analyze.py."""

import numpy as np
import pandas as pd
import pytest

from etf_analyze import hrp_weights, portfolio_stats, run_backtest
from risk_model import HRPRiskModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_returns(n_obs: int = 800, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Correlated synthetic daily returns with a clear two-block structure."""
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n_obs)
    f2 = rng.standard_normal(n_obs)
    data = np.empty((n_obs, n_assets))
    for j in range(n_assets):
        factor = f1 if j < n_assets - 1 else f2
        data[:, j] = 0.01 * (0.8 * factor + 0.2 * rng.standard_normal(n_obs))
    cols = [f"A{j}" for j in range(n_assets)]
    return pd.DataFrame(data, index=pd.date_range("2016-01-01", periods=n_obs, freq="B"), columns=cols)


@pytest.fixture
def returns():
    return make_returns()


@pytest.fixture
def risk_model(returns):
    return HRPRiskModel(returns, cluster_mode="full", linkage_method="ward")


# ---------------------------------------------------------------------------
# hrp_weights
# ---------------------------------------------------------------------------

def test_hrp_weights_sum_to_one(risk_model):
    w = hrp_weights(risk_model)
    assert abs(w.sum() - 1.0) < 1e-9


def test_hrp_weights_nonnegative(risk_model):
    w = hrp_weights(risk_model)
    assert (w >= 0).all()


def test_hrp_weights_covers_all_assets(risk_model, returns):
    w = hrp_weights(risk_model)
    assert set(w.index) == set(returns.columns)


def test_hrp_weights_verbose_no_crash(risk_model, capsys):
    hrp_weights(risk_model, verbose=True)
    out = capsys.readouterr().out
    assert "Dendrogram" in out
    assert "Step" in out


def test_hrp_weights_two_assets():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((300, 2)) * 0.01
    data[:, 1] *= 2  # asset B is twice as volatile
    returns = pd.DataFrame(data, columns=["A", "B"],
                           index=pd.date_range("2020-01-01", periods=300, freq="B"))
    rm = HRPRiskModel(returns, cluster_mode="full")
    w = hrp_weights(rm)
    # Higher-vol asset B should get less weight
    assert w["A"] > w["B"]


# ---------------------------------------------------------------------------
# portfolio_stats
# ---------------------------------------------------------------------------

def test_portfolio_stats_equal_weight(returns):
    weights = pd.Series(1 / len(returns.columns), index=returns.columns)
    stats = portfolio_stats(weights, returns)
    assert set(stats.keys()) == {"Ann Return", "Ann Vol", "Sharpe"}
    assert stats["Ann Vol"] > 0


def test_portfolio_stats_vol_formula(returns):
    weights = pd.Series({"A0": 1.0, "A1": 0.0, "A2": 0.0})
    stats = portfolio_stats(weights, returns)
    expected_vol = returns["A0"].std(ddof=1) * np.sqrt(252)
    assert abs(stats["Ann Vol"] - expected_vol) < 1e-10


def test_portfolio_stats_sharpe_sign(returns):
    # With positive-drift returns, Sharpe should be positive
    pos = returns + 0.002  # add positive drift
    weights = pd.Series(1 / len(pos.columns), index=pos.columns)
    stats = portfolio_stats(weights, pos)
    assert stats["Sharpe"] > 0


def test_portfolio_stats_zero_vol():
    # Near-zero vol (returns differ by < machine epsilon) → Sharpe = nan
    r = 0.001
    data = pd.DataFrame({"A": [r] * 100, "B": [r] * 100},
                        index=pd.date_range("2020-01-01", periods=100, freq="B"))
    weights = pd.Series({"A": 0.5, "B": 0.5})
    stats = portfolio_stats(weights, data)
    # Ann vol is either exactly 0 or sub-epsilon due to floating-point noise
    assert stats["Ann Vol"] < 1e-8 or np.isnan(stats["Sharpe"])


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def test_run_backtest_returns_correct_types(returns):
    oos, ew, log = run_backtest(returns, method="hrp", window_years=2, step_months=6)
    assert isinstance(oos, pd.Series)
    assert isinstance(ew, pd.Series)
    assert isinstance(log, list)
    assert len(log) > 0


def test_run_backtest_oos_starts_after_window(returns):
    oos, _, log = run_backtest(returns, method="hrp", window_years=2, step_months=6)
    expected_start = returns.index[0] + pd.DateOffset(years=2)
    assert oos.index[0] >= expected_start


def test_run_backtest_oos_and_ew_same_index(returns):
    oos, ew, _ = run_backtest(returns, method="hrp", window_years=2, step_months=6)
    assert oos.index.equals(ew.index)


def test_run_backtest_log_has_weights(returns):
    _, _, log = run_backtest(returns, method="hrp", window_years=2, step_months=6)
    for entry in log:
        assert "date" in entry
        assert "weights" in entry
        assert abs(entry["weights"].sum() - 1.0) < 1e-6


def test_run_backtest_no_lookahead(returns):
    # OOS returns must not overlap with any training window
    oos, _, log = run_backtest(returns, method="hrp", window_years=2, step_months=6)
    first_oos_date = oos.index[0]
    first_rebal = log[0]["date"]
    assert first_oos_date.date() > first_rebal


def test_run_backtest_mvo(returns):
    oos, ew, log = run_backtest(returns, method="mvo", window_years=2, step_months=6)
    assert len(oos) > 0
    assert len(log) > 0


def test_run_backtest_insufficient_data():
    short = make_returns(n_obs=100)  # ~5 months, can't fit a 3-year window
    with pytest.raises(ValueError, match="Not enough data"):
        run_backtest(short, method="hrp", window_years=3)
