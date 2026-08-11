import numpy as np
import pandas as pd
import pytest

from hierofolio.signals import HistoricalMeanSignal


def make_returns(n_obs: int = 300, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_obs, n_assets)) * 0.01
    return pd.DataFrame(
        data,
        index=pd.date_range("2020-01-01", periods=n_obs, freq="B"),
        columns=[f"A{i}" for i in range(n_assets)],
    )


def test_historical_mean_signal_matches_old_formula():
    r = make_returns()
    mu, uncertainty = HistoricalMeanSignal().signal(r)
    expected = (1 + r.mean()) ** 252 - 1
    assert np.allclose(mu.values, expected.values)


def test_historical_mean_signal_uncertainty_is_none():
    r = make_returns()
    _, uncertainty = HistoricalMeanSignal().signal(r)
    assert uncertainty is None


def test_historical_mean_signal_index_matches_columns():
    r = make_returns()
    mu, _ = HistoricalMeanSignal().signal(r)
    assert list(mu.index) == list(r.columns)
