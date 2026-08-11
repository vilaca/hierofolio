"""Walk-forward backtesting engine."""

from typing import Callable, Optional
import itertools

import numpy as np
import pandas as pd

from hierofolio.risk_model import RiskModel
from hierofolio.signals import HistoricalMeanSignal, SignalModel


def _rebalance_cost(
    w_old: pd.Series,
    w_new: pd.Series,
    portfolio_size_eur: float,
    flat_fee_eur: float,
    cost_bps_per_side: float,
    min_fee_eur: float = 0.0,
) -> float:
    """Transaction cost as a fraction of portfolio value for one rebalance.

    Per traded asset the cost is ``flat_fee_eur + bps × notional`` but never
    less than ``min_fee_eur`` — modelling brokers (e.g. IBKR) whose commission
    has a per-order floor that dominates on small trades.
    """
    deltas = (w_new - w_old).abs()
    traded = deltas[deltas > 1e-4]
    cost_eur = sum(
        max(min_fee_eur, flat_fee_eur + cost_bps_per_side / 10_000 * delta * portfolio_size_eur)
        for delta in traded
    )
    return cost_eur / portfolio_size_eur


class WalkForwardEngine:
    """Rolling or anchored walk-forward backtest engine.

    Parameters
    ----------
    risk_model_factory:
        Callable that takes a returns DataFrame and returns a RiskModel.
    signal_model:
        SignalModel used to compute (mu, uncertainty) from training returns.
    allocator:
        Allocator used to turn a risk model + signal into portfolio weights.
    window_years:
        Length of the training window in years (rolling mode).
    step_months:
        Rebalance frequency in months.
    window_policy:
        "rolling" uses a fixed-length trailing window; "anchored" always starts
        from the first available observation (expanding window).
    purge_days:
        Number of index positions to drop from the start of the hold period
        (starting at rebal_date). Default 1 matches the current run_backtest
        behavior of excluding rebal_date itself from the OOS period.
    embargo_days:
        Additional index positions to skip after purge_days before the hold
        period starts.
    param_grid:
        Optional dict mapping param names to lists of candidate values. When
        provided, an inner walk-forward over the training window is run for each
        candidate combination and the combination with the best inner-OOS Sharpe
        is used for the outer fold.
    """

    def __init__(
        self,
        risk_model_factory: Callable[[pd.DataFrame], RiskModel],
        signal_model: SignalModel,
        allocator,
        window_years: int = 3,
        step_months: int = 3,
        window_policy: str = "rolling",
        purge_days: int = 1,
        embargo_days: int = 0,
        param_grid: Optional[dict] = None,
        flat_fee_eur: float = 0.0,
        cost_bps_per_side: float = 0.0,
        min_fee_eur: float = 0.0,
        portfolio_size_eur: float = 10_000.0,
    ):
        self.risk_model_factory = risk_model_factory
        self.signal_model = signal_model
        self.allocator = allocator
        self.window_years = window_years
        self.step_months = step_months
        self.window_policy = window_policy
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.param_grid = param_grid
        self.flat_fee_eur = flat_fee_eur
        self.cost_bps_per_side = cost_bps_per_side
        self.min_fee_eur = min_fee_eur
        self.portfolio_size_eur = portfolio_size_eur

    def _generate_rebalance_dates(self, idx: pd.DatetimeIndex) -> list:
        d = idx[0] + pd.DateOffset(years=self.window_years)
        dates = []
        while d <= idx[-1]:
            prior = idx[idx <= d]
            if len(prior):
                dates.append(prior[-1])
            d += pd.DateOffset(months=self.step_months)
        return dates

    def _select_params(self, train_returns: pd.DataFrame) -> dict:
        """Inner walk-forward param selection over train_returns."""
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))
        best_sharpe = -np.inf
        best_params = dict(zip(keys, combos[0]))

        inner_window = max(1, self.window_years - 1)
        inner_engine = WalkForwardEngine(
            risk_model_factory=self.risk_model_factory,
            signal_model=self.signal_model,
            allocator=self.allocator,
            window_years=inner_window,
            step_months=self.step_months,
            window_policy=self.window_policy,
            purge_days=self.purge_days,
            embargo_days=self.embargo_days,
            param_grid=None,
            flat_fee_eur=self.flat_fee_eur,
            cost_bps_per_side=self.cost_bps_per_side,
            min_fee_eur=self.min_fee_eur,
            portfolio_size_eur=self.portfolio_size_eur,
        )

        for combo in combos:
            params = dict(zip(keys, combo))
            inner_engine._fixed_params = params
            try:
                inner_oos, _, _ = inner_engine.run(train_returns)
            except (ValueError, Exception):
                continue
            if len(inner_oos) == 0:
                continue
            vol = float(inner_oos.std(ddof=1) * np.sqrt(252))
            ret = float((1 + inner_oos).prod() ** (252 / len(inner_oos)) - 1)
            sharpe = ret / vol if vol > 1e-10 else -np.inf
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        return best_params

    def run(self, returns: pd.DataFrame) -> tuple[pd.Series, pd.Series, list]:
        """Run the walk-forward backtest.

        Returns
        -------
        oos_returns : pd.Series
            Daily out-of-sample portfolio returns.
        ew_returns : pd.Series
            Equal-weight benchmark returns over the same period.
        log : list of dict
            Per-rebalance records with keys: date, n_train, weights, cost, params.
        """
        idx = returns.index
        rebalance_dates = self._generate_rebalance_dates(idx)

        if not rebalance_dates:
            raise ValueError(
                f"Not enough data for a {self.window_years}-year window "
                f"(available: {(idx[-1] - idx[0]).days / 365:.1f} years)."
            )

        fixed_params = getattr(self, "_fixed_params", {})

        oos_segments: list[pd.Series] = []
        ew_segments: list[pd.Series] = []
        log: list[dict] = []
        prev_weights: Optional[pd.Series] = None

        for i, rebal_date in enumerate(rebalance_dates):
            train_start = (
                rebal_date - pd.DateOffset(years=self.window_years)
                if self.window_policy == "rolling"
                else idx[0]
            )
            train = returns.loc[train_start:rebal_date]
            if len(train) < 60:
                continue

            if self.param_grid is not None:
                selected_params = self._select_params(train)
            else:
                selected_params = {}

            all_params = {**selected_params, **fixed_params}

            risk_model = self.risk_model_factory(train)
            mu, _ = self.signal_model.signal(train)
            weights = self.allocator.allocate(
                risk_model, signal=mu, current_weights=None, **all_params
            )

            next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else idx[-1]

            rebal_pos = idx.get_loc(rebal_date)
            hold_start_pos = rebal_pos + self.purge_days + self.embargo_days
            if hold_start_pos >= len(idx):
                continue
            hold = returns.loc[idx[hold_start_pos] : next_rebal]
            if hold.empty:
                continue

            period_oos = hold @ weights.reindex(hold.columns).fillna(0)

            w_old = prev_weights if prev_weights is not None else pd.Series(
                1 / len(weights), index=weights.index
            )
            cost = _rebalance_cost(
                w_old, weights, self.portfolio_size_eur, self.flat_fee_eur,
                self.cost_bps_per_side, self.min_fee_eur,
            )
            period_oos.iloc[0] -= cost

            oos_segments.append(period_oos)
            ew_segments.append(hold.mean(axis=1))
            log.append({
                "date": rebal_date.date(),
                "n_train": len(train),
                "weights": weights,
                "cost": cost,
                "params": all_params,
            })
            prev_weights = weights

        if not oos_segments:
            raise ValueError("No out-of-sample periods were generated.")

        return pd.concat(oos_segments), pd.concat(ew_segments), log
