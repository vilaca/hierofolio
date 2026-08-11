#!/usr/bin/env python
"""hierofolio analysis — returns and summary stats from the price DB.

Usage:
    ./etf_analyze.py returns              # returns for all ETFs
    ./etf_analyze.py returns IE00BM67HK77 # returns for a single ISIN
    ./etf_analyze.py summary              # annualized stats + correlation
    ./etf_analyze.py allocate             # HRP portfolio weights
    ./etf_analyze.py allocate --method mvo    # MVO weights
    ./etf_analyze.py allocate --method robust # Robust weights
    ./etf_analyze.py backtest             # rolling-window out-of-sample backtest

Reads the SQLite DB populated by etf_fetch.py; run that first.
"""

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import yaml

from etf_common import DEFAULT_CONFIG, DEFAULT_CURRENCY_META, DEFAULT_DB
from risk_model import ConstrainedMVOOptimizer, HRPRiskModel, RobustOptimizer


def read_prices(db_path: str, isin: str = None) -> pd.DataFrame:
    """Load a wide (date x isin) close-price frame from the DB."""
    with sqlite3.connect(db_path) as conn:
        if isin:
            df = pd.read_sql_query(
                "SELECT date, close FROM prices WHERE isin = ? ORDER BY date",
                conn,
                params=(isin,),
                index_col='date',
                parse_dates=['date']
            )
            return df.rename(columns={'close': isin})
        df = pd.read_sql_query(
            "SELECT isin, date, close FROM prices ORDER BY date, isin",
            conn,
            parse_dates=['date']
        )
        return df.pivot(index='date', columns='isin', values='close')


def read_returns(db_path: str, isin: str = None) -> pd.DataFrame:
    """Daily simple returns derived from the stored prices."""
    return read_prices(db_path, isin).pct_change().dropna()


def read_currencies(meta_path: str = DEFAULT_CURRENCY_META) -> dict:
    """ISIN -> pinned quote currency, from the fetch sidecar (empty if absent)."""
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    return {isin: (entry or {}).get('currency', '') for isin, entry in meta.items()}


def currency_warning(isins, meta_path: str = DEFAULT_CURRENCY_META):
    """Warn if the given ISINs aren't all in one currency.

    Returns a message string when the panel mixes currencies (its per-asset
    returns aren't comparable without FX conversion), else None.
    """
    ccy = read_currencies(meta_path)
    present = {isin: ccy.get(isin, '') for isin in isins}
    distinct = {c for c in present.values() if c}
    if len(distinct) > 1:
        return (f"Mixed currencies across ISINs {present}; returns/covariance "
                f"are not comparable without FX conversion.")
    return None


def quality_report(prices: pd.DataFrame) -> dict:
    """Data-quality metrics for a long (isin, date, close) price frame.

    Returns raw counts/maxima so callers decide what's acceptable:
    duplicate keys, null/non-positive closes, weekend rows, the largest
    per-ISIN calendar gap (days), and the largest absolute daily return.
    """
    p = prices.copy()
    p['date'] = pd.to_datetime(p['date'])
    p = p.sort_values(['isin', 'date'])

    def _max_gap(dates: pd.Series) -> int:
        gaps = dates.diff().dt.days.dropna()
        return int(gaps.max()) if len(gaps) else 0

    def _max_abs_return(close: pd.Series) -> float:
        rets = close.pct_change().abs().dropna()
        return float(rets.max()) if len(rets) else 0.0

    gaps = p.groupby('isin')['date'].apply(_max_gap)
    rets = p.groupby('isin')['close'].apply(_max_abs_return)
    return {
        'rows': int(len(p)),
        'duplicates': int(p.duplicated(subset=['isin', 'date']).sum()),
        'nulls': int(p['close'].isna().sum()),
        'non_positive': int((p['close'] <= 0).sum()),
        'weekend_rows': int((p['date'].dt.weekday >= 5).sum()),
        'max_gap_days': int(gaps.max()) if len(gaps) else 0,
        'max_abs_return': float(rets.max()) if len(rets) else 0.0,
    }


def read_long(db_path: str) -> pd.DataFrame:
    """Full price table in long (isin, date, close) form."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT isin, date, close FROM prices", conn, parse_dates=['date']
        )


def read_names(config_path: str = DEFAULT_CONFIG) -> dict:
    """ISIN -> name from the universe config (empty dict if absent)."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return {isin: entry.get('name', isin) for isin, entry in data.get('etfs', {}).items()}


def hrp_weights(risk_model: HRPRiskModel, verbose: bool = False) -> pd.Series:
    """HRP weights via inverse-variance recursive bisection on the dendrogram."""
    cov = risk_model.covariance()
    assets = risk_model._leaf_order  # dendrogram-ordered for quasi-diagonalization

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
                    ls = '{' + ', '.join(left) + '}'
                    rs = '{' + ', '.join(right) + '}'
                    print(f"  Step {step}: {ls} vs {rs}")
                    print(f"    {ls:<40} branch vol {np.sqrt(lv * 252)*100:5.1f}%  →  {alpha:.1%} of parent budget")
                    print(f"    {rs:<40} branch vol {np.sqrt(rv * 252)*100:5.1f}%  →  {1-alpha:.1%} of parent budget")
                next_clusters += [left, right]
        clusters = next_clusters
    return weights / weights.sum()


def portfolio_stats(weights: pd.Series, returns: pd.DataFrame) -> dict:
    """Annualized return, vol, and Sharpe for a set of weights."""
    w = weights.reindex(returns.columns).fillna(0).values
    port_returns = returns.values @ w
    ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252))
    ann_return = float((1 + port_returns).prod() ** (252 / len(port_returns)) - 1)
    sharpe = ann_return / ann_vol if ann_vol > 1e-10 else float("nan")
    return {"Ann Return": ann_return, "Ann Vol": ann_vol, "Sharpe": sharpe}


def run_backtest(
    returns: pd.DataFrame,
    method: str,
    window_years: int = 3,
    step_months: int = 3,
    max_weight: float = None,
    risk_aversion: float = 1.0,
    robustness_penalty: float = 1.0,
) -> tuple:
    """Rolling-window backtest. Returns (out-of-sample daily returns, rebalance log)."""
    idx = returns.index

    # Generate rebalance dates by stepping forward from the first full window
    d = idx[0] + pd.DateOffset(years=window_years)
    rebalance_dates = []
    while d <= idx[-1]:
        prior = idx[idx <= d]
        if len(prior):
            rebalance_dates.append(prior[-1])
        d += pd.DateOffset(months=step_months)

    if not rebalance_dates:
        raise ValueError(
            f"Not enough data for a {window_years}-year window "
            f"(available: {(idx[-1] - idx[0]).days / 365:.1f} years)."
        )

    oos_segments = []
    ew_segments = []
    log = []

    for i, rebal_date in enumerate(rebalance_dates):
        train_start = rebal_date - pd.DateOffset(years=window_years)
        train = returns.loc[train_start:rebal_date]
        if len(train) < 60:
            continue

        risk_model = HRPRiskModel(
            returns=train,
            shrinkage_method="constant_correlation",
            shrinkage_intensity=0.3,
            cluster_mode="full",
            linkage_method="ward",
        )

        if method == 'hrp':
            weights = hrp_weights(risk_model)
        else:
            alpha = (1 + train.mean()) ** 252 - 1
            if method == 'mvo':
                optimizer = ConstrainedMVOOptimizer(
                    risk_model=risk_model, alpha=alpha, risk_aversion=risk_aversion,
                )
            else:
                optimizer = RobustOptimizer(
                    risk_model=risk_model, alpha=alpha, risk_aversion=risk_aversion,
                    robustness_penalty=robustness_penalty,
                )
            weights = optimizer.solve(max_weight=max_weight)

        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else idx[-1]
        hold = returns.loc[rebal_date:next_rebal].iloc[1:]  # exclude rebal_date itself
        if hold.empty:
            continue

        oos_segments.append(hold @ weights.reindex(hold.columns).fillna(0))
        ew_segments.append(hold.mean(axis=1))
        log.append({'date': rebal_date.date(), 'n_train': len(train), 'weights': weights})

    if not oos_segments:
        raise ValueError("No out-of-sample periods were generated.")

    return pd.concat(oos_segments), pd.concat(ew_segments), log


def main():
    parser = argparse.ArgumentParser(
        prog="etf_analyze",
        description="Returns and summary statistics from the price DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show the returns DataFrame (tail) for all ETFs
  ./etf_analyze.py returns

  # Returns for a single ETF
  ./etf_analyze.py returns IE00BM67HK77

  # Annualized stats and correlation matrix
  ./etf_analyze.py summary

  # Portfolio weights (HRP by default)
  ./etf_analyze.py allocate
  ./etf_analyze.py allocate --method mvo
  ./etf_analyze.py allocate --method robust --max-weight 0.5

  # Rolling-window out-of-sample backtest (3-year window, quarterly rebalance)
  ./etf_analyze.py backtest
  ./etf_analyze.py backtest --method mvo --window 5 --step 6
        """
    )

    parser.add_argument('--db', '-d', default=DEFAULT_DB, help='Database file path')
    parser.add_argument('--currency-meta', default=DEFAULT_CURRENCY_META,
                        help='Pinned ftgo resolution / currency sidecar path')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    returns_parser = subparsers.add_parser('returns', help='Show returns DataFrame')
    returns_parser.add_argument('isin', nargs='?', help='ISIN to show (all if omitted)')

    subparsers.add_parser('summary', help='Show data summary')

    allocate_parser = subparsers.add_parser('allocate', help='Compute portfolio weights')
    allocate_parser.add_argument(
        '--method', choices=['hrp', 'mvo', 'robust'], default='hrp',
        help='Optimization method (default: hrp)'
    )
    allocate_parser.add_argument(
        '--max-weight', type=float, default=None, metavar='W',
        help='Max weight per asset for mvo/robust (e.g. 0.5)'
    )
    allocate_parser.add_argument(
        '--risk-aversion', type=float, default=1.0, metavar='λ',
        help='Risk aversion for mvo/robust (default: 1.0)'
    )
    allocate_parser.add_argument(
        '--robustness-penalty', type=float, default=1.0, metavar='ρ',
        help='Robustness penalty for robust method (default: 1.0; try 10–100 to see diversification)'
    )
    allocate_parser.add_argument(
        '--verbose', action='store_true',
        help='Show HRP bisection steps (leaf order, branch variances, budget splits)'
    )

    backtest_parser = subparsers.add_parser('backtest', help='Rolling-window out-of-sample backtest')
    backtest_parser.add_argument('--method', choices=['hrp', 'mvo', 'robust'], default='hrp',
                                 help='Optimization method (default: hrp)')
    backtest_parser.add_argument('--window', type=int, default=3, metavar='YEARS',
                                 help='Training window in years (default: 3)')
    backtest_parser.add_argument('--step', type=int, default=3, metavar='MONTHS',
                                 help='Rebalance interval in months (default: 3)')
    backtest_parser.add_argument('--max-weight', type=float, default=None, metavar='W',
                                 help='Max weight per asset for mvo/robust')
    backtest_parser.add_argument('--risk-aversion', type=float, default=1.0, metavar='λ')
    backtest_parser.add_argument('--robustness-penalty', type=float, default=1.0, metavar='ρ')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'returns':
        try:
            returns = read_returns(args.db, args.isin)
            warning = currency_warning(returns.columns, args.currency_meta)
            if warning:
                print(f"⚠ {warning}")
            print(f"Returns for {len(returns.columns)} ETFs")
            print(f"Date range: {returns.index.min()} to {returns.index.max()}")
            print("\nLast 10 returns:")
            print(returns.tail(10).round(6))
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0

    if args.command == 'summary':
        try:
            returns = read_returns(args.db)
            if returns.empty:
                print("No returns data. Run: ./etf_fetch.py")
                return 1

            print("\n" + "=" * 70)
            print("Hierofolio Summary")
            print("=" * 70)

            warning = currency_warning(returns.columns, args.currency_meta)
            if warning:
                print(f"\n⚠ {warning}")

            ann_returns = (1 + returns.mean()) ** 252 - 1
            ann_vol = returns.std() * np.sqrt(252)

            stats = pd.DataFrame({
                'Ann Return': ann_returns,
                'Ann Vol': ann_vol,
                'Sharpe': ann_returns / ann_vol
            })
            print("\nAnnualized Statistics:")
            print(stats.round(4))

            print("\nCorrelation Matrix:")
            print(returns.corr().round(3))

            print("\n" + "=" * 70)
            print("Ready for HRPRiskModel:")
            print("""
    from etf_analyze import read_returns
    from risk_model import HRPRiskModel

    returns = read_returns("hierofolio.db")

    risk_model = HRPRiskModel(
        returns=returns,
        shrinkage_method="constant_correlation",
        shrinkage_intensity=0.3,
        cluster_mode="full",
        linkage_method="ward"
    )
            """)
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0

    if args.command == 'allocate':
        try:
            returns = read_returns(args.db)
            if returns.empty:
                print("No returns data. Run: ./etf_fetch.py")
                return 1

            warning = currency_warning(returns.columns, args.currency_meta)
            if warning:
                print(f"⚠ {warning}")

            risk_model = HRPRiskModel(
                returns=returns,
                shrinkage_method="constant_correlation",
                shrinkage_intensity=0.3,
                cluster_mode="full",
                linkage_method="ward",
            )

            method = args.method
            if method == 'hrp':
                weights = hrp_weights(risk_model, verbose=args.verbose)
            else:
                alpha = (1 + returns.mean()) ** 252 - 1
                if method == 'mvo':
                    optimizer = ConstrainedMVOOptimizer(
                        risk_model=risk_model,
                        alpha=alpha,
                        risk_aversion=args.risk_aversion,
                    )
                else:
                    optimizer = RobustOptimizer(
                        risk_model=risk_model,
                        alpha=alpha,
                        risk_aversion=args.risk_aversion,
                        robustness_penalty=args.robustness_penalty,
                    )
                weights = optimizer.solve(max_weight=args.max_weight)

            stats = portfolio_stats(weights, returns)
            names = read_names(DEFAULT_CONFIG)

            print("\n" + "=" * 70)
            print(f"Hierofolio Allocation — {method.upper()}")
            print("=" * 70)
            print("\nWeights:")
            for isin, w in weights.sort_values(ascending=False).items():
                name = names.get(isin, isin)
                print(f"  {isin}  {w:6.1%}  {name}")
            print(f"\nPortfolio Statistics (in-sample):")
            print(f"  Ann Return  {stats['Ann Return']:8.2%}")
            print(f"  Ann Vol     {stats['Ann Vol']:8.2%}")
            print(f"  Sharpe      {stats['Sharpe']:8.4f}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0

    if args.command == 'backtest':
        try:
            returns = read_returns(args.db)
            if returns.empty:
                print("No returns data. Run: ./etf_fetch.py")
                return 1

            warning = currency_warning(returns.columns, args.currency_meta)
            if warning:
                print(f"⚠ {warning}")

            oos_returns, ew_returns, log = run_backtest(
                returns=returns,
                method=args.method,
                window_years=args.window,
                step_months=args.step,
                max_weight=args.max_weight,
                risk_aversion=args.risk_aversion,
                robustness_penalty=args.robustness_penalty,
            )

            names = read_names(DEFAULT_CONFIG)
            isins = returns.columns.tolist()

            print("\n" + "=" * 70)
            print(f"Hierofolio Backtest — {args.method.upper()}")
            print(f"Window: {args.window}y  Step: {args.step}m  Periods: {len(log)}")
            print("=" * 70)

            # Rebalance log as an aligned table
            weight_rows = {entry['date']: entry['weights'].reindex(isins) for entry in log}
            weight_df = pd.DataFrame(weight_rows).T
            weight_df.index.name = 'Rebalance'
            weight_df.columns = [names.get(c, c) for c in weight_df.columns]
            print("\nRebalance Log:")
            print(weight_df.map(lambda x: f"{x:.1%}").to_string())

            # Out-of-sample stats — strategy vs equal-weight benchmark
            def _stats(r):
                vol = float(r.std(ddof=1) * np.sqrt(252))
                ret = float((1 + r).prod() ** (252 / len(r)) - 1)
                eq = (1 + r).cumprod()
                dd = float(((eq - eq.cummax()) / eq.cummax()).min())
                return ret, vol, ret / vol if vol > 0 else float("nan"), dd

            s_ret, s_vol, s_shr, s_dd = _stats(oos_returns)
            e_ret, e_vol, e_shr, e_dd = _stats(ew_returns)
            start, end = oos_returns.index[0].date(), oos_returns.index[-1].date()
            label = args.method.upper()

            print(f"\nOut-of-Sample Statistics ({start} → {end}):")
            print(f"  {'':20}  {label:>12}  {'Equal Weight':>12}")
            print(f"  {'Ann Return':20}  {s_ret:>11.2%}  {e_ret:>11.2%}")
            print(f"  {'Ann Vol':20}  {s_vol:>11.2%}  {e_vol:>11.2%}")
            print(f"  {'Sharpe':20}  {s_shr:>11.4f}  {e_shr:>11.4f}")
            print(f"  {'Max Drawdown':20}  {s_dd:>11.2%}  {e_dd:>11.2%}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
