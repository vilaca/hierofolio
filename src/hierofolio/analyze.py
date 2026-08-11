#!/usr/bin/env python
"""hierofolio analysis — returns and summary stats from the price DB.

Usage:
    hierofolio analyze returns              # returns for all ETFs
    hierofolio analyze returns IE00BM67HK77 # returns for a single ISIN
    hierofolio analyze summary              # annualized stats + correlation
    hierofolio analyze allocate             # HRP portfolio weights
    hierofolio analyze allocate --method mvo    # MVO weights
    hierofolio analyze allocate --method robust # Robust weights
    hierofolio analyze backtest             # rolling-window out-of-sample backtest

Reads the SQLite DB populated by the fetch command; run that first.
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hierofolio.allocators import ALLOCATORS, HRPAllocator
from hierofolio.backtest import WalkForwardEngine, _rebalance_cost  # noqa: F401
from hierofolio.common import DEFAULT_CONFIG, DEFAULT_CURRENCY_META, DEFAULT_DB
from hierofolio.risk_model import HRPRiskModel
from hierofolio.signals import HistoricalMeanSignal

# broker_profiles.yaml is hand-authored config; resolve it against the project
# root (see hierofolio.common) rather than the package dir.
DEFAULT_BROKER_PROFILES = str(Path(__file__).resolve().parents[2] / "config" / "broker_profiles.yaml")


def load_broker_profiles(path: str = DEFAULT_BROKER_PROFILES) -> dict:
    """Load broker cost profiles from YAML (flat_eur + bps_per_side per broker)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


BROKER_PROFILES = load_broker_profiles()


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


def cost_currency_warning(isins, flat_fee_eur, min_fee_eur,
                          meta_path: str = DEFAULT_CURRENCY_META):
    """Warn if a EUR-denominated fee is applied to a non-EUR panel.

    Percentage (bps) costs are currency-invariant, but the flat and minimum
    fees are absolute euros — only meaningful when both `--portfolio-size` and
    the panel are in EUR. Returns a message when such a fee meets a panel whose
    known quote currency isn't purely EUR (there is no FX conversion), else None.
    """
    if flat_fee_eur <= 0 and min_fee_eur <= 0:
        return None
    ccy = read_currencies(meta_path)
    known = {ccy.get(isin, '') for isin in isins} - {''}
    if known and known != {'EUR'}:
        return (f"Flat/minimum fees are in EUR but the panel quotes in "
                f"{sorted(known)}; euro fees and --portfolio-size assume a EUR "
                f"book (no FX conversion is applied).")
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
    return HRPAllocator().allocate(risk_model, verbose=verbose)


def sri_class(ann_vol: float) -> int:
    """EU PRIIP Summary Risk Indicator (1–7) from annualised volatility."""
    for cls, threshold in enumerate([0.005, 0.05, 0.12, 0.20, 0.30, 0.80], start=1):
        if ann_vol < threshold:
            return cls
    return 7


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
    flat_fee_eur: float = 0.0,
    cost_bps_per_side: float = 0.0,
    min_fee_eur: float = 0.0,
    portfolio_size_eur: float = 10_000.0,
    shrinkage_method: str = "constant_correlation",
    shrinkage_intensity: float = 0.3,
    linkage_method: str = "ward",
    signal_model=None,
    gamma: float = 0.5,
) -> tuple:
    """Rolling-window backtest. Returns (out-of-sample daily returns, rebalance log)."""
    if signal_model is None:
        signal_model = HistoricalMeanSignal()

    def _risk_model_factory(train):
        return HRPRiskModel(
            returns=train,
            shrinkage_method=shrinkage_method,
            shrinkage_intensity=shrinkage_intensity,
            cluster_mode="full",
            linkage_method=linkage_method,
        )

    method_params: dict = {}
    if method in ("mvo", "robust"):
        method_params["max_weight"] = max_weight
        method_params["risk_aversion"] = risk_aversion
    if method == "robust":
        method_params["robustness_penalty"] = robustness_penalty
    if method == "schur-hrp":
        method_params["gamma"] = gamma
        if max_weight is not None:
            method_params["max_weight"] = max_weight

    engine = WalkForwardEngine(
        risk_model_factory=_risk_model_factory,
        signal_model=signal_model,
        allocator=ALLOCATORS[method],
        window_years=window_years,
        step_months=step_months,
        window_policy="rolling",
        purge_days=1,
        embargo_days=0,
        param_grid=None,
        flat_fee_eur=flat_fee_eur,
        cost_bps_per_side=cost_bps_per_side,
        min_fee_eur=min_fee_eur,
        portfolio_size_eur=portfolio_size_eur,
    )
    engine._fixed_params = method_params
    return engine.run(returns)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="hierofolio analyze",
        description="Returns and summary statistics from the price DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show the returns DataFrame (tail) for all ETFs
  hierofolio analyze returns

  # Returns for a single ETF
  hierofolio analyze returns IE00BM67HK77

  # Annualized stats and correlation matrix
  hierofolio analyze summary

  # Portfolio weights (HRP by default)
  hierofolio analyze allocate
  hierofolio analyze allocate --method mvo
  hierofolio analyze allocate --method robust --max-weight 0.5

  # Rolling-window out-of-sample backtest (3-year window, quarterly rebalance)
  hierofolio analyze backtest
  hierofolio analyze backtest --method mvo --window 5 --step 6
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
        '--method', choices=['hrp', 'schur-hrp', 'mvo', 'robust'], default='hrp',
        help='Optimization method (default: hrp)'
    )
    allocate_parser.add_argument(
        '--gamma', type=float, default=0.5, metavar='γ',
        help='Schur-HRP cross-block weight (0=plain HRP, higher→toward min-variance; default 0.5)'
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
    allocate_parser.add_argument(
        '--shrinkage-method', choices=['ledoit_wolf', 'constant_correlation', 'identity'],
        default='constant_correlation', help='Covariance shrinkage method (default: constant_correlation)'
    )
    allocate_parser.add_argument(
        '--shrinkage-intensity', type=float, default=0.3, metavar='α',
        help='Shrinkage intensity 0–1 (default: 0.3; ignored for ledoit_wolf)'
    )
    allocate_parser.add_argument(
        '--linkage-method', choices=['ward', 'average', 'complete', 'single'],
        default='ward', help='Hierarchical clustering linkage (default: ward)'
    )

    backtest_parser = subparsers.add_parser('backtest', help='Rolling-window out-of-sample backtest')
    backtest_parser.add_argument(
        '--method', choices=['hrp', 'schur-hrp', 'mvo', 'robust'], default='hrp',
        help='Optimization method (default: hrp)'
    )
    backtest_parser.add_argument(
        '--gamma', type=float, default=0.5, metavar='γ',
        help='Schur-HRP cross-block weight (0=plain HRP, higher→toward min-variance; default 0.5)'
    )
    backtest_parser.add_argument('--window', type=int, default=3, metavar='YEARS',
                                 help='Training window in years (default: 3)')
    backtest_parser.add_argument('--step', type=int, default=3, metavar='MONTHS',
                                 help='Rebalance interval in months (default: 3)')
    backtest_parser.add_argument('--max-weight', type=float, default=None, metavar='W',
                                 help='Max weight per asset for mvo/robust')
    backtest_parser.add_argument('--risk-aversion', type=float, default=1.0, metavar='λ')
    backtest_parser.add_argument('--robustness-penalty', type=float, default=1.0, metavar='ρ')
    backtest_parser.add_argument(
        '--shrinkage-method', choices=['ledoit_wolf', 'constant_correlation', 'identity'],
        default='constant_correlation', help='Covariance shrinkage method (default: constant_correlation)'
    )
    backtest_parser.add_argument(
        '--shrinkage-intensity', type=float, default=0.3, metavar='α',
        help='Shrinkage intensity 0–1 (default: 0.3; ignored for ledoit_wolf)'
    )
    backtest_parser.add_argument(
        '--linkage-method', choices=['ward', 'average', 'complete', 'single'],
        default='ward', help='Hierarchical clustering linkage (default: ward)'
    )
    backtest_parser.add_argument(
        '--broker', choices=list(BROKER_PROFILES), default=None, metavar='BROKER',
        help=f"Broker cost profile ({', '.join(BROKER_PROFILES)}); see config/broker_profiles.yaml"
    )
    backtest_parser.add_argument(
        '--portfolio-size', type=float, default=10_000.0, metavar='EUR',
        help='Portfolio size in EUR for flat-fee cost calculation (default: 10000)'
    )
    backtest_parser.add_argument(
        '--cost-bps', type=float, default=None, metavar='BPS',
        help='Manual round-trip cost override in bps (overrides --broker)'
    )

    args = parser.parse_args(argv)

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
                shrinkage_method=args.shrinkage_method,
                shrinkage_intensity=args.shrinkage_intensity,
                cluster_mode="full",
                linkage_method=args.linkage_method,
            )

            method = args.method
            mu, _ = HistoricalMeanSignal().signal(returns)
            method_params = {
                "verbose": args.verbose,
                "max_weight": args.max_weight,
                "risk_aversion": args.risk_aversion,
                "robustness_penalty": args.robustness_penalty,
                "gamma": args.gamma,
            }
            weights = ALLOCATORS[method].allocate(
                risk_model, signal=mu, current_weights=None, **method_params
            )

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
            print(f"  SRI         {sri_class(stats['Ann Vol']):>8}/7  (EU PRIIP risk class)")
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

            # Resolve cost model: --cost-bps overrides --broker
            if args.cost_bps is not None:
                flat_fee_eur = 0.0
                cost_bps_per_side = args.cost_bps / 2
                min_fee_eur = 0.0
                cost_label = f"manual ({args.cost_bps} bps round-trip)"
            elif args.broker:
                profile = BROKER_PROFILES[args.broker]
                flat_fee_eur = profile['flat_eur']
                cost_bps_per_side = profile['bps_per_side']
                min_fee_eur = profile.get('min_eur', 0.0)
                cost_label = f"{profile['name']} — {profile['note']}"
            else:
                flat_fee_eur = 0.0
                cost_bps_per_side = 0.0
                min_fee_eur = 0.0
                cost_label = "none (use --broker or --cost-bps to model transaction costs)"

            cost_ccy_warning = cost_currency_warning(
                returns.columns, flat_fee_eur, min_fee_eur, args.currency_meta
            )
            if cost_ccy_warning:
                print(f"⚠ {cost_ccy_warning}")

            oos_returns, ew_returns, log = run_backtest(
                returns=returns,
                method=args.method,
                window_years=args.window,
                step_months=args.step,
                max_weight=args.max_weight,
                risk_aversion=args.risk_aversion,
                robustness_penalty=args.robustness_penalty,
                flat_fee_eur=flat_fee_eur,
                cost_bps_per_side=cost_bps_per_side,
                min_fee_eur=min_fee_eur,
                portfolio_size_eur=args.portfolio_size,
                shrinkage_method=args.shrinkage_method,
                shrinkage_intensity=args.shrinkage_intensity,
                linkage_method=args.linkage_method,
                gamma=args.gamma,
            )

            names = read_names(DEFAULT_CONFIG)
            isins = returns.columns.tolist()
            total_cost = sum(e['cost'] for e in log)

            print("\n" + "=" * 70)
            print(f"Hierofolio Backtest — {args.method.upper()}")
            print(f"Window: {args.window}y  Step: {args.step}m  Periods: {len(log)}")
            print(f"Cost:   {cost_label}")
            if flat_fee_eur > 0 or min_fee_eur > 0:
                print(f"        (portfolio size: €{args.portfolio_size:,.0f})")
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
            if total_cost > 0:
                print(f"  {'Cost drag (total)':20}  {-total_cost:>11.2%}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
