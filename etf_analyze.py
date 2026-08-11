#!/usr/bin/env python
"""hierofolio analysis — returns and summary stats from the price DB.

Usage:
    ./etf_analyze.py returns              # returns for all ETFs
    ./etf_analyze.py returns IE00BM67HK77 # returns for a single ISIN
    ./etf_analyze.py summary              # annualized stats + correlation

Reads the SQLite DB populated by etf_fetch.py; run that first.
"""

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import yaml

from etf_common import DEFAULT_CURRENCY_META, DEFAULT_DB


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
        """
    )

    parser.add_argument('--db', '-d', default=DEFAULT_DB, help='Database file path')
    parser.add_argument('--currency-meta', default=DEFAULT_CURRENCY_META,
                        help='Pinned ftgo resolution / currency sidecar path')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    returns_parser = subparsers.add_parser('returns', help='Show returns DataFrame')
    returns_parser.add_argument('isin', nargs='?', help='ISIN to show (all if omitted)')

    subparsers.add_parser('summary', help='Show data summary')

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
    from RiskModel import HRPRiskModel

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
