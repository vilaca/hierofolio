#!/usr/bin/env python
"""hierofolio analysis — returns and summary stats from the price DB.

Usage:
    ./etf_analyze.py returns              # returns for all ETFs
    ./etf_analyze.py returns IE00BM67HK77 # returns for a single ISIN
    ./etf_analyze.py summary              # annualized stats + correlation

Reads the SQLite DB populated by etf_fetch.py; run that first.
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

from etf_common import DEFAULT_DB


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
