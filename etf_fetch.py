#!/usr/bin/env python
"""hierofolio data fetcher — populate the SQLite price DB for the ETF universe.

Usage:
    ./etf_fetch.py                 # fetch all ETFs in the config
    ./etf_fetch.py IE00BM67HK77    # fetch a single ISIN
    ./etf_fetch.py --force         # ignore the cache and re-download

Prices are sourced from ftgo (FT Markets) with a yfinance fallback, and stored
in the SQLite DB read by etf_analyze.py.
"""

import argparse
import logging
import re
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from ftgo import get_xid, get_historical_prices

from etf_common import (
    ConfigManager,
    ETFDefinition,
    DEFAULT_CONFIG,
    DEFAULT_DB,
    DEFAULT_START_DATE,
)

logger = logging.getLogger(__name__)


class DataExtractor:
    """Fetch historical ETF prices and persist them to SQLite."""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        db_path: str = DEFAULT_DB,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ):
        self.config_path = config_path
        self.db_path = db_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        self.force_refresh = force_refresh

        # Load config
        self.config_manager = ConfigManager(config_path)
        self.etf_universe = self._load_universe()

        # Rate limiting
        self._ftgo_throttled = False
        self._ftgo_wait_until = None
        self._yf_throttled = False
        self._yf_wait_until = None

        # Cache
        self._data_cache = {}

        self._init_database()

    @staticmethod
    def _summary(isin: str, name: str, source: str, df: pd.DataFrame,
                 ticker: str = None) -> str:
        """One-line result: ISIN, name, ticker, count, source and date range."""
        span = f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
        tag = f" ({ticker})" if ticker else ""
        return f"{isin} {name}{tag} — {len(df)} days - {source} - {span}"

    def _load_universe(self) -> Dict[str, ETFDefinition]:
        """Load ETF universe from config."""
        config = self.config_manager.config
        universe = {}
        for isin, data in config.get('etfs', {}).items():
            if not data.get('tickers'):
                continue
            universe[isin] = ETFDefinition.from_config(isin, data)
        return universe

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    isin TEXT,
                    date TEXT,
                    close REAL,
                    PRIMARY KEY (isin, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_isin_date
                ON prices (isin, date)
            """)
            conn.commit()

    def _is_cached(self, isin: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        if self.force_refresh:
            return False, None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE isin = ?", (isin,))
            count = cursor.fetchone()[0]

            if count == 0:
                return False, None

            df = pd.read_sql_query(
                "SELECT date, close FROM prices WHERE isin = ? ORDER BY date",
                conn,
                params=(isin,),
                index_col='date',
                parse_dates=['date']
            )

            if not df.empty:
                latest = df.index.max()
                days_behind = (self.end_date - latest).days
                # Refetch whenever we're not current; the DO NOTHING upsert
                # cheaply adds only the missing days.
                if days_behind > 0:
                    return False, df

            return True, df

    def _save_prices(self, isin: str, df: pd.DataFrame):
        prices = df[['Close']].copy()
        prices.columns = ['close']            # flattens yfinance's MultiIndex too
        prices.index = pd.to_datetime(prices.index).strftime('%Y-%m-%d %H:%M:%S')
        rows = [(isin, date, float(close)) for date, close in prices['close'].items()]

        # By default keep already-stored closes and only add new dates; --force
        # overwrites existing rows with the freshly fetched values.
        on_conflict = (
            "DO UPDATE SET close = excluded.close" if self.force_refresh else "DO NOTHING"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO prices (isin, date, close) VALUES (?, ?, ?) "
                f"ON CONFLICT(isin, date) {on_conflict}",
                rows,
            )
            conn.commit()

    def _fetch_ftgo(self, ticker: str) -> Optional[pd.DataFrame]:
        if self._ftgo_throttled:
            if self._ftgo_wait_until and datetime.now() < self._ftgo_wait_until:
                wait_seconds = (self._ftgo_wait_until - datetime.now()).total_seconds()
                time.sleep(wait_seconds + 1)
                self._ftgo_throttled = False
            else:
                self._ftgo_throttled = False

        try:
            xid = get_xid(ticker)
            df = get_historical_prices(
                xid,
                self.start_date.strftime("%d%m%Y"),
                self.end_date.strftime("%d%m%Y")
            )

            if df is not None and not df.empty:
                df = df.rename(columns={'date': 'Date', 'close': 'Close'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                return df[['Close']]
        except ValueError as e:
            # get_xid raises this for a ticker that isn't on FT Markets; fall
            # back to yfinance rather than aborting. Other ValueErrors propagate.
            if "No data found" not in str(e):
                raise
            logger.info(f"ftgo has no data for {ticker}, falling back")
        except requests.RequestException as e:
            error_str = str(e)
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 or '429' in error_str or 'rate limit' in error_str.lower():
                self._ftgo_throttled = True
                wait_match = re.search(r'wait\s+(\d+)\s*(?:second|minute)', error_str, re.IGNORECASE)
                if wait_match:
                    value = int(wait_match.group(1))
                    wait_seconds = value * 60 if 'minute' in wait_match.group(0).lower() else value
                else:
                    wait_seconds = 60
                self._ftgo_wait_until = datetime.now() + timedelta(seconds=wait_seconds)
                logger.warning(f"ftgo rate limited. Waiting {wait_seconds}s")
            else:
                logger.warning(f"ftgo request failed for {ticker}: {e}")
        return None

    def _fetch_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        if self._yf_throttled:
            if self._yf_wait_until and datetime.now() < self._yf_wait_until:
                wait_seconds = (self._yf_wait_until - datetime.now()).total_seconds()
                time.sleep(wait_seconds + 1)
                self._yf_throttled = False
            else:
                self._yf_throttled = False

        try:
            tickers_to_try = [ticker]
            if not any(ticker.endswith(suffix) for suffix in ['.L', '.DE']):
                for suffix in ['.L', '.DE']:
                    tickers_to_try.append(ticker + suffix)

            for t in tickers_to_try:
                df = yf.download(t, start=self.start_date, end=self.end_date, progress=False)
                if df is not None and not df.empty:
                    return df[['Close']]
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                self._yf_throttled = True
                self._yf_wait_until = datetime.now() + timedelta(seconds=60)
                logger.warning("yfinance rate limited. Waiting 60s")
        return None

    def fetch(self, isin: Optional[str] = None) -> pd.DataFrame:
        """Fetch data for specific ISIN or all ETFs and persist to the DB."""
        if isin:
            etfs = {isin: self.etf_universe.get(isin)}
            if not etfs[isin]:
                raise ValueError(f"ISIN {isin} not in config")
        else:
            etfs = self.etf_universe

        data_dict = {}
        for isin, etf in etfs.items():
            if not etf:
                continue

            cached, df = self._is_cached(isin)
            if cached and df is not None and not df.empty:
                logger.info(self._summary(isin, etf.name, "cache", df))
                data_dict[isin] = df['close']
                continue

            for ticker in etf.tickers:
                df = self._fetch_ftgo(ticker)
                if df is not None and not df.empty:
                    self._save_prices(isin, df)
                    logger.info(self._summary(isin, etf.name, "ftgo", df, ticker))
                    data_dict[isin] = df['Close']
                    break

                df = self._fetch_yfinance(ticker)
                if df is not None and not df.empty:
                    self._save_prices(isin, df)
                    logger.info(self._summary(isin, etf.name, "yfinance", df, ticker))
                    data_dict[isin] = df['Close']
                    break

                time.sleep(0.5)

            if isin not in data_dict:
                logger.warning(f"✗ {isin} {etf.name} — all sources failed")

        if not data_dict:
            raise RuntimeError("No data fetched")

        combined = pd.DataFrame(data_dict)
        combined = combined.sort_index().ffill().dropna()

        if combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)

        self._data_cache = data_dict
        return combined


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')
    # ftgo logs its own progress using the DDMMYYYY strings it requires;
    # quiet it and emit our own yyyy-mm-dd lines instead.
    logging.getLogger("ftgo").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="etf_fetch",
        description="Populate the SQLite price DB for the ETF universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch price data for all ETFs in the config
  ./etf_fetch.py

  # Fetch a single ETF
  ./etf_fetch.py IE00BM67HK77

  # Force a re-download, ignoring the cache
  ./etf_fetch.py --force
        """
    )

    parser.add_argument('isin', nargs='?', help='ISIN to fetch (all if omitted)')
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='Config file path')
    parser.add_argument('--db', '-d', default=DEFAULT_DB, help='Database file path')
    parser.add_argument('--start', '-s', default=DEFAULT_START_DATE, help='Start date')
    parser.add_argument('--force', '-f', action='store_true', help='Force refresh')

    args = parser.parse_args()

    try:
        extractor = DataExtractor(
            config_path=args.config,
            db_path=args.db,
            start_date=args.start,
            force_refresh=args.force
        )
        prices = extractor.fetch(args.isin)
        logger.info(
            f"Fetched {len(prices.columns)} ETFs, {len(prices)} observations "
            f"({prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d})"
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
