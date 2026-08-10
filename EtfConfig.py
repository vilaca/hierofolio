#!/usr/bin/env python
"""
Minimal UCITS ETF Data Extractor for Hierarchical Risk Parity Framework

Uses ISIN-driven configuration with auto-resolved fields from OpenFIGI.

Usage:
    python extract.py
    python extract.py etf_universe.yaml
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
import logging
import time
import os
import re
import yaml
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Required imports
from ftgo import get_xid, get_historical_prices
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


@dataclass
class ETFDefinition:
    """Definition of an ETF from config."""
    isin: str
    name: str
    tickers: List[str]
    exchange: str = ""
    figi: str = ""
    
    @classmethod
    def from_config(cls, isin: str, data: dict) -> "ETFDefinition":
        """Create from config dictionary."""
        return cls(
            isin=isin,
            name=data.get('name', isin),
            tickers=data.get('tickers', []),
            exchange=data.get('exchange', ''),
            figi=data.get('figi', '')
        )


class UCITSDataExtractor:
    """
    Minimal data extractor for UCITS ETFs.
    
    Database stores only: isin, date, close
    Configuration loaded from YAML with auto-resolved fields.
    """
    
    def __init__(
        self,
        config_path: str = "etf_universe.yaml",
        db_path: str = "ucits_prices.db",
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ):
        self.config_path = config_path
        self.db_path = db_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        self.force_refresh = force_refresh
        
        # Validate config exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create it using: python manage_etf.py <ISIN>"
            )
        
        # Load configuration
        self.etf_universe = self._load_config(config_path)
        
        # Rate limit tracking
        self._ftgo_throttled = False
        self._ftgo_wait_until = None
        self._yf_throttled = False
        self._yf_wait_until = None
        
        # Cache
        self._data_cache = {}
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Loaded {len(self.etf_universe)} ETFs from {config_path}")
        logger.info(f"Database: {db_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, ETFDefinition]:
        """Load ETF universe from YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config or 'etfs' not in config:
            raise ValueError(f"Invalid config: {config_path} must contain 'etfs' key")
        
        universe = {}
        for isin, data in config.get('etfs', {}).items():
            if not data.get('tickers'):
                raise ValueError(f"ETF {isin} has no tickers. Please re-add using manage_etf.py.")
            universe[isin] = ETFDefinition.from_config(isin, data)
        
        return universe
    
    def _init_database(self):
        """Create database file and schema if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    isin TEXT,
                    date TEXT,
                    close REAL,
                    PRIMARY KEY (isin, date)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_isin_date 
                ON prices (isin, date)
            """)
            
            conn.commit()
    
    def _is_cached(self, isin: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Check if data exists in cache."""
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
                if days_behind > 7:
                    return False, df
            
            return True, df
    
    def _save_prices(self, isin: str, df: pd.DataFrame):
        """Save price data to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM prices WHERE isin = ?", (isin,))
            
            df_to_save = df[['Close']].copy()
            df_to_save['isin'] = isin
            df_to_save.reset_index(inplace=True)
            df_to_save.rename(columns={'index': 'date', 'Close': 'close'}, inplace=True)
            
            df_to_save[['isin', 'date', 'close']].to_sql(
                'prices', conn, if_exists='append', index=False
            )
            
            conn.commit()
    
    def _fetch_ftgo(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch data using ftgo."""
        if self._ftgo_throttled:
            if self._ftgo_wait_until and datetime.now() < self._ftgo_wait_until:
                wait_seconds = (self._ftgo_wait_until - datetime.now()).total_seconds()
                logger.debug(f"ftgo throttled, waiting {wait_seconds:.1f}s")
                time.sleep(wait_seconds + 1)
                self._ftgo_throttled = False
            else:
                self._ftgo_throttled = False
        
        try:
            xid = get_xid(ticker)
            df = get_historical_prices(
                xid,
                self.start_date.strftime("%Y%m%d"),
                self.end_date.strftime("%Y%m%d")
            )
            
            if df is not None and not df.empty:
                df = df.rename(columns={'date': 'Date', 'close': 'Close'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                return df[['Close']]
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'rate limit' in error_str.lower():
                self._ftgo_throttled = True
                wait_match = re.search(r'wait\s+(\d+)\s*(?:second|minute)', error_str, re.IGNORECASE)
                if wait_match:
                    value = int(wait_match.group(1))
                    if 'minute' in wait_match.group(0).lower():
                        wait_seconds = value * 60
                    else:
                        wait_seconds = value
                else:
                    wait_seconds = 60
                self._ftgo_wait_until = datetime.now() + timedelta(seconds=wait_seconds)
                logger.warning(f"ftgo rate limited. Waiting {wait_seconds}s")
            else:
                logger.debug(f"ftgo failed for {ticker}: {e}")
        return None
    
    def _fetch_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance."""
        if self._yf_throttled:
            if self._yf_wait_until and datetime.now() < self._yf_wait_until:
                wait_seconds = (self._yf_wait_until - datetime.now()).total_seconds()
                logger.debug(f"yfinance throttled, waiting {wait_seconds:.1f}s")
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
            error_str = str(e)
            if '429' in error_str or 'rate limit' in error_str.lower():
                self._yf_throttled = True
                self._yf_wait_until = datetime.now() + timedelta(seconds=60)
                logger.warning(f"yfinance rate limited. Waiting 60s")
            else:
                logger.debug(f"yfinance failed for {ticker}: {e}")
        return None
    
    def fetch_etf(self, isin: str, skip_cache: bool = False) -> Optional[pd.DataFrame]:
        """Fetch data for a single ETF."""
        if isin not in self.etf_universe:
            logger.error(f"ISIN {isin} not in universe")
            return None
        
        etf = self.etf_universe[isin]
        logger.info(f"Fetching {etf.name} ({isin})...")
        
        if not skip_cache:
            cached, df = self._is_cached(isin)
            if cached and df is not None and not df.empty:
                logger.info(f"✓ {isin} loaded from cache ({len(df)} observations)")
                return df
        
        for ticker in etf.tickers:
            logger.debug(f"  Trying {ticker}...")
            
            df = self._fetch_ftgo(ticker)
            if df is not None and not df.empty:
                self._save_prices(isin, df)
                logger.info(f"✓ {isin} fetched via ftgo ({ticker})")
                return df
            
            df = self._fetch_yfinance(ticker)
            if df is not None and not df.empty:
                self._save_prices(isin, df)
                logger.info(f"✓ {isin} fetched via yfinance ({ticker})")
                return df
            
            time.sleep(0.5)
        
        logger.warning(f"✗ {isin} - All sources failed")
        return None
    
    def fetch_all(self, clean: bool = True, skip_cache: bool = False) -> pd.DataFrame:
        """Fetch data for all ETFs in the universe."""
        data_dict = {}
        
        for isin in self.etf_universe:
            df = self.fetch_etf(isin, skip_cache=skip_cache)
            if df is not None and not df.empty:
                data_dict[isin] = df['Close']
                self._data_cache[isin] = df
        
        if not data_dict:
            raise RuntimeError("No data fetched for any ETF")
        
        combined = pd.DataFrame(data_dict)
        
        if clean:
            combined = self._clean_data(combined)
        
        return combined
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and align the combined DataFrame."""
        df = df.sort_index()
        df = df.ffill()
        df = df.dropna()
        
        returns = df.pct_change()
        mask = (returns.abs() <= 0.30).all(axis=1)
        df = df[mask]
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        return df
    
    def get_from_db(self, isin: str = None) -> pd.DataFrame:
        """Load data directly from database."""
        with sqlite3.connect(self.db_path) as conn:
            if isin:
                df = pd.read_sql_query(
                    "SELECT date, close FROM prices WHERE isin = ? ORDER BY date",
                    conn,
                    params=(isin,),
                    index_col='date',
                    parse_dates=['date']
                )
                return df.rename(columns={'close': isin})
            else:
                df = pd.read_sql_query(
                    "SELECT isin, date, close FROM prices ORDER BY date, isin",
                    conn,
                    parse_dates=['date']
                )
                pivot = df.pivot(index='date', columns='isin', values='close')
                return pivot
    
    def get_returns(self, isin: str = None) -> pd.DataFrame:
        """Get returns directly from the database."""
        prices = self.get_from_db(isin)
        returns = prices.pct_change().dropna()
        return returns
    
    def summary(self) -> pd.DataFrame:
        """Return a summary of fetched data."""
        summary_data = []
        for isin, df in self._data_cache.items():
            etf = self.etf_universe.get(isin)
            summary_data.append({
                "ISIN": isin,
                "Name": etf.name if etf else isin,
                "Start Date": df.index.min().strftime("%Y-%m-%d") if not df.empty else 'N/A',
                "End Date": df.index.max().strftime("%Y-%m-%d") if not df.empty else 'N/A',
                "Observations": len(df)
            })
        return pd.DataFrame(summary_data)


def main():
    """Main execution function."""
    print("=" * 70)
    print("Minimal UCITS ETF Data Extractor")
    print("=" * 70)
    
    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "etf_universe.yaml"
    
    try:
        extractor = UCITSDataExtractor(
            config_path=config_path,
            db_path="ucits_prices.db",
            start_date="2018-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            force_refresh=False
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease create a configuration file using:")
        print("  python manage_etf.py IE00BM67HK77")
        return 1
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    print(f"\nLoaded {len(extractor.etf_universe)} ETFs from {config_path}")
    print(f"Database: {extractor.db_path}\n")
    
    try:
        prices = extractor.fetch_all(clean=True)
        print(f"\n✓ Successfully fetched data for {len(prices.columns)} ETFs")
        print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
        print(f"  Observations: {len(prices)}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    summary_df = extractor.summary()
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Sample Data (Last 5 days)")
    print("=" * 70)
    print(prices.tail())
    
    returns = prices.pct_change().dropna()
    
    print("\n" + "=" * 70)
    print("Returns Statistics (Annualized)")
    print("=" * 70)
    ann_returns = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    stats = pd.DataFrame({
        'Ann Return': ann_returns,
        'Ann Vol': ann_vol,
        'Sharpe': ann_returns / ann_vol
    })
    print(stats.round(4))
    
    print("\n" + "=" * 70)
    print("Correlation Matrix")
    print("=" * 70)
    print(returns.corr().round(3))
    
    print("\n" + "=" * 70)
    print("Ready for HRPRiskModel")
    print("=" * 70)
    print("""
    from your_risk_model import HRPRiskModel
    
    # Load returns directly from database
    extractor = UCITSDataExtractor("etf_universe.yaml")
    returns = extractor.get_returns()
    
    risk_model = HRPRiskModel(
        returns=returns,
        shrinkage_method="constant_correlation",
        shrinkage_intensity=0.3,
        cluster_mode="full",
        linkage_method="ward"
    )
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
