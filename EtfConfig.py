#!/usr/bin/env python
"""
Hierofolio — Hierarchical Risk Parity for UCITS ETFs

Usage:
    hierofolio add IE00BM67HK77
    hierofolio add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66
    hierofolio list
    hierofolio fetch
    hierofolio fetch IE00BM67HK77
    hierofolio update
    hierofolio returns
    hierofolio summary

Examples:
    # Add an ETF by ISIN (auto-resolves name, tickers, exchange)
    hierofolio add IE00BM67HK77

    # Add multiple ETFs
    hierofolio add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66

    # List all ETFs in configuration
    hierofolio list

    # Fetch data for all ETFs
    hierofolio fetch

    # Fetch data for a specific ETF
    hierofolio fetch IE00BM67HK77

    # Update config from OpenFIGI (refresh metadata)
    hierofolio update IE00BM67HK77

    # Show returns DataFrame
    hierofolio returns

    # Show summary statistics
    hierofolio summary
"""

import argparse
import sys
import os
import yaml
import re
import requests
import sqlite3
import warnings
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from ftgo import get_xid, get_historical_prices
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = "etf_universe.yaml"
DEFAULT_DB = "hierofolio.db"
DEFAULT_START_DATE = "2018-01-01"


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
        return cls(
            isin=isin,
            name=data.get('name', isin),
            tickers=data.get('tickers', []),
            exchange=data.get('exchange', ''),
            figi=data.get('figi', '')
        )


# ============================================================================
# OpenFIGI Resolver
# ============================================================================

class OpenFIGIResolver:
    """Resolve ISIN to ETF metadata using OpenFIGI API."""
    
    BASE_URL = "https://api.openfigi.com/v3/mapping"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['X-OPENFIGI-APIKEY'] = api_key
        self.session = requests.Session()
    
    def resolve(self, isin: str) -> Optional[dict]:
        """Resolve ISIN to ETF metadata."""
        if not re.match(r'^[A-Z]{2}[A-Z0-9]{10}$', isin):
            print(f"✗ Invalid ISIN: {isin}")
            return None
        
        payload = [{"idType": "ID_ISIN", "idValue": isin}]
        
        try:
            response = self.session.post(
                self.BASE_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data or not data[0].get('data'):
                print(f"✗ No data found for ISIN: {isin}")
                return None
            
            result = data[0]['data'][0]
            
            return {
                'name': result.get('name', f"ETF {isin}"),
                'tickers': [result.get('ticker')] if result.get('ticker') else [],
                'exchange': result.get('exchCode', ''),
                'figi': result.get('figi', ''),
                'resolved_at': datetime.now().isoformat(),
                'source': 'OpenFIGI'
            }
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API error for {isin}: {e}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            print(f"✗ Error parsing response for {isin}: {e}")
            return None


# ============================================================================
# Config Manager
# ============================================================================

class ConfigManager:
    """Manage ETF configuration in YAML file."""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG):
        self.config_path = config_path
        self.resolver = OpenFIGIResolver()
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {'etfs': {}}
        return {'etfs': {}}
    
    def _save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def add(self, isin: str) -> bool:
        """Add ETF by ISIN (auto-resolves all fields)."""
        if isin in self.config.get('etfs', {}):
            print(f"⚠ ISIN {isin} already exists")
            return False
        
        print(f"🔍 Resolving {isin}...")
        info = self.resolver.resolve(isin)
        
        if not info:
            return False
        
        if 'etfs' not in self.config:
            self.config['etfs'] = {}
        
        self.config['etfs'][isin] = info
        self._save_config()
        
        print(f"✓ Added {isin}")
        print(f"  Name: {info['name']}")
        print(f"  Tickers: {', '.join(info['tickers'])}")
        print(f"  Exchange: {info['exchange']}")
        print(f"  FIGI: {info['figi']}")
        return True
    
    def list(self) -> List[Tuple[str, dict]]:
        """List all ETFs in config."""
        return sorted(self.config.get('etfs', {}).items())
    
    def get(self, isin: str) -> Optional[dict]:
        """Get ETF config by ISIN."""
        return self.config.get('etfs', {}).get(isin)
    
    def update(self, isin: str) -> bool:
        """Update ETF metadata from OpenFIGI."""
        if isin not in self.config.get('etfs', {}):
            print(f"✗ ISIN {isin} not found")
            return False
        
        print(f"🔍 Updating {isin}...")
        info = self.resolver.resolve(isin)
        
        if not info:
            return False
        
        self.config['etfs'][isin] = info
        self._save_config()
        
        print(f"✓ Updated {isin}")
        print(f"  Name: {info['name']}")
        return True


# ============================================================================
# Data Extractor
# ============================================================================

class DataExtractor:
    """Extract historical price data for ETFs."""
    
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
                if days_behind > 7:
                    return False, df
            
            return True, df
    
    def _save_prices(self, isin: str, df: pd.DataFrame):
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
                    wait_seconds = value * 60 if 'minute' in wait_match.group(0).lower() else value
                else:
                    wait_seconds = 60
                self._ftgo_wait_until = datetime.now() + timedelta(seconds=wait_seconds)
                logger.warning(f"ftgo rate limited. Waiting {wait_seconds}s")
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
        """Fetch data for specific ISIN or all ETFs."""
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
            
            logger.info(f"Fetching {etf.name} ({isin})...")
            
            cached, df = self._is_cached(isin)
            if cached and df is not None and not df.empty:
                logger.info(f"✓ {isin} loaded from cache ({len(df)} observations)")
                data_dict[isin] = df['Close']
                continue
            
            for ticker in etf.tickers:
                df = self._fetch_ftgo(ticker)
                if df is not None and not df.empty:
                    self._save_prices(isin, df)
                    logger.info(f"✓ {isin} fetched via ftgo ({ticker})")
                    data_dict[isin] = df['Close']
                    break
                
                df = self._fetch_yfinance(ticker)
                if df is not None and not df.empty:
                    self._save_prices(isin, df)
                    logger.info(f"✓ {isin} fetched via yfinance ({ticker})")
                    data_dict[isin] = df['Close']
                    break
                
                time.sleep(0.5)
            
            if isin not in data_dict:
                logger.warning(f"✗ {isin} - All sources failed")
        
        if not data_dict:
            raise RuntimeError("No data fetched")
        
        combined = pd.DataFrame(data_dict)
        combined = combined.sort_index().ffill().dropna()
        
        if combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)
        
        self._data_cache = data_dict
        return combined
    
    def get_prices(self, isin: Optional[str] = None) -> pd.DataFrame:
        """Load prices from database."""
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
                return df.pivot(index='date', columns='isin', values='close')
    
    def get_returns(self, isin: Optional[str] = None) -> pd.DataFrame:
        """Get returns from database."""
        prices = self.get_prices(isin)
        returns = prices.pct_change().dropna()
        return returns
    
    def summary(self) -> pd.DataFrame:
        """Get summary of fetched data."""
        data = []
        for isin, df in self._data_cache.items():
            etf = self.etf_universe.get(isin)
            data.append({
                "ISIN": isin,
                "Name": etf.name if etf else isin,
                "Start": df.index.min().strftime("%Y-%m-%d") if not df.empty else 'N/A',
                "End": df.index.max().strftime("%Y-%m-%d") if not df.empty else 'N/A',
                "Obs": len(df)
            })
        return pd.DataFrame(data)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="hierofolio",
        description="Hierarchical Risk Parity for UCITS ETFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add an ETF by ISIN
  hierofolio add IE00BM67HK77

  # Add multiple ETFs
  hierofolio add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66

  # List all ETFs
  hierofolio list

  # Fetch data for all ETFs
  hierofolio fetch

  # Fetch data for specific ETF
  hierofolio fetch IE00BM67HK77

  # Update ETF metadata
  hierofolio update IE00BM67HK77

  # Show returns
  hierofolio returns

  # Show summary
  hierofolio summary
        """
    )
    
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='Config file path')
    parser.add_argument('--db', '-d', default=DEFAULT_DB, help='Database file path')
    parser.add_argument('--start', '-s', default=DEFAULT_START_DATE, help='Start date')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Add
    add_parser = subparsers.add_parser('add', help='Add ETF by ISIN')
    add_parser.add_argument('isins', nargs='+', help='ISINs to add')
    
    # List
    list_parser = subparsers.add_parser('list', help='List all ETFs')
    
    # Fetch
    fetch_parser = subparsers.add_parser('fetch', help='Fetch price data')
    fetch_parser.add_argument('isin', nargs='?', help='ISIN to fetch (all if omitted)')
    fetch_parser.add_argument('--force', '-f', action='store_true', help='Force refresh')
    
    # Update
    update_parser = subparsers.add_parser('update', help='Update ETF metadata')
    update_parser.add_argument('isin', help='ISIN to update')
    
    # Returns
    returns_parser = subparsers.add_parser('returns', help='Show returns DataFrame')
    returns_parser.add_argument('isin', nargs='?', help='ISIN to show (all if omitted)')
    
    # Summary
    summary_parser = subparsers.add_parser('summary', help='Show data summary')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # ========================================================================
    # List
    # ========================================================================
    if args.command == 'list':
        config = ConfigManager(args.config)
        etfs = config.list()
        
        if not etfs:
            print("No ETFs in configuration")
            print(f"Add one: hierofolio add IE00BM67HK77")
            return 0
        
        print(f"\n{'ISIN':<14} {'Name':<50} {'Ticker':<12} {'Exchange'}")
        print("-" * 90)
        for isin, data in etfs:
            name = data.get('name', 'Unknown')[:48]
            ticker = data.get('tickers', [''])[0] if data.get('tickers') else ''
            exchange = data.get('exchange', '')
            print(f"{isin:<14} {name:<50} {ticker:<12} {exchange}")
        print(f"\nTotal: {len(etfs)} ETFs")
        print(f"Config: {args.config}")
        return 0
    
    # ========================================================================
    # Add
    # ========================================================================
    if args.command == 'add':
        config = ConfigManager(args.config)
        success = 0
        for isin in args.isins:
            if config.add(isin):
                success += 1
            print()
        print(f"✓ Added {success}/{len(args.isins)} ETFs")
        return 0 if success == len(args.isins) else 1
    
    # ========================================================================
    # Update
    # ========================================================================
    if args.command == 'update':
        config = ConfigManager(args.config)
        success = config.update(args.isin)
        return 0 if success else 1
    
    # ========================================================================
    # Fetch
    # ========================================================================
    if args.command == 'fetch':
        try:
            extractor = DataExtractor(
                config_path=args.config,
                db_path=args.db,
                start_date=args.start,
                force_refresh=args.force
            )
            prices = extractor.fetch(args.isin)
            print(f"✓ Fetched {len(prices.columns)} ETFs")
            print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
            print(f"  Observations: {len(prices)}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0
    
    # ========================================================================
    # Returns
    # ========================================================================
    if args.command == 'returns':
        try:
            extractor = DataExtractor(config_path=args.config, db_path=args.db)
            returns = extractor.get_returns(args.isin)
            print(f"Returns for {len(returns.columns)} ETFs")
            print(f"Date range: {returns.index.min()} to {returns.index.max()}")
            print("\nLast 10 returns:")
            print(returns.tail(10).round(6))
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
        return 0
    
    # ========================================================================
    # Summary
    # ========================================================================
    if args.command == 'summary':
        try:
            extractor = DataExtractor(config_path=args.config, db_path=args.db)
            returns = extractor.get_returns()
            if returns.empty:
                print("No returns data. Run: hierofolio fetch")
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
    from hierofolio import DataExtractor, HRPRiskModel
    
    extractor = DataExtractor()
    returns = extractor.get_returns()
    
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
