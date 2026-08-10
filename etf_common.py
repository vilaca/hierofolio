"""Shared primitives for the hierofolio ETF tooling.

Holds the pieces used by more than one script: default paths, the ETF
definition dataclass, the OpenFIGI resolver, and the YAML config manager.
The config-building script (``etf_config.py``) writes the universe; the
data-fetching script (``etf_fetch.py``) reads it back.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import yaml

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
