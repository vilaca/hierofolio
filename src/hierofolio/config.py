#!/usr/bin/env python
"""hierofolio config builder — create/maintain the ETF universe YAML from ISINs.

Usage:
    hierofolio config add IE00BM67HK77
    hierofolio config add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66
    hierofolio config list
    hierofolio config update IE00BM67HK77

Each ISIN is resolved via OpenFIGI (name, tickers, exchange, FIGI) and written
to the YAML config consumed by the fetch command.
"""

import argparse
import sys

from hierofolio.common import ConfigManager, DEFAULT_CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="hierofolio config",
        description="Build the ETF universe YAML from ISINs (via OpenFIGI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add an ETF by ISIN (auto-resolves name, tickers, exchange)
  hierofolio config add IE00BM67HK77

  # Add multiple ETFs
  hierofolio config add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66

  # List all ETFs in the config
  hierofolio config list

  # Refresh an ETF's metadata from OpenFIGI
  hierofolio config update IE00BM67HK77
        """
    )

    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG, help='Config file path')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    add_parser = subparsers.add_parser('add', help='Add ETF(s) by ISIN')
    add_parser.add_argument('isins', nargs='+', help='ISINs to add')

    subparsers.add_parser('list', help='List all ETFs')

    update_parser = subparsers.add_parser('update', help='Update ETF metadata')
    update_parser.add_argument('isin', help='ISIN to update')

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'list':
        config = ConfigManager(args.config)
        etfs = config.list()

        if not etfs:
            print("No ETFs in configuration")
            print("Add one: hierofolio config add IE00BM67HK77")
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

    if args.command == 'add':
        config = ConfigManager(args.config)
        success = 0
        for isin in args.isins:
            if config.add(isin):
                success += 1
            print()
        print(f"✓ Added {success}/{len(args.isins)} ETFs")
        return 0 if success == len(args.isins) else 1

    if args.command == 'update':
        config = ConfigManager(args.config)
        success = config.update(args.isin)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
