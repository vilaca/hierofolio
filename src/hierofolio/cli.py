"""hierofolio command-line entry point.

Dispatches the top-level ``config`` / ``fetch`` / ``analyze`` commands to the
respective module's ``main``. Each of those keeps its own argparse definitions;
this layer only routes the first token and forwards the rest (so that, e.g.,
``hierofolio analyze --help`` reaches the analyze parser).

    hierofolio config add IE00BM67HK77
    hierofolio fetch
    hierofolio analyze allocate --method hrp
"""

import argparse
import sys

from hierofolio import analyze, config, fetch

COMMANDS = {
    "config": config.main,
    "fetch": fetch.main,
    "analyze": analyze.main,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hierofolio",
        description="ETF universe config, price fetching, and portfolio analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  config    Build/maintain the ETF universe YAML from ISINs
  fetch     Populate the SQLite price DB for the universe
  analyze   Returns, summary stats, allocation, and backtests

Run 'hierofolio <command> --help' for command-specific options.
        """,
    )
    parser.add_argument("command", choices=list(COMMANDS), help="Command to run")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    # Route only the first token; everything after it belongs to the
    # subcommand's own parser (including its --help).
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0 if argv else 1

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        parser.error(f"invalid choice: {command!r} (choose from {', '.join(COMMANDS)})")

    return COMMANDS[command](rest)


if __name__ == "__main__":
    sys.exit(main())
