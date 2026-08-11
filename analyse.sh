#!/usr/bin/env bash
# Portfolio analysis — excludes short-history funds (< 2-3 years) so the
# common return window goes back to 2019 rather than 2024.
#
# Short-history funds excluded:
#   IE0006WW1TQ4  X MSCI WORLD EX USA 1C      (from Mar 2024)
#   IE0003XJA0J9  AMUNDI PRME ALL CTRY WLD ACC (from Jun 2024)
#   IE000YYE6WK5  VANECK DEFENSE ETF           (from Apr 2023)

EXCLUDE="--exclude IE0006WW1TQ4 IE0003XJA0J9 IE000YYE6WK5"

echo "=== Summary (correlations + annualised stats) ==="
uv run python -m hierofolio.analyze summary $EXCLUDE

echo ""
echo "=== HRP (risk-balanced) ==="
uv run python -m hierofolio.analyze allocate --method hrp $EXCLUDE

echo ""
echo "=== CRISP γ=0.5 (signal-following, anti-redundancy) ==="
uv run python -m hierofolio.analyze allocate --method crisp --corr-penalty 0.5 $EXCLUDE
