#!/usr/bin/env bash
# Portfolio analysis — all 93 ETFs (validated clean: >= 3yr history, no sparse
# data, no cash-like instruments). Run `hierofolio config validate` to recheck.

echo "=== Summary (correlations + annualised stats) ==="
uv run python -m hierofolio.analyze summary

echo ""
echo "=== HRP (risk-balanced) ==="
uv run python -m hierofolio.analyze allocate --method hrp

echo ""
echo "=== CRISP γ=0.5 (signal-following, anti-redundancy) ==="
uv run python -m hierofolio.analyze allocate --method crisp --corr-penalty 0.5
