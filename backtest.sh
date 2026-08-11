#!/usr/bin/env bash
# Out-of-sample method comparison on the long-history ETFs.
# Reproduces docs/portfolio-backtest-notes.md.
#
# Usage: ./backtest.sh [WINDOW_YEARS] [STEP_MONTHS] [BROKER]
#   WINDOW_YEARS  training window in years   (default 2)
#   STEP_MONTHS   rebalance interval in months (default 3)
#   BROKER        optional broker cost profile (e.g. degiro); omit for no costs
#
# Short-history funds are excluded so the common window reaches back to 2019:
#   IE0006WW1TQ4  Xtrackers MSCI World ex-USA     (from Mar 2024)
#   IE0003XJA0J9  Amundi Prime All Country World  (from Jun 2024)
#   IE000YYE6WK5  VanEck Defense                  (from Apr 2023)

set -euo pipefail

WINDOW="${1:-2}"
STEP="${2:-3}"
BROKER="${3:-}"

# NOTE: pass these as separate words (argparse nargs='+'), never as one
# unquoted variable — zsh/bash split differently and it breaks parsing.
EXCLUDE=(--exclude IE0006WW1TQ4 IE0003XJA0J9 IE000YYE6WK5)

COST_ARGS=()
COST_LABEL="no costs"
if [[ -n "$BROKER" ]]; then
  COST_ARGS=(--broker "$BROKER" --portfolio-size 10000)
  COST_LABEL="$BROKER costs, €10k"
fi

run() {  # run <method-and-flags...> ; echoes the OOS stats block
  uv run python -m hierofolio.analyze backtest "$@" \
    "${EXCLUDE[@]}" --window "$WINDOW" --step "$STEP" "${COST_ARGS[@]}" 2>&1 \
    | sed -n '/Out-of-Sample Statistics/,$p'
}

stat() {  # stat <name> <method-and-flags...>
  local name="$1"; shift
  local block; block="$(run "$@")"
  printf "%-22s Sharpe %-7s Ret %-8s Vol %-8s MaxDD %-8s\n" \
    "$name" \
    "$(awk '/Sharpe/{print $2}' <<<"$block")" \
    "$(awk '/Ann Return/{print $3}' <<<"$block")" \
    "$(awk '/Ann Vol/{print $3}' <<<"$block")" \
    "$(awk '/Max Drawdown/{print $3}' <<<"$block")"
}

echo "OOS method comparison — window ${WINDOW}y, step ${STEP}m, ${COST_LABEL}"
echo "----------------------------------------------------------------------"
stat "HRP"            --method hrp
stat "Schur-HRP g0.5" --method schur-hrp --gamma 0.5
stat "HRP-Sigma-Mu"   --method hrp-sigma-mu --tau 1
stat "MVO uncapped"   --method mvo
stat "MVO cap0.25"    --method mvo --max-weight 0.25
stat "Robust cap0.25" --method robust --max-weight 0.25 --robustness-penalty 10
stat "CRISP g0.3"     --method crisp --corr-penalty 0.3
stat "CRISP g0.5"     --method crisp --corr-penalty 0.5
stat "CRISP g1.0"     --method crisp --corr-penalty 1.0

echo "----------------------------------------------------------------------"
echo "Equal-weight benchmark (from any run's second column):"
run --method hrp | awk '/Ann Return|Ann Vol|Sharpe|Max Draw/{print "  "$0}'
