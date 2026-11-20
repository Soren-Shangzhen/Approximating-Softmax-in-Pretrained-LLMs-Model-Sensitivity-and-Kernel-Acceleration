#!/usr/bin/env bash
# Nsight Compute counters of A0 vs A1, FP8 causal 8K (optional; ERR_NVGPUCTRPERM in all of the paper's sessions,
# so no counter-level result is reported).  Needs counter permission (sudo or the driver setting).
# Some metric names differ on Blackwell; --list-metrics probes availability first and unknowns are skipped.
set -uo pipefail
cd "$(dirname "$0")"
NCU="$(command -v ncu || echo /usr/local/cuda/bin/ncu)"
METRICS=(
  sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
  sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
  sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  launch__registers_per_thread
  sm__maximum_warps_per_active_cycle_pct
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
)
if [ "${1:-}" = "--list-metrics" ]; then
  for m in "${METRICS[@]}"; do
    if sudo "$NCU" --query-metrics 2>/dev/null | grep -q "${m%%.*}"; then echo "OK   $m"; else echo "MISS $m"; fi
  done
  exit 0
fi
ARM="${1:-A1}"; ENV=0; [ "$ARM" = "A1" ] && ENV=1
JOINED=$(IFS=, ; echo "${METRICS[*]}")
FA_USE_POT_H15=$ENV sudo -E "$NCU" --metrics "$JOINED" --target-processes all \
  --csv --page raw -o "ncu_${ARM}" \
  python trace_target.py --arm "$ENV" --warmup 1 --iters 3 2>&1 | tee "ncu_${ARM}.txt" || \
  echo "WARNING: ncu run returned non-zero; check ncu_${ARM}.txt for skipped metrics."
