#!/usr/bin/env bash
# Nsight Systems CUDA trace of both arms, FP8, head dim 128, 16 heads, batch 1 (Table B4, Figure S2B).
# One process per kernel and length; the reports give the kernel duration (cuda_gpu_kern_sum, median over all
# traced calls) and the cudaLaunchKernelExC cost (cuda_kern_exec_sum / cuda_api_sum); trace_target.py writes the
# host time per timed call.
# Counter-based profiling (Nsight Compute, run_ncu.sh) was not available in the paper's sessions.
set -uo pipefail
cd "$(dirname "$0")"
R="${OUT:-nsys_outputs}"; mkdir -p "$R"
NSYS="$(command -v nsys || echo /usr/local/cuda/bin/nsys)"
for cs in "1 1024" "1 2048" "1 4096" "1 8192" "1 16384" "1 32768" "0 8192"; do set -- $cs
  for arm in 0 1; do
    name="fp8_c$1_s$2_A$arm"
    "$NSYS" profile --trace=cuda --force-overwrite true -o "$R/$name" \
      python trace_target.py --arm $arm --seqlen $2 --causal $1 --out "$R/$name.host.json"
    "$NSYS" stats --force-export=true --format csv --report cuda_gpu_kern_sum --report cuda_kern_exec_sum --report cuda_api_sum \
      --output "$R/$name" "$R/$name.nsys-rep"
  done
done
