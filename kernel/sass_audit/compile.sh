#!/usr/bin/env bash
# Static instruction audit of Appendix B.5 (any CUDA toolkit with sm_86 support; no GPU needed to compile).
# Registers per thread from ptxas; per-element instructions, pipes and dependency chain read from the SASS.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-sass_outputs}"; mkdir -p "$OUT"
nvcc -arch=sm_86 -O3 -Xptxas -v -cubin -o "$OUT/exp_emu_regcount.cubin" exp_emu_regcount.cu 2> "$OUT/ptxas_verbose.txt"
nvdisasm -c "$OUT/exp_emu_regcount.cubin" > "$OUT/full_sass.txt"
cat "$OUT/ptxas_verbose.txt"
