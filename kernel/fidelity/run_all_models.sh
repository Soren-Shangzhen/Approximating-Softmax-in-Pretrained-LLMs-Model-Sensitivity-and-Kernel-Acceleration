#!/usr/bin/env bash
# BF16-path fidelity of the patched kernel in the five models of Table 2 (one B200; the 70B/72B models need
# enough GPU memory for BF16 weights).  Output names follow results/kernel/fidelity/.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-fidelity_outputs}"; mkdir -p "$OUT"
declare -A M=( [qwen15]=Qwen/Qwen2.5-1.5B [qwen72]=Qwen/Qwen2.5-72B [mistral7]=mistralai/Mistral-7B-v0.3
               [llama8]=meta-llama/Meta-Llama-3.1-8B [llama70]=meta-llama/Llama-3.1-70B )
for alias in qwen15 qwen72 mistral7 llama8 llama70; do
  name=$([ "$alias" = qwen15 ] && echo "" || echo "_$alias")
  for arm in 0 1; do
    python run_ppl_arm.py --arm $arm --model "${M[$alias]}" --out "$OUT/ppl${name}_A$arm.csv"
  done
done
