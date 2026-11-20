#!/usr/bin/env bash
# Llama-3.1-70B and Qwen2.5-72B (paper: three 80 GB GPUs per job; the model is sharded over all visible GPUs).
#   1) the shared characterization tasks (modern_core profile + allocation_extended), into the same root as the
#      other models;
#   2) the long-context (T11, 2K/8K/16K) and zero-shot (T12) extension, into its own root (App. C).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-runs/characterization}"
EXT="${EXT:-runs/characterization_large_T11_T12}"
export APPROX_SOFTMAX_GPU_RESERVE_GB="${APPROX_SOFTMAX_GPU_RESERVE_GB:-15}"
export APPROX_SOFTMAX_REQUIRE_ALL_VISIBLE_GPUS=1
for m in qwen72 llama70; do
  python experiments/characterization/large_model_entry.py --model "$m" --out-root "$OUT" --resume
  python experiments/characterization/large_model_entry.py --model "$m" --tasks T11,T12 \
    --contexts 2048,8192,16384 --out-root "$EXT" \
    --zero-shot-tasks lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande \
    --zero-shot-bs 8 --zero-shot-limit 0 --checkpoint-every 1 --resume
done
