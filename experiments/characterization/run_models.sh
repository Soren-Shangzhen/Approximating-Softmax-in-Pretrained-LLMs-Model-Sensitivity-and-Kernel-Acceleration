#!/usr/bin/env bash
# Section 4 / Appendices A, C, D: characterization of the eight single-GPU models (one 24 GB GPU each).
# Task profiles come from softmax_approx/characterization.py (MODEL_REGISTRY / PROFILE_TASKS):
#   qwen15 deep_reference (T00-T13); qwen05, qwen3, llama1, llama3, gemma2 core; llama8, mistral7 modern_core.
# Outputs: $OUT/<model>/... (same layout as results/characterization).  --resume continues an interrupted run.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-runs/characterization}"
for m in qwen15 qwen05 qwen3 llama1 llama3 gemma2 llama8 mistral7; do
  python -m softmax_approx.characterization --model "$m" --out-root "$OUT" --resume
done
