#!/usr/bin/env bash
# FlashAttention-4 (CuTe DSL) at the pinned commit, for the B200 experiments.  Run after env/verify_gpu.py.
# Recorded software for the paper's B200 runs (Table B3): NVIDIA driver 580.126.09, torch 2.14.0+cu130, CUDA 13.0,
# nvidia-cutlass-dsl 4.7.1 (one session: torch 2.11.0+cu128, nvidia-cutlass-dsl 4.8.0).
set -euo pipefail
FA4_COMMIT="$(cat "$(dirname "$0")/../patch/fa4_commit.txt")"
: "${WORK:=$PWD/fa4_work}"; mkdir -p "$WORK"; cd "$WORK"
[ -d flash-attention ] || git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git fetch --all
git checkout "${FA4_COMMIT}"
echo "FA4 commit: $(git rev-parse HEAD)"
python -m pip install "nvidia-cutlass-dsl>=4.6.2" einops "apache-tvm-ffi>=0.1.12,<0.2"
# the CuTe DSL kernels need no compiled FlashAttention-2 extension (see patch/PATCH_NOTES.md, installation note)
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE python -m pip install -e .
echo "FA4_ROOT=${WORK}/flash-attention"
echo "next: python kernel/patch/apply_patch.py --fa4-root ${WORK}/flash-attention --dry-run"
