#!/usr/bin/env bash
# All B200 call-latency cells of Table 2, Table B5 and Figures 4C-D / S2A (one B200, FA4 patched; Appendix B.3).
# Each arm runs in its own process.  Default: 7 windows x 100 calls after 50 warm-up calls, heads 16, head dim
# 128, batch 1.  Three-round cells alternate the kernel order A0A1, A1A0, A0A1, except the FP8 non-causal 8K cell
# (A0 before A1 in every round).  The speed-up of a cell is 100 (median A0 / median A1 - 1) over the pooled windows
# (paper/tables/check_tables.py).  File names match results/kernel/timing/.
set -uo pipefail
cd "$(dirname "$0")/.."
R="${OUT:-timing_outputs}"; mkdir -p "$R"
B="python benchmark/bench_arm.py"
abba() {   # abba <prefix> <tag-prefix> <args...>
  local pre=$1 tag=$2; shift 2
  for rd in 1 2 3; do
    [ $((rd % 2)) -eq 1 ] && ord="0 1" || ord="1 0"
    for arm in $ord; do $B --arm $arm --out "$R/${pre}_r${rd}_A$arm.csv" --tag "${tag}_r$rd" "$@"; done
  done
}
pair() {   # pair <name> <tag> <args...>
  local name=$1 tag=$2; shift 2
  for arm in 0 1; do $B --arm $arm --out "$R/${name}_A$arm.csv" --tag "$tag" "$@"; done
}
# main matrix: FP8 E4M3 and BF16, causal, 4K/8K/16K (Table 2, Table B5a, Figure 4C)
abba A_main main
# FP8 causal sequence sweep 1K-32K (Table B5b, Figure S2A) and the 1K/2K retest with 9 windows
abba S_seqsweep seq --dtypes fp8_e4m3 --seqlens 1024,2048,4096,8192,16384,32768
pair S_short short --dtypes fp8_e4m3 --seqlens 1024,2048 --runs 9
# FP8 non-causal 8K, A0 before A1 in every round (Table 2, Table B5c)
for rd in 1 2 3; do for arm in 0 1; do
  $B --arm $arm --out "$R/C2_noncausal_r${rd}_A$arm.csv" --tag "nc_r$rd" --dtypes fp8_e4m3 --seqlens 8192 --causal 0
done; done
# FP16 causal / FP16 and BF16 non-causal 8K (Table B5a/c, Figure 4C); FP16 causal 16K single round
abba J_fp16_causal fp16c --dtypes fp16 --seqlens 8192
abba K_fp16_noncausal fp16nc --dtypes fp16 --seqlens 8192 --causal 0
abba L_bf16_noncausal bf16nc --dtypes bf16 --seqlens 8192 --causal 0
pair Y_fp16 fp16 --dtypes fp16 --seqlens 8192,16384
# single-round non-causal FP8/BF16 4K-16K (Table B5c)
pair C_noncausal noncausal --causal 0
# shape robustness (Table B5e): heads, batch 4, head dimension 64, 2K/32K extension; GQA 32q/8kv (three rounds)
for h in 4 8 32; do pair G_heads$h heads$h --dtypes fp8_e4m3 --seqlens 8192 --heads $h; done
pair H_b4 b4 --dtypes fp8_e4m3 --seqlens 8192 --batch 4
pair B_hd64 hd64 --hd 64
pair D_seqext seqext --dtypes fp8_e4m3 --seqlens 2048,32768
for rd in 1 2 3; do
  [ $((rd % 2)) -eq 1 ] && ord="0 1" || ord="1 0"
  for arm in $ord; do python benchmark/bench_gqa.py $arm "$R/I_gqa_r${rd}_A$arm.csv" --tag "gqa_r$rd"; done
done
# stock-FA4 emulation settings vs Rowmax-H15, FP8 causal 8K (Figure 4D, Table B5d)
$B --arm 0 --freq 0 --dtypes fp8_e4m3 --seqlens 8192 --out "$R/E_freq0.csv" --tag freq0
for cfg in "16 1" "8 1" "8 0" "4 1" "4 0" "2 1" "2 0"; do set -- $cfg
  $B --arm 0 --freq $1 --start-frg $2 --dtypes fp8_e4m3 --seqlens 8192 --out "$R/E_freq${1}_s${2}.csv" --tag "freq${1}_s${2}"
done
$B --arm 1 --dtypes fp8_e4m3 --seqlens 8192 --out "$R/E_h15.csv" --tag freq_h15
# external references, BF16 (Table B5g)
for be in cudnn torch_flash mem_eff; do
  python benchmark/bench_sdpa_reference.py --backend $be --out "$R/X_${be}_bf16.csv"
done
for be in cudnn torch_flash; do
  python benchmark/bench_sdpa_reference.py --backend $be --causal 0 --out "$R/X_${be}_bf16_nc.csv"
done
# energy per forward, FP8 causal 16K, 45 s loop (Table 2, Table B5f)
for arm in 0 1; do python profile/energy_loop.py $arm "$R/F_power_A$arm.csv" 45; done
ls "$R" | wc -l
