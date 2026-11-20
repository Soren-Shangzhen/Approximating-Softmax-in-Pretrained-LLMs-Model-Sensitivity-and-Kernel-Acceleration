# Reference results

The result files behind every figure and table of the paper, in the layout that the code in this repository writes.
A rerun with `--out-root runs/<name>` produces the same layout, so every analysis script can be pointed at a rerun
(`APPROX_SOFTMAX_RESULTS=<dir>` for `paper/`, `--lattice-root` etc. for `experiments/lattice_operators/analyze.py`).
Model aliases: `qwen05` Qwen2.5-0.5B, `qwen15` Qwen2.5-1.5B, `qwen3` Qwen2.5-3B, `qwen72` Qwen2.5-72B, `llama1`
Llama-3.2-1B, `llama3` Llama-3.2-3B, `llama8` Llama-3.1-8B, `llama70` Llama-3.1-70B, `gemma2` Gemma-2-2B, `mistral7`
Mistral-7B-v0.3.

| Directory | Content | Paper |
|---|---|---|
| `characterization/<model>/` | tasks T00-T13 of the ten models (`softmax_approx.characterization`) | Sec. 4; App. A, C, D |
| `characterization/crossmodel_{summary,effects}.csv`, `crossmodel_manifest.json` | one row per model / per primary contrast, and the per-model file manifest | Table 3 |
| `characterization_large_T11_T12/{qwen72,llama70}/` | long-context (T11) and zero-shot (T12) runs of the two large models, with the tasks they depend on | App. C, Fig. S3, Tables C1-C2 |
| `lattice_operators/qwen15/` | Section 5 lattice operators and Rowmax-H15 semantic simulations on Qwen2.5-1.5B | Sec. 5.1-5.2, Tables 2, B1, B2, Fig. 4A-B |
| `exp_map_ablation/qwen15/` | exponential-map ablation (softmax, Rowmax-H15, PoT m=2, S(x), S_k) | App. B.2 sub-table of Table B2 |
| `kernel/` | Rowmax-H15 in FlashAttention-4 on B200: model fidelity, call latency, energy, Nsight Systems, correctness readbacks, static instruction audit | Sec. 5.3-5.5, App. B, Tables 2, B4, B5, Figs. 4, S1, S2 |

## Per-block files (`condition_blocks/<condition>__L<context>.csv`)

One row per aligned block (97 blocks of 2048 tokens at the default context; 24 at 8K and 12 at 16K):

| column | meaning |
|---|---|
| `model`, `task`, `condition`, `block`, `seq_len` | identification (`task` is the task that first evaluated the condition) |
| `n_pred` | next-token predictions in the block (2047) |
| `baseline_nll`, `nll`, `delta_nll` | mean NLL of the softmax baseline and of the condition on this block, and their difference |
| `output_kl_forward`, `output_kl_reverse`, `output_jsd` | vocabulary-output divergences from the baseline distribution (App. C.4) |
| `top1_agreement`, `top1_matches` | fraction / count of predictions whose arg max equals the baseline's |

Every contrast of the paper is a mean over blocks of a per-block difference, with a 95% percentile paired block
bootstrap (5,000 replicates, seed 0; `softmax_approx/stats.py`).  `condition_meta/<condition>__L<context>.json` holds
the forward specification of the condition (operator and parameters), the protocol it was evaluated under (model
revision, token signature, query chunk, dtype) and the cache state; `divergence/<condition>__L<context>__W<n>.json`
the mean row-wise attention divergence from softmax on the first n blocks (used for matching, T06-T09, T11).

## Task outputs of the characterization pipeline (`characterization/<model>/T*/`)

| task | files |
|---|---|
| T00 | `engineering_checks.csv` (query-chunk memory gate, eager-implementation agreement, padding, quantizer mirror, probe identity), `baseline_summary.json` |
| T01 | `conditions.csv`, `effects.csv` (support sweep), `baseline_mean_support.json` |
| T01B | `effects_by_layer.csv` (D_W per target layer), `aggregate_effect.csv`, `block_effects_by_layer.csv`, `support_audit.csv` |
| T02 | `conditions.csv`, `effects.csv` (exponential vs linear map) |
| T03 | `conditions.csv` (precision grid) |
| T04 | `conditions.csv`, `effects.csv` (D_R(K), I_R), `error_geometry.csv`, `error_bins_delta.csv`, `mass_concentration.csv` |
| T05 | `conditions.csv`, `error_geometry.csv`, `error_bins_delta.csv`, `analytic_audit.csv` (reconstruction envelopes) |
| T06 | `K0.json` (J0 of the model), `pot_bound_audit.csv`, `pot_error_geometry.csv`, `pot_error_bins_delta.csv`, `rowmax_pot_conditions.csv`, `rowmax_tail_probe.csv` |
| T07 | `distortion_points.csv`, `fit_summary.csv` (JSD-dNLL response map) |
| T08, T09 | `conditions.csv`, `directional_effects.csv` (D_T, D_TV), `match_manifest.json`, `match_trace.csv` (bisection of alpha) |
| T10 | `probe_summary.csv` (leakage L_q, local pV), `output_probe.csv`, `mass_probe.csv`, `rescue_conditions.csv`, `rescue_effects.csv` (G_gap), `rescue_probe.csv`, `probe_summary_with_rescue.csv` |
| T11 | `conditions.csv`, `within_context_effects.csv`, `match_manifest.json`, `match_trace.csv` |
| T12 | `zero_shot_results.csv` (lm-evaluation-harness metrics per condition and task) |
| T13 | `output_fidelity_conditions.csv`, `diagnostic_correlations.csv` |

Model-level files: `protocol.json` and `execution_manifest.json` (the frozen settings of the run), `model_info.json`,
`model_summary.json`, `condition_summary.csv` and `blocks.csv` (all conditions), `layer_metrics.csv` (per-layer
probe summaries), `engineering_checks.json`, `analysis_plan.json`, `run_status.json` and `T*/status.json`
(completion markers used by `--resume`), `shared_numerical_probe/` (Qwen2.5-1.5B only: one baseline-trajectory pass for the error geometry,
reconstruction, PoT error, attention-mass concentration and Rowmax-PoT tail probes of T04-T06).

## `lattice_operators/qwen15/` and `exp_map_ablation/qwen15/`

`condition_blocks/` has the per-block files of the conditions listed in `experiments/lattice_operators/run.py`,
`protocol.json` the protocol shared with `characterization/qwen15` (same model revision, token stream and query
chunk).  `leakage/leakage_summary.csv` gives, per probed condition, the signed leakage L_0.9 statistics of the
first four blocks and the Pearson correlation `pearson_r` with the lattice phase the operator uses (`phase`:
`r_oct`, `r_half`, `r_p2`, `r_ps2` or `r_h15`; definitions in `experiments/lattice_operators/leakage.py`);
`leak_mass_by_layer_*.csv` and `leak_output_by_layer_*.csv` are the per-layer aggregates.  The per-row arrays
(`leak_rows_<condition>.npz`, 35-60 MB each) are not included; `run.py --leakage paper` regenerates them.
`analysis/` is written by `experiments/lattice_operators/analyze.py`.  The ablation comes from two runs on the same
blocks; the second (S_k, k = 2, 3) reproduced the first run's per-block softmax NLLs exactly.

## `kernel/` (B200)

| path | content |
|---|---|
| `fidelity/ppl[_<model>]_A{0,1}.csv` | per-block NLL with stock FA4 (A0) and the patched kernel (A1) on the BF16 path; no suffix = Qwen2.5-1.5B |
| `timing/<set>_A{0,1}.csv`, `timing/<set>_r{1,2,3}_A{0,1}.csv` | call-latency windows: one row per window with `dtype`, `seqlen`, `heads` (or `hq`/`hkv`), `hd`, `batch`, `causal`, `freq`, `start_frg`, `run`, `latency_ms`, `tflops`, `sm_clock`, `temp_c`, `power_w`; sets as in `kernel/benchmark/run_timing_matrix.sh` (`X_*`: SDPA/cuDNN references) |
| `timing/F_power_A{0,1}.csv` (+ `.samples.csv`) | energy loop summary (`energy_per_fwd_mJ`, `mean_power_w`, `per_fwd_ms`, ...) and the raw nvidia-smi samples (time, power W, SM clock MHz, temperature C) |
| `nsys/kernel_vs_wall.csv`, `nsys/launch_breakdown.csv` | Nsight Systems: median kernel duration over the traced calls, host call latency per timed call, launch and kernel shares (Table B4) |
| `correctness_dumps/L1_singletile_P.pt`, `L15_multitile_P.pt` | element-wise probability readbacks of the two correctness gates (Figure S1) |
| `sass_audit/` | ptxas register report, SASS disassembly and instruction counts of the sm_86 micro-kernels (App. B.5) |
