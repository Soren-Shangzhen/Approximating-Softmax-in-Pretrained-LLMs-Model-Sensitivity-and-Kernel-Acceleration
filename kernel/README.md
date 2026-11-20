# Rowmax-H15 in FlashAttention-4 on B200 (Section 5.3-5.5, Appendix B)

Arm **A0** is stock FlashAttention-4 (FA4, CuTe DSL, sm_100) at commit `0dc2cb48e484894c48a6cc5c6503fa5cded350c3`
(`patch/fa4_commit.txt`); arm **A1** is the same kernel with every exponential replaced by Rowmax-H15
(`patch/`, selected by `FA_USE_POT_H15=1`). The arm is fixed when the kernel is built and the CuTe DSL caches the
compiled kernel per process, so **every script runs one arm per process** and every A0/A1 comparison is made on one
GPU with the same environment, shape and dtype.

Recorded software of the paper's B200 measurements (Table B3): NVIDIA driver 580.126.09, torch 2.14.0+cu130,
CUDA 13.0, nvidia-cutlass-dsl 4.7.1 (the measured BF16 fidelity of the four added models: torch 2.11.0+cu128,
nvidia-cutlass-dsl 4.8.0). The SM clock was not locked; the timing CSVs record it per window.

## 1. Environment and patch

```bash
python kernel/env/verify_gpu.py                                    # must report sm_100 (B200)
bash kernel/env/setup_fa4.sh                                       # FA4 at the pinned commit + CuTe DSL; prints FA4_ROOT
python kernel/patch/apply_patch.py --fa4-root $FA4_ROOT --dry-run  # all ten anchors present and unique
python kernel/patch/apply_patch.py --fa4-root $FA4_ROOT            # apply (backups <file>.orig)
python kernel/reference/test_h15_reference.py                      # CPU: bit-level reference of the kernel operator
```

`patch/PATCH_NOTES.md` explains the edits (doubling only the exponential path, clamp at -252, round-to-nearest-even,
the 22-bit shift) and an installation note; `patch/softmax.diff` and `patch/flash_fwd_sm100.diff` show the result.
`python kernel/patch/revert_patch.py --fa4-root $FA4_ROOT` restores stock FA4.

## 2. Correctness gates (Appendix B.2, Figure S1)

```bash
bash kernel/correctness/run_gates.sh      # smoke test of both arms, single-tile gate, multi-tile gate
```

* `smoke_test.py`: fixed-input FP8 causal forward (seqlen 4096); the mean |O| must differ between arms and was
  identical in every measurement session (A0 0.188664, A1 0.188038).
* `gate_single_tile.py` (seqlen 128, one key tile): the kernel's probabilities, read back through a one-hot V, against
  the bit-level reference (`reference/h15_reference.py`); the tolerance is the E4M3 storage grid (paper: mean L1 per
  row 4.253e-3, maximum 2.469e-2).
* `gate_multi_tile.py` (seqlen 512, four tiles): running-maximum and rescale path; log2(P1/P0) within the octave
  envelope [-0.25, 0.335] plus the E4M3 tolerance, exact row maximum, row sums near 1.

The dumps (`L1_DUMP`, `L15_DUMP`) are the inputs of Figure S1 (`results/kernel/correctness_dumps/`).

## 3. Model-level fidelity on the BF16 path (Table 2, Table B2, Appendix B.2.1)

```bash
python kernel/fidelity/run_ppl_arm.py --arm 0 --model Qwen/Qwen2.5-1.5B --out ppl_A0.csv
python kernel/fidelity/run_ppl_arm.py --arm 1 --model Qwen/Qwen2.5-1.5B --out ppl_A1.csv
bash kernel/fidelity/run_all_models.sh    # the five models of Table 2
```

Each model runs with FA4 as its attention (`fidelity/fa4cute_attn.py`; head dimension 128, no logit softcap) on the 97
aligned 2048-token WikiText-103 blocks; the dNLL of a model is the per-block paired difference A1 - A0. The FP8 path
was not measured directly: its estimate is the PyTorch semantic simulation `h15tiled` of
`experiments/lattice_operators/` (the BF16 simulation `h15tiled_bf16` is the cross-check of this measurement).

## 4. Call latency, energy and traces (Section 5.5, Appendix B.3-B.4)

```bash
bash kernel/benchmark/run_timing_matrix.sh        # every latency cell of Tables 2 and B5, Figures 4C-D and S2A; energy loop
bash kernel/profile/run_nsys.sh                   # Nsight Systems CUDA trace per kernel and length (Table B4, Figure S2B)
bash kernel/sass_audit/compile.sh                 # static instruction audit (App. B.5); any CUDA toolkit, no GPU
```

* **Call latency** (`benchmark/bench_arm.py`): after 50 warm-up calls and a synchronization, windows of 100
  back-to-back calls of the FA4 Python interface are timed with two CUDA events; window time / 100 is the call
  latency (launch and dispatch gaps included). Seven windows per kernel and round; three-round cells alternate the
  kernel order A0A1, A1A0, A0A1 (the FP8 non-causal 8K cell ran A0 before A1 in every round). The speed-up of a cell
  is 100 (median A0 / median A1 - 1) over the pooled windows; the per-round spread is the population sd of the
  per-round speed-ups (a spread, not a confidence interval). Main shape: batch 1, 16 heads, head dimension 128.
* **Emulation settings** (Figure 4D, Table B5d): `--freq` / `--start-frg` set FA4's own polynomial-emulation share
  beta of the stock kernel; Rowmax-H15 replaces the primitive and has no beta.
* **Energy** (`profile/energy_loop.py`): FP8 causal 16K, 45 s loop per kernel, nvidia-smi samples at a requested
  100 ms interval; energy per forward = mean power x loop time / forwards (no idle power subtracted).
* **Kernel duration** (`profile/run_nsys.sh`, `profile/trace_target.py`): each kernel and length in its own process
  under `nsys profile --trace=cuda` (50 warm-up + 200 timed calls); kernel duration = median of the
  `cuda_gpu_kern_sum` report over all traced calls, call latency = host time per timed call. `profile/run_ncu.sh`
  (hardware counters) was refused with ERR_NVGPUCTRPERM in all of the paper's sessions; no counter result is reported.
* **External references** (`benchmark/bench_sdpa_reference.py`): PyTorch SDPA with cuDNN, flash or memory-efficient
  backends in BF16; context, not a controlled baseline.

Outputs use the file names of `results/kernel/` (`timing/`, `fidelity/`, `correctness_dumps/`, `sass_audit/`);
`paper/tables/check_tables.py` recomputes Tables 2 and B5 from them.  The two Nsight Systems summary tables in
`results/kernel/nsys/` were extracted from such reports: `kernel_vs_wall.csv` (median and minimum kernel duration and
instance count from `cuda_gpu_kern_sum`, host call latency per timed call) and `launch_breakdown.csv` (median
`cudaLaunchKernelExC` time and the kernel, launch and remaining shares of the timed interval);
`paper/tables/make_table_B4.py` writes Table B4 from them.
