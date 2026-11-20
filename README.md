# Approximating Softmax in Pretrained LLMs: Model Sensitivity and Kernel Acceleration

Code and reference results for the paper (arXiv:2609.33586). The paper replaces only the map from a row of attention
scores to its normalized weights in ten frozen decoder-only models (0.5B-72B), measures which properties of that
map the models need (support, within-support weighting, grid resolution and its allocation, divergence direction),
and evaluates Rowmax-H15, a coarse exponential anchored at the row maximum, inside FlashAttention-4 on NVIDIA B200.

## Repository layout

| Path | What it implements | Paper |
|---|---|---|
| `softmax_approx/attention.py` | attention backend (Transformers attention interface): FP32 scores, the model's scaling, softcap and mask, and every Section 4 intervention on the score-to-probability map: mean-threshold and top-k support with softmax or uniform weights, full-range K-interval grid with allocation ratio R and three reconstruction rules, PoT and temperature perturbations, Rowmax-PoT | Sec. 3-4, App. A.1-A.7 |
| `softmax_approx/characterization.py` | evaluation pipeline and tasks T00-T13: 97 aligned 2048-token WikiText-103 blocks, per-block NLL against the softmax baseline, vocabulary-output fidelity, divergence matching, high-mass leakage / local pV / rescue probes, long context, zero-shot, paired block bootstrap | Sec. 4; App. A, C, D |
| `softmax_approx/multigpu/` | the same pipeline with the model sharded over several GPUs (Llama-3.1-70B, Qwen2.5-72B) | App. C |
| `softmax_approx/lattice_operators.py` | Section 5 operators: absolute vs row-maximum anchors (octave, half-octave, probability-domain log2 / log-sqrt2 grids), Rowmax-H15 (bit-level construction; offline, FP8-path and BF16-path kernel semantics), Schraudolph maps S(x) and S_k | Sec. 5.1-5.2; App. B.1-B.2 |
| `softmax_approx/{model_loader,data,stats}.py` | model loading with the backend registered, the WikiText-103 token stream, the paired block bootstrap | Sec. 3.1; App. A.2, A.9 |
| `experiments/characterization/` | run scripts for the ten models (`run_models.sh`, `run_large_models.sh`, `large_model_entry.py`) | Sec. 4; App. C, D |
| `experiments/lattice_operators/` | `run.py` (lattice operators, Rowmax-H15 semantic simulations, exponential-map ablation, leakage-phase probe), `analyze.py` (numbers of Tables B1, B2 and the ablation), `bit_identity_test.py` | Sec. 5.1-5.2; App. B.1-B.2 |
| `kernel/` | Rowmax-H15 patch for FlashAttention-4, bit-level reference, correctness gates, model fidelity on B200, call-latency matrix, energy, Nsight traces, static instruction audit; see `kernel/README.md` | Sec. 5.2-5.5; App. B |
| `paper/` | scripts of every figure, the LaTeX sources of the tables, `tables/check_tables.py` (recomputes every table number); see `paper/README.md` | all figures and tables |
| `results/` | the result files behind every figure and table (per-block CSVs, task outputs, B200 measurements); see `results/README.md` | |
| `tests/` | CPU tests of the operators and a GPU regression against `results/` | |

## Installation

```bash
pip install -r requirements.txt
```

The characterization and lattice runs used Python 3.11, torch 2.5.1+cu121, transformers 5.14.1 and datasets 5.0.0
(lm-evaluation-harness 0.4.12 for the zero-shot task), BF16 weights with FP32 attention scores and score-to-probability
map. Models up to 8B fit on one 24 GB GPU (Qwen2.5-1.5B takes about one second per block and condition); the 70B/72B
runs used three 80 GB GPUs. Model weights and WikiText-103 are downloaded from Hugging Face (Llama, Gemma and Mistral
weights are gated). The B200 experiments need FlashAttention-4 at a pinned commit (`kernel/env/setup_fa4.sh`).
Run every command from the repository root.

## Quick start without a GPU

```bash
python tests/test_attention_backend.py && python tests/test_lattice_operators.py   # operator tests (CPU, seconds)
python experiments/lattice_operators/bit_identity_test.py                         # App. B.2 bit-level test, k = 0..13
python experiments/lattice_operators/analyze.py       # Tables B1, B2 and the ablation from results/
bash paper/make_all.sh                                # all figures, Tables B4 and D3, check of every table number
```

`paper/make_all.sh` rewrites `paper/figures/` and `paper/tables/table_B4_nsys.tex`, `table_D3_ggap.tex` and ends
with `772/772 checks pass` (every numeric cell of Tables 2, 3, B1, B2, B5, C1, C2, D1, D2 recomputed from
`results/`). With Python 3.13.5, numpy 2.4.6, pandas 3.0.5 and matplotlib 3.11.0 the figures are byte-identical to
those of the paper.

## Reproducing the experiments

### Section 4: sensitivity of frozen models (ten models)

```bash
python -m softmax_approx.characterization --model qwen15 --dry-run                    # task plan of a model
python -m softmax_approx.characterization --model qwen15 --out-root runs/characterization --resume
bash experiments/characterization/run_models.sh          # the eight single-GPU models
bash experiments/characterization/run_large_models.sh    # Llama-3.1-70B, Qwen2.5-72B (+ T11/T12 extension)
```

Each model runs the tasks of its profile (`MODEL_REGISTRY` / `PROFILE_TASKS` in `softmax_approx/characterization.py`):
Qwen2.5-1.5B all of T00-T13; Qwen2.5-0.5B/3B, Llama-3.2-1B/3B and Gemma-2-2B the core set; Llama-3.1-8B,
Mistral-7B-v0.3 and the two large models the modern-core set. `--tasks T10` runs one task and its dependencies. All
statistical and engineering settings are frozen in the defaults and written to `<out-root>/<model>/execution_manifest.json`;
a run refuses to resume into a directory created with different settings.

| Task | Content | Paper |
|---|---|---|
| T00 | engineering gates (query chunking, agreement with the eager implementation, padding, quantizer mirror, probe identity) | App. A.2 |
| T01, T01B | support sweep; strict single-layer same-support control D_W(l) | Sec. 4.1, Fig. 1A-B, App. A.3, D.1 |
| T02, T03 | weighting geometry (exponential vs linear); precision grid, K=32, R=4 across models | Fig. 1C, Fig. S4, Table 3 |
| T04 | allocation: R sweep, D_R(21), I_R, error geometry, mass concentration | Sec. 4.2, Fig. 2, Table D1 |
| T05, T06 | reconstruction rules; PoT perturbation audit, J0, Rowmax-PoT | Sec. 4.4, Fig. S7, Table 3 |
| T07, T08, T09 | JSD-dNLL response map; matched-JSD D_T(J0), D_T(10 J0); matched TV | Sec. 4.3, Fig. 3, Table D2 |
| T10 | high-mass leakage, local pV output, mass-partition rescue G_gap(10 J0) | Figs. S5-S6, Table D3 |
| T11, T12, T13 | long context; zero-shot (lm-evaluation-harness); vocabulary-output fidelity | App. C, Fig. S3, Tables C1-C2 |

### Section 5.1-5.2: anchor, resolution and Rowmax-H15 semantics (Qwen2.5-1.5B)

```bash
python experiments/lattice_operators/run.py --model qwen15 --out-root runs/lattice_operators --resume \
    --conditions paper --leakage paper            # 8 operators of Tables B1/B2 + leakage-phase probes
python experiments/lattice_operators/run.py --model qwen15 --out-root runs/exp_map_ablation --resume \
    --conditions ablation                         # App. B.2: softmax, Rowmax-H15, PoT m=2, S(x), S_k (k=2,3)
python experiments/lattice_operators/analyze.py --lattice-root runs/lattice_operators \
    --ablation-root runs/exp_map_ablation --reference-root runs/characterization
```

The runs reuse the characterization pipeline's evaluation path (same blocks, query chunk 512, baseline recomputed
through the same backend); PoT m=1 and m=2 (the row-maximum anchors of Table B1) come from the characterization run
of Qwen2.5-1.5B. `h15tiled` and `h15tiled_bf16` are the PyTorch semantic simulations of the patched kernel's FP8 and
BF16 paths (Table 2, Table B2).

### Section 5.3-5.5: Rowmax-H15 in FlashAttention-4 on B200

See `kernel/README.md`: patch, correctness gates, BF16 model fidelity of five models, the call-latency matrix
(Tables 2, B5), energy, Nsight Systems traces (Table B4) and the static instruction audit (App. B.5).

### Checking a rerun

```bash
python tests/gpu_regression.py      # re-evaluates 27 representative conditions on 2 blocks and compares with results/
APPROX_SOFTMAX_RESULTS=runs bash paper/make_all.sh   # figures and table checks from a complete set of reruns
```

With the recorded software stack the rerun values were bit-identical to `results/`; other GPUs or library versions
can differ in the last floating-point bits. A rerun directory used with `APPROX_SOFTMAX_RESULTS` must contain the same
subdirectories as `results/`.

## Conventions and names

* **Statistics** (App. A.9): the statistical unit is the aligned block. A contrast is the mean over blocks of a
  per-block difference with a two-sided 95% percentile paired bootstrap (5,000 replicates, seed 0, block indices
  shared across conditions); it is *resolved* when the interval excludes zero. Perplexity change = 100 (exp(dNLL) - 1).
  B200 speed-up = 100 (median A0 / median A1 - 1) over the pooled timing windows.
* **Condition ids**: `K0` in an id is the reference divergence J0 of the paper (`jsd_10K0_Aminus` = flattening matched
  to 10 J0; `Aminus`/`Aplus` = flattening/sharpening); `alloc_K21_R4` = K=21, R=4; `precision_K32_R4` = the common
  grid; `pot_m1`, `response_pot_m2` = PoT with m = 1, 2; `rowmax_pot_K20_clamp` = Rowmax-PoT, K_max = 20;
  `ilpot_v1` = absolute octave lattice ("absolute-lattice ILPoT"); `ilpot_ha_v1` = ILPoT-H-A; `ilpot_h15_v1` =
  absolute-anchored {1, 1.5} x 2^k lattice; `pq_log2_b8`, `pq_logsqrt2_b8` = 8-bit probability-domain grids;
  `h15rowmax`, `h15tiled`, `h15tiled_bf16` = Rowmax-H15 offline / FP8-path / BF16-path semantics.
* **In the code**: BTA is the intervention attention backend (registered in Transformers as `approx_softmax`); P4 is
  the high-mass redistribution probe (leakage, local pV, rescue; App. A.8); A0 / A1 are stock FA4 / the Rowmax-H15
  kernel; T00-T13 are the pipeline tasks above.

## Citation

```bibtex
@article{zhu2026approximating,
  title   = {Approximating Softmax in Pretrained {LLMs}: Model Sensitivity and Kernel Acceleration},
  author  = {Zhu, Shangzhen and Hu, Muyan and Kozlowski, Tomasz},
  journal = {arXiv preprint arXiv:2609.33586},
  year    = {2026}
}
```

## License

MIT (`LICENSE`). FlashAttention-4 (BSD 3-Clause License) is not redistributed here; the patch in `kernel/patch/`
modifies files of a clone of it at the pinned commit.
