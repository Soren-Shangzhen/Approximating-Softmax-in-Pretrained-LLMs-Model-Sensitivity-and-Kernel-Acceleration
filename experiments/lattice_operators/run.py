#!/usr/bin/env python3
"""Evaluate the lattice operators of Section 5 / Appendix B on a frozen model (Qwen2.5-1.5B in the paper).

Every condition goes through the characterization pipeline's evaluation path (CrossModelRunner.evaluate_group:
the same 97 aligned 2048-token WikiText-103 blocks, FP32 scores, query chunk 512, per-block NLL against the
softmax baseline, vocabulary-output fidelity).  Results are written as
<out-root>/<model>/condition_blocks/<condition>__L2048.csv (+ condition_meta/), the leakage probe to
<out-root>/<model>/leakage/.  analyze.py turns them into the numbers of Tables B1, B2 and the ablation.

    # Tables B1 and B2 (Section 5.1-5.2): eight operators + leakage-phase probes (about 15 min on one GPU)
    python experiments/lattice_operators/run.py --model qwen15 --out-root runs/lattice_operators --resume \\
        --conditions paper --leakage paper
    # exponential-map ablation (Appendix B.2): a separate run that recomputes softmax, Rowmax-H15 and PoT m=2
    python experiments/lattice_operators/run.py --model qwen15 --out-root runs/exp_map_ablation --resume \\
        --conditions ablation

Condition ids (see softmax_approx/lattice_operators.py for the definitions):
  ilpot_v1         absolute octave lattice ("absolute-lattice ILPoT", Table B1 notes)
  pq_log2_b8       8-bit log2 probability grid, absolute anchor (Table B1)
  pq_logsqrt2_b8   8-bit log-sqrt2 probability grid, absolute anchor (Table B1)
  ilpot_ha_v1      absolute half-octave lattice with exact maximum entry, ILPoT-H-A (Table B2 reference)
  ilpot_h15_v1     absolute-anchored {1,1.5} x 2^k lattice (Table B2 control)
  h15rowmax        offline whole-row Rowmax-H15 (Table B2 control; ablation reference)
  h15tiled         Rowmax-H15, FP8-path kernel semantics (Table 2 / B2 "FP8 semantic simulation")
  h15tiled_bf16    Rowmax-H15, BF16-path kernel semantics (Table 2 / B2 "BF16 semantic simulation")
  schraudolph_rowmax, schraudolph_rowmax_k2, schraudolph_rowmax_k3    S(x) and S(Q_{2^-k}(x)) (ablation)
and the pipeline conditions baseline (softmax), pot_m1 and response_pot_m2 (PoT m = 1 and m = 2, Eq. 18).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from softmax_approx import attention  # noqa: E402
from softmax_approx import characterization as cm  # noqa: E402
from softmax_approx import lattice_operators as lops  # noqa: E402
import leakage  # noqa: E402

TASK = "LATTICE"

# id -> (mode, family, description)
LATTICE = {
    "ilpot_v1": ("ilpot", "absolute_anchor", "absolute octave lattice, n = floor(sigma + 1/2)"),
    "pq_log2_b8": ("pq_log2_b8", "absolute_anchor", "probability-domain log2 grid, 8 bit"),
    "pq_logsqrt2_b8": ("pq_logsqrt2_b8", "absolute_anchor", "probability-domain log-sqrt2 grid, 8 bit"),
    "ilpot_ha_v1": ("ilpot_ha", "absolute_anchor", "absolute half-octave lattice, exact maximum entry"),
    "ilpot_h15_v1": ("ilpot_h15", "absolute_anchor", "absolute {1,1.5} x 2^k lattice"),
    "h15rowmax": ("h15rowmax", "rowmax_h15", "offline whole-row Rowmax-H15"),
    "h15tiled": ("h15tiled", "rowmax_h15", "Rowmax-H15, FP8-path kernel semantics (tile 128, running max)"),
    "h15tiled_bf16": ("h15tiled_bf16", "rowmax_h15", "Rowmax-H15, BF16-path kernel semantics (rescale threshold 8)"),
    "schraudolph_rowmax": ("schraudolph_rowmax", "exp_map", "unshifted Schraudolph S(x), whole-row anchor"),
    "schraudolph_rowmax_k2": ("schraudolph_rowmax_k2", "exp_map", "S(Q_{2^-2}(x)), whole-row anchor"),
    "schraudolph_rowmax_k3": ("schraudolph_rowmax_k3", "exp_map", "S(Q_{2^-3}(x)), whole-row anchor"),
}
PIPELINE = {
    "baseline": lambda: cm.c_baseline(),
    "pot_m1": lambda: cm.c_pot(1.0, role="K0_anchor"),
    "response_pot_m2": lambda: cm.c_pot(2.0, cid="response_pot_m2"),
}
PRESETS = {
    "paper": ["ilpot_v1", "pq_log2_b8", "pq_logsqrt2_b8", "ilpot_ha_v1", "ilpot_h15_v1",
              "h15rowmax", "h15tiled", "h15tiled_bf16"],
    "ablation": ["schraudolph_rowmax", "baseline", "h15rowmax", "response_pot_m2",
                 "schraudolph_rowmax_k2", "schraudolph_rowmax_k3"],
}
LEAK_PRESETS = {"paper": ["ilpot_v1", "pot_m1", "pq_log2_b8", "pq_logsqrt2_b8", "h15tiled"]}


def spec_for(cid: str) -> cm.ConditionSpec:
    if cid in PIPELINE:
        return PIPELINE[cid]()
    if cid not in LATTICE:
        raise SystemExit(f"unknown condition {cid}; known: {sorted(LATTICE) + sorted(PIPELINE)}")
    mode, family, desc = LATTICE[cid]
    params = dict(mode=mode, description=desc)
    if mode.startswith("h15"):
        params.update(n_block=lops.H15_NBLOCK, rounding="round_nearest_even", clamp=lops.CLAMP_LO_H15)
    if mode == "h15tiled_bf16":
        params["rescale_threshold"] = lops.RESCALE_THRESHOLD_BF16
    return cm.ConditionSpec(cid, "lattice", params=params, role="lattice_operator", family=family)


_pipeline_configure = cm.configure_condition


def configure_condition(attn, spec, n_layers, q_chunk):
    if spec.operator == "lattice":
        attn.CFG.reset()
        attn.CFG.q_chunk = int(q_chunk)
        attn.CFG.layer_mode = {i: spec.params["mode"] for i in range(n_layers)}
        return
    return _pipeline_configure(attn, spec, n_layers, q_chunk)


def register() -> None:
    """Install the lattice modes and the phase stash before the pipeline installs its probe patch."""
    lops.install(attention)
    leakage.install_phase_stash(attention)
    cm.configure_condition = configure_condition


def expand(arg: str, presets: dict) -> list[str]:
    out = []
    for item in cm.parse_csv_str(arg):
        out.extend(presets.get(item, [item]))
    return out


def main():
    ap = cm.build_argparser()
    ap.set_defaults(q_chunk="512", out_root="runs/lattice_operators")
    ap.add_argument("--conditions", default="paper",
                    help="comma-separated condition ids or presets: paper (Tables B1/B2), ablation (App. B.2)")
    ap.add_argument("--leakage", default="",
                    help="condition ids or preset 'paper' for the leakage-phase probe (first 4 blocks)")
    args = ap.parse_args()
    if args.tasks is None:
        args.tasks = "T00"            # needed by the runner's planner only; T00 is not run here
    cm.validate_args(args)
    if str(args.q_chunk).lower() == "auto":
        raise SystemExit("pass an explicit --q-chunk (512 in the paper)")
    conds = expand(args.conditions, PRESETS)
    leaks = expand(args.leakage, LEAK_PRESETS)
    specs = [spec_for(c) for c in conds]
    leak_specs = [spec_for(c) for c in leaks]

    register()
    runner = cm.CrossModelRunner(args)
    runner.load_runtime()
    L = int(args.seq)
    if specs:
        runner.evaluate_group(specs, L, task=TASK)
    leak_dir = runner.store.model_dir / "leakage"
    for sp in leak_specs:
        leakage.run_leak_probe(runner, sp, leak_dir)
    if leak_specs:
        summ = leakage.summarize_leakage(leak_dir)
        if summ is not None:
            cm.write_csv(leak_dir / "leakage_summary.csv", summ.to_dict("records"))
            print(summ[["condition", "phase", "n_rows", "mean_abs_L", "pearson_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
