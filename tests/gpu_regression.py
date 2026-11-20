#!/usr/bin/env python3
"""GPU regression: re-evaluate the first blocks of representative conditions and compare with results/.

    python tests/gpu_regression.py                  # Qwen2.5-1.5B, 2 blocks per condition (a few minutes, one GPU)
    python tests/gpu_regression.py --blocks 4 --conditions precision_K32_R4,h15tiled

Every condition goes through the same evaluation path as the paper's runs (CrossModelRunner.evaluate_group, FP32
scores, query chunk 512).  Characterization conditions are rebuilt from their stored forward specification
(results/characterization/qwen15/condition_meta/<id>__L2048.json); lattice conditions from
experiments/lattice_operators/run.py.  For each condition the per-block NLL, softmax-baseline NLL, vocabulary
KL/JSD and top-1 agreement of the rerun are compared with the shipped rows.  With the recorded software stack
(torch 2.5.1+cu121, transformers 5.14.1) the values were bit-identical; on other GPUs or library versions small
floating-point differences are expected, and the script reports the largest one.  A condition shipped in two runs (h15rowmax and
response_pot_m2 appear in the lattice run or the characterization run and again in the exponential-map ablation;
the two runs differ by at most 4.4e-16) is compared with the closer copy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lattice_operators"))

from softmax_approx import characterization as cm  # noqa: E402
import run as lattice_run  # noqa: E402

DEFAULT = [
    # Section 4 (characterization pipeline)
    "mean_sparse_softmax", "mean_sparse_uniform", "strict_mean_L05_uniform", "topk50_softmax",
    "geometry_exp_K16", "geometry_linear_K16", "precision_K32_R4", "alloc_K21_R1", "alloc_K21_R0.25",
    "recon_K21_R4_lerp", "pot_m1", "response_pot_m2", "rowmax_pot_K20_clamp", "rowmax_pot_K12_drop",
    "jsd_K0_Aminus", "jsd_10K0_Aplus", "rescue_K0_Aminus_q0p9",
    # Section 5 / Appendix B (lattice operators)
    "ilpot_v1", "pq_log2_b8", "pq_logsqrt2_b8", "ilpot_ha_v1", "ilpot_h15_v1", "h15rowmax", "h15tiled",
    "h15tiled_bf16", "schraudolph_rowmax", "schraudolph_rowmax_k2",
]
COLS = ["n_pred", "baseline_nll", "nll", "delta_nll", "output_kl_forward", "output_kl_reverse", "output_jsd",
        "top1_agreement"]


def reference(cid: str, model: str) -> tuple[cm.ConditionSpec, list[Path]]:
    """Forward spec of a condition and every shipped per-block file of it (some appear in two runs)."""
    res = ROOT / "results"
    paths = [res / sub / model / "condition_blocks" / f"{cid}__L2048.csv"
             for sub in ("characterization", "lattice_operators", "exp_map_ablation")]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no reference result for {cid}")
    meta = res / "characterization" / model / "condition_meta" / f"{cid}__L2048.json"
    if meta.exists():
        fc = json.loads(meta.read_text())["forward_condition"]
        return cm.ConditionSpec(fc["id"], fc["operator"], params=fc.get("params", {}), rescue_q=fc.get("rescue_q")), paths
    return lattice_run.spec_for(cid), paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen15")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--conditions", default=",".join(DEFAULT))
    ap.add_argument("--keep", default=None, help="keep the rerun outputs in this directory")
    a = ap.parse_args()
    out = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="approx_softmax_regression_"))
    cids = cm.parse_csv_str(a.conditions)
    specs = [reference(c, a.model) for c in cids]

    args = cm.build_argparser().parse_args(["--model", a.model, "--q-chunk", "512", "--tasks", "T00",
                                            "--max-blocks", str(a.blocks), "--out-root", str(out), "--resume"])
    cm.validate_args(args)
    lattice_run.register()
    runner = cm.CrossModelRunner(args)
    runner.load_runtime()
    runner.evaluate_group([s for s, _ in specs], 2048, task="REGRESSION")

    rows = []
    for (spec, ref_paths), cid in zip(specs, cids):
        new = pd.read_csv(runner.store.condition_block_path(spec.id, 2048)).sort_values("block").set_index("block")
        best = None
        for rp in ref_paths:
            ref = pd.read_csv(rp).drop_duplicates("block", keep="last").set_index("block").loc[new.index]
            diff = {c: float(np.nanmax(np.abs(new[c].to_numpy(float) - ref[c].to_numpy(float)))) for c in COLS if c in new}
            cand = dict(condition=cid, blocks=len(new), identical=all(v == 0.0 for v in diff.values()),
                        max_abs_diff_nll=diff["nll"], max_abs_diff_any=max(diff.values()),
                        reference=rp.relative_to(ROOT / "results").parts[0])
            if best is None or cand["max_abs_diff_any"] < best["max_abs_diff_any"]:
                best = cand
        rows.append(best)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(tab.to_string(index=False))
    n_id = int(tab.identical.sum())
    print(f"\n{n_id}/{len(tab)} conditions bit-identical to results/ on the first {a.blocks} blocks; "
          f"largest |difference| {tab.max_abs_diff_any.max():.3g}")
    if not a.keep:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
