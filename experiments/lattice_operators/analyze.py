#!/usr/bin/env python3
"""Numbers of Tables B1 and B2 and of the exponential-map ablation (Section 5.1-5.2, Appendix B) from run outputs.

    python experiments/lattice_operators/analyze.py                       # shipped reference results
    python experiments/lattice_operators/analyze.py --lattice-root runs/lattice_operators \\
        --ablation-root runs/exp_map_ablation --reference-root runs/characterization

Inputs (per-block files written by run.py and by the characterization pipeline):
  <lattice-root>/<model>/condition_blocks/*.csv      the lattice operators (Tables B1, B2)
  <lattice-root>/<model>/leakage/                    leak_rows_<id>.npz (or a precomputed leakage_summary.csv)
  <reference-root>/<model>/condition_blocks/         pot_m1 and response_pot_m2 (PoT m = 1, 2; Eq. 18)
  <kernel-root>/ppl_A0.csv, ppl_A1.csv               measured B200 kernel NLL per block (stock / patched FA4)
  <ablation-root>/<model>/condition_blocks/*.csv     the ablation run (softmax, S(x), S_k, Rowmax-H15, PoT m=2)

Outputs in <lattice-root>/<model>/analysis/: operator_summary.csv (dNLL [95% CI], PPL change, top-1), paired_contrasts.csv,
ratios.csv, leakage_summary.csv; in <ablation-root>/<model>/analysis/: exp_map_ablation.csv.  Every interval is the
95% percentile paired block bootstrap (5,000 replicates, seed 0).  CPU only.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from softmax_approx import stats  # noqa: E402

L = 2048
NB = 97
LABELS = {
    "pq_log2_b8": "log2-p 8-bit grid (absolute anchor, octave)",
    "pq_logsqrt2_b8": "log-sqrt2-p 8-bit grid (absolute anchor, half-octave)",
    "pot_m1": "PoT m=1 (rowmax anchor, octave)",
    "response_pot_m2": "PoT m=2 (rowmax anchor, half-octave)",
    "ilpot_v1": "absolute octave lattice (absolute-lattice ILPoT)",
    "ilpot_ha_v1": "ILPoT-H-A (absolute half-octave lattice, exact maximum entry)",
    "ilpot_h15_v1": "absolute-anchored {1,1.5} x 2^k lattice",
    "h15rowmax": "offline whole-row Rowmax-H15",
    "h15tiled": "Rowmax-H15, FP8-path semantic simulation",
    "h15tiled_bf16": "Rowmax-H15, BF16-path semantic simulation",
    "b200_kernel_bf16": "B200 patched FlashAttention-4 kernel, BF16 path (measured)",
}
PAIRS = [  # (table, contrast, a, b): estimate of mean_b [dNLL(a) - dNLL(b)]
    ("B1", "anchor: absolute -> rowmax, octave spacing", "pq_log2_b8", "pot_m1"),
    ("B1", "anchor: absolute -> rowmax, half-octave spacing", "pq_logsqrt2_b8", "response_pot_m2"),
    ("B1", "spacing: octave -> half-octave, absolute anchor", "pq_log2_b8", "pq_logsqrt2_b8"),
    ("B1", "spacing: octave -> half-octave, rowmax anchor", "pot_m1", "response_pot_m2"),
    ("B2", "FP8 semantic Rowmax-H15 - PoT m=2", "h15tiled", "response_pot_m2"),
    ("B2", "FP8 semantic Rowmax-H15 - ILPoT-H-A", "h15tiled", "ilpot_ha_v1"),
    ("B2", "FP8 semantic Rowmax-H15 - offline whole-row Rowmax-H15", "h15tiled", "h15rowmax"),
    ("B2", "BF16 semantic Rowmax-H15 - FP8 semantic Rowmax-H15", "h15tiled_bf16", "h15tiled"),
]
RATIOS = [("log2-p / PoT m=1", "pq_log2_b8", "pot_m1"),
          ("absolute-lattice ILPoT / log2-p", "ilpot_v1", "pq_log2_b8"),
          ("log-sqrt2-p / PoT m=2", "pq_logsqrt2_b8", "response_pot_m2"),
          ("log2-p / PoT m=2", "pq_log2_b8", "response_pot_m2")]
ABLATION = [("h15rowmax", "Rowmax-H15, H(x) = S(Q_1/2(x))"), ("response_pot_m2", "Rowmax-PoT m=2 (Eq. 18)"),
            ("schraudolph_rowmax", "unshifted Schraudolph S(x)"),
            ("schraudolph_rowmax_k2", "S(Q_{2^-2}(x)), k=2"), ("schraudolph_rowmax_k3", "S(Q_{2^-3}(x)), k=3")]


def rel(p: Path) -> str:
    """Path as written into output files: relative to the repository root when inside it."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return p.name


def load_blocks(root: Path, model: str, cid: str) -> pd.DataFrame | None:
    p = root / model / "condition_blocks" / f"{cid}__L{L}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).sort_values("block").drop_duplicates("block", keep="last")
    d = d[d.block.isin(range(NB))].reset_index(drop=True)
    return d if len(d) == NB else None


def kernel_blocks(kernel_root: Path) -> pd.DataFrame | None:
    f0, f1 = kernel_root / "ppl_A0.csv", kernel_root / "ppl_A1.csv"
    if not (f0.exists() and f1.exists()):
        return None
    d0 = pd.read_csv(f0).sort_values("block").reset_index(drop=True)
    d1 = pd.read_csv(f1).sort_values("block").reset_index(drop=True)
    if len(d0) != NB or not (d0.block.values == d1.block.values).all():
        return None
    return pd.DataFrame(dict(block=d0.block, n_pred=d0.n_pred, baseline_nll=d0.nll, nll=d1.nll,
                             delta_nll=d1.nll.to_numpy() - d0.nll.to_numpy()))


def summary_row(cid: str, d: pd.DataFrame, source: str) -> dict:
    e, lo, hi = stats.mean_ci(d.delta_nll.to_numpy(float))
    r = dict(condition=cid, label=LABELS.get(cid, cid), source=source, n_blocks=len(d), dnll=e, ci_low=lo,
             ci_high=hi, resolved=stats.resolved(lo, hi), ppl_change_pct=stats.ppl_change(e))
    if "top1_agreement" in d:
        r["top1_agreement"] = float(np.average(d.top1_agreement, weights=d.n_pred))
    return r


def analyze_lattices(lroot: Path, rroot: Path, kroot: Path, model: str) -> None:
    blocks, source = {}, {}
    for cid in LABELS:
        for root in (lroot, rroot):
            d = load_blocks(root, model, cid)
            if d is not None:
                blocks[cid], source[cid] = d, rel(root)
                break
    k = kernel_blocks(kroot)
    if k is not None:
        blocks["b200_kernel_bf16"], source["b200_kernel_bf16"] = k, rel(kroot)
    if not blocks:
        print(f"[lattices] no per-block files under {lroot} / {rroot}")
        return
    out = lroot / model / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows = [summary_row(c, d, source[c]) for c, d in blocks.items()]
    ref = blocks.get("pot_m1")
    if ref is not None:   # alignment of the per-block softmax baselines with the characterization run
        for r in rows:
            if r["condition"] != "b200_kernel_bf16":
                r["max_abs_baseline_diff_vs_reference"] = float(
                    np.abs(blocks[r["condition"]].baseline_nll.to_numpy() - ref.baseline_nll.to_numpy()).max())
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "operator_summary.csv", index=False)

    pairs = []
    for table, name, a, b in PAIRS:
        if a in blocks and b in blocks:
            x = blocks[a].delta_nll.to_numpy(float) - blocks[b].delta_nll.to_numpy(float)
            e, lo, hi = stats.mean_ci(x)
            pairs.append(dict(table=table, contrast=name, a=a, b=b, estimate=e, ci_low=lo, ci_high=hi,
                              resolved=stats.resolved(lo, hi), blocks_a_lower=int((x < 0).sum())))
    if "b200_kernel_bf16" in blocks and "h15tiled_bf16" in blocks:
        km, sm = blocks["b200_kernel_bf16"].delta_nll.mean(), blocks["h15tiled_bf16"].delta_nll.mean()
        pairs.append(dict(table="B2", contrast="simulation vs measured kernel (BF16 path): (kernel - sim) / kernel",
                          a="b200_kernel_bf16", b="h15tiled_bf16", estimate=float((km - sm) / km)))
    pd.DataFrame(pairs).to_csv(out / "paired_contrasts.csv", index=False)

    ratios = [dict(ratio=name, a=a, b=b, value=float(blocks[a].delta_nll.mean() / blocks[b].delta_nll.mean()))
              for name, a, b in RATIOS if a in blocks and b in blocks]
    pd.DataFrame(ratios).to_csv(out / "ratios.csv", index=False)

    leak_dir = lroot / model / "leakage"
    if any(leak_dir.glob("leak_rows_*.npz")):
        import leakage                         # needs scipy; recompute from the per-row arrays
        ls = leakage.summarize_leakage(leak_dir)
        ls.to_csv(leak_dir / "leakage_summary.csv", index=False)
    if (leak_dir / "leakage_summary.csv").exists():
        shutil.copyfile(leak_dir / "leakage_summary.csv", out / "leakage_summary.csv")

    pd.set_option("display.width", 220)
    cols = ["condition", "dnll", "ci_low", "ci_high", "ppl_change_pct"] + (
        ["top1_agreement"] if "top1_agreement" in summ else [])
    print(summ[cols].to_string(index=False, float_format=lambda z: f"{z:.4g}"))
    if pairs:
        print(pd.DataFrame(pairs)[["table", "contrast", "estimate", "ci_low", "ci_high"]].to_string(
            index=False, float_format=lambda z: f"{z:.4g}"))
    print(f"[lattices] wrote {rel(out)}")


def analyze_ablation(aroot: Path, model: str) -> None:
    nll = {}
    for cid in ["baseline"] + [c for c, _ in ABLATION]:
        d = load_blocks(aroot, model, cid)
        if d is not None:
            nll[cid] = d
    if "baseline" not in nll:
        print(f"[ablation] no softmax baseline under {aroot}")
        return
    base = nll["baseline"].nll.to_numpy(float)
    rows = []
    for cid, label in ABLATION:
        if cid not in nll:
            continue
        x = nll[cid].nll.to_numpy(float)
        if (nll[cid].n_pred.to_numpy() != nll["baseline"].n_pred.to_numpy()).any():
            raise SystemExit(f"{cid}: block sizes differ from the baseline")
        e, lo, hi = stats.mean_ci(x - base)
        r = dict(condition=cid, label=label, dnll_vs_softmax=e, ci_low=lo, ci_high=hi,
                 resolved=stats.resolved(lo, hi), ppl_change_pct=stats.ppl_change(e),
                 blocks_below_softmax=int((x < base).sum()),
                 max_abs_baseline_diff=float(np.abs(nll[cid].baseline_nll.to_numpy() - base).max()))
        if cid != "h15rowmax" and "h15rowmax" in nll:
            h = nll["h15rowmax"].nll.to_numpy(float)
            e2, lo2, hi2 = stats.mean_ci(x - h)
            r.update(dnll_vs_rowmax_h15=e2, vs_h15_ci_low=lo2, vs_h15_ci_high=hi2,
                     vs_h15_resolved=stats.resolved(lo2, hi2), blocks_below_rowmax_h15=int((x < h).sum()))
        rows.append(r)
    out = aroot / model / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "exp_map_ablation.csv", index=False)
    print(tab[["condition", "dnll_vs_softmax", "ci_low", "ci_high", "ppl_change_pct"]].to_string(
        index=False, float_format=lambda z: f"{z:.4g}"))
    print(f"[ablation] wrote {rel(out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen15")
    ap.add_argument("--lattice-root", default=str(ROOT / "results" / "lattice_operators"))
    ap.add_argument("--reference-root", default=str(ROOT / "results" / "characterization"))
    ap.add_argument("--kernel-root", default=str(ROOT / "results" / "kernel" / "fidelity"))
    ap.add_argument("--ablation-root", default=str(ROOT / "results" / "exp_map_ablation"))
    a = ap.parse_args()
    analyze_lattices(Path(a.lattice_root), Path(a.reference_root), Path(a.kernel_root), a.model)
    analyze_ablation(Path(a.ablation_root), a.model)


if __name__ == "__main__":
    main()
