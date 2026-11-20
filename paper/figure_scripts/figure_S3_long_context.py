#!/usr/bin/env python3
"""Figure S3 (Appendix C.2): within-context allocation D_R and matched-JSD direction D_T at 2K, 8K and 16K.

Each context has its own baseline, J0 and matched temperatures (T11).  Data: results/characterization/
{qwen15,llama3}/condition_blocks and results/characterization_large_T11_T12/{qwen72,llama70}/condition_blocks;
95% paired block-bootstrap intervals, hollow markers when the interval includes zero."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
roster = model_roster(); DISPLAY = dict(zip(roster.alias, roster.display))
SPEC = {"qwen72": ("70b", [2048, 8192, 16384]), "llama70": ("70b", [2048, 8192, 16384]), "qwen15": ("formal", [2048, 8192, 16384]), "llama3": ("formal", [2048, 8192])}
STYLE = {"qwen72": (C["blue"], "o", 1.0), "llama70": (C["vermil"], "s", 1.0), "qwen15": (C["sky"], "^", 0.7), "llama3": (C["orange"], "D", 0.7)}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.5), layout="constrained")  # side by side, 7 pt
for ax, (eff, ca, cb, tag, ylabel) in zip((axA, axB), [("D_R", "longctx_alloc_R1", "longctx_alloc_R4", "DR", "$D_R$"),
                                                       ("D_T", "longctx_Aminus", "longctx_Aplus", "DT", "$D_T$")]):
    for k, (a, (root, Ls)) in enumerate(SPEC.items()):
        col, mk, alpha = STYLE[a]
        off = (k - 1.5) * 0.09
        for L in Ls:
            x, paths = paired_delta(a, ca, cb, L, root); e, l, h = boot_mean_ci(x)
            resolved = (l > 0) or (h < 0)
            ax.errorbar([np.log2(L) + off], [e], yerr=[[e - l], [h - e]], fmt=mk, ecolor="k", color=col, mfc=col if resolved else "white", alpha=alpha, capsize=1.5, ms=4.5, lw=0.6, elinewidth=0.6)
            for which, v in [("est", e), ("lo", l), ("hi", h)]:
                rec(f"fig7 panel{'A' if tag=='DR' else 'B'} {eff}", SIG3P.format(v), v, {"est": "paired block bootstrap mean", "lo": "bootstrap 2.5%", "hi": "bootstrap 97.5%"}[which], paths, f"col: delta_nll; seq_len={L}", "{:+.5f}",
                    dict(kind="boot", alias=a, cond_a=ca, cond_b=cb, L=L, root_a=root, root_b=root, which=which), rid=f"fig7_{tag}_{a}_L{L}_{which}")
        ax.plot([], [], mk, color=col, alpha=alpha, label=DISPLAY[a])
    ax.plot([], [], "o", color="k", mfc="white", label="hollow = interval crosses 0")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks([11, 13, 14]); ax.set_xticklabels(["2K", "8K", "16K"]); ax.set_xlabel("context length (tokens)")
    ax.set_ylabel(ylabel)
fig.legend(*axA.get_legend_handles_labels(), loc="outside upper center", ncols=5, handletextpad=0.2, columnspacing=0.8)  # one figure-level legend
panel_label(axA, "A"); panel_label(axB, "B")
save(fig, "fig7_longcontext"); flush_manifest("fig7_longcontext")
