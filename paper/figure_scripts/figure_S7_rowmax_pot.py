#!/usr/bin/env python3
"""Figure S7 (Section 4.4, Appendix D.5): Rowmax-PoT across models and tail cutoffs.

A perplexity increase of the K_max = 20 clamp condition in ten models; B Qwen2.5-1.5B K_max sweep with clamp
and drop tail modes.  Data: results/characterization/<model>/condition_blocks/rowmax_pot_K*_{clamp,drop}."""
import sys, math; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
qw = DEEP_REF
roster = model_roster(); DISPLAY = dict(zip(roster.alias, roster.display)); PAR = roster.set_index("alias").parameter_count
fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.5), layout="constrained")  # side by side, 7 pt
def pp(v): return (math.exp(v) - 1) * 100
rows = []
for a in ALL_MODELS:
    x, paths = paired_delta(a, "rowmax_pot_K20_clamp"); e, l, h = boot_mean_ci(x); rows.append(dict(alias=a, est=pp(e), lo=pp(l), hi=pp(h), params=PAR[a]))
    rec("figS3 panelA bar", f"{pp(e):.3f}%", pp(e), "100*(exp(paired block bootstrap mean delta_nll)-1)", paths, "col: delta_nll", "{:.3f}", dict(kind="boot", alias=a, cond_a="rowmax_pot_K20_clamp", cond_b=None, L=2048, root_a="formal", root_b="formal", which="ppl_pct"), rid=f"figS3A_{a}")
d = pd.DataFrame(rows).sort_values("params"); xp = np.arange(len(d))
axA.bar(xp, d.est, color=C["green"], edgecolor="k", lw=0.5, width=0.7); axA.errorbar(xp, d.est, yerr=[d.est - d.lo, d.hi - d.est], fmt="none", ecolor="k", capsize=1.5, lw=0.6, elinewidth=0.6)
axA.axhline(1, color=C["vermil"], ls="--", lw=0.9); axA.text(-0.5, 1.03, "1% PPL", ha="left", va="bottom", color=C["vermil"])
axA.set_xticks(xp); axA.set_xticklabels([DISPLAY[a] for a in d.alias], rotation=45, ha="right", fontsize=6); axA.set_ylabel("PPL increase (%)"); panel_label(axA, "A")
for mode, col, mk, Ks in [("clamp", C["blue"], "o", [12, 14, 16, 20, 24, 32, 48]), ("drop", C["orange"], "s", [12, 16, 20, 24])]:
    es, los, his = [], [], []
    for K in Ks:
        c = f"rowmax_pot_K{K}_{mode}"; x, paths = paired_delta(qw, c); e, l, h = boot_mean_ci(x); es.append(pp(e)); los.append(pp(l)); his.append(pp(h))
        rec("figS3 panelB point", f"{pp(e):.3f}%", pp(e), "100*(exp(paired block bootstrap mean delta_nll)-1)", paths, "col: delta_nll", "{:.3f}", dict(kind="boot", alias=qw, cond_a=c, cond_b=None, L=2048, root_a="formal", root_b="formal", which="ppl_pct"), rid=f"figS3B_{c}")
    es, los, his = map(np.array, (es, los, his))
    axB.errorbar(Ks, es, yerr=[es - los, his - es], fmt=mk + "-", ecolor="k", color=col, mfc="white" if mode == "drop" else col, ms=4, capsize=1.5, lw=0.6, label=f"tail mode: {mode}", elinewidth=0.6)
axB.set_yscale("log"); axB.axhline(1, color=C["vermil"], ls="--", lw=0.9); axB.set_xlabel("$K_{\\max}$ (Qwen2.5-1.5B)"); axB.set_ylabel("PPL increase (%)"); axB.legend(); panel_label(axB, "B")
save(fig, "figS3_rowmax_pot_crossmodel"); flush_manifest("figS3_rowmax_pot_crossmodel")
