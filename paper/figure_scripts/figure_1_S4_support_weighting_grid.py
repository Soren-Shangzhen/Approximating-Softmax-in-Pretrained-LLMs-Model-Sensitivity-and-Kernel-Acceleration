#!/usr/bin/env python3
"""Figure 1 and Figure S4 (Section 4.1, Appendix D.1): support, same-support weighting, full-range grid.

Figure 1: A support sweep of Qwen2.5-1.5B with softmax or uniform weights (T01); B strict single-layer
same-support control D_W(l) (T01B); C perplexity increase of the common K = 32, R = 4 grid in ten models (T03).
Figure S4: weighting geometry versus K, exponential vs linear level-to-weight map (T02).
Data: results/characterization/<model>/{T01,T01B} and condition_blocks; intervals are recomputed paired block
bootstraps (5,000 replicates, seed 0)."""
import sys, math; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
qw = DEEP_REF
roster = model_roster(); DISPLAY = dict(zip(roster.alias, roster.display))

AX = v13_axes("fig2_amount")
axA, axB, axC, axD = AX["A"], AX["B"], AX["C"], AX["D"]

# A: deep-reference support sweep (T01)
t01p = RESULTS_FORMAL / qw / "T01" / "conditions.csv"; t01 = pd.read_csv(t01p)
def arm(suffix):
    d = t01[t01.condition.str.startswith("topk") & t01.condition.str.endswith(suffix)].copy()
    d["ppl_pct"] = (d.ppl_ratio - 1) * 100
    return d.sort_values("support_density")
soft, unif = arm("softmax"), arm("uniform")
axA.plot(soft.support_density, soft.ppl_pct, "o-", color=C["blue"], label="softmax", mfc="white")
axA.plot(unif.support_density, unif.ppl_pct, "s--", color=C["vermil"], label="uniform", mfc=C["vermil"])
ms = t01[t01.condition == "mean_sparse_softmax"].iloc[0]; mu = t01[t01.condition == "mean_sparse_uniform"].iloc[0]
# both diamonds are open outlines, larger and thinner, so the series marker inside stays visible
axA.plot([ms.support_density], [(ms.ppl_ratio - 1) * 100], "D", color=C["blue"], ms=7.5, mfc="none", mew=0.7)
axA.plot([mu.support_density], [(mu.ppl_ratio - 1) * 100], "D", color=C["vermil"], ms=7.5, mfc="none", mew=0.7)
axA.set_xscale("log"); axA.set_yscale("log")
_x0, _x1 = axA.get_xlim(); axA.set_xlim(_x0, _x1 * 1.35)  # room on the right so the mean-threshold diamonds (ms 7.5) are not clipped
axA.set_xlabel("kept fraction of entries"); axA.set_ylabel("PPL increase (%)")
axA.axhline(1.0, color=C["vermil"], lw=0.9, ls="--")  # same 1% reference style as panel C
import matplotlib.patheffects as pe; import matplotlib.transforms as mtrans  # "1%" label at right end, white stroke
axA.text(1.0, 1.0, "1% ", transform=mtrans.blended_transform_factory(axA.transAxes, axA.transData), fontsize=6, color=C["vermil"], ha="right", va="bottom", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
axA.legend(loc="center right", bbox_to_anchor=(1.02, 0.61), handlelength=0.7, borderpad=0.1, ncol=1, columnspacing=0.6, handletextpad=0.3, labelspacing=0.2)  # two-column legend in the empty band between the curves
panel_label(axA, "A")
for _, r in pd.concat([soft, unif]).iterrows():
    rec("fig2 panelA point", f"{r.ppl_pct:.3f}% @ density {r.support_density:.4f}", r.ppl_pct, "100*(ppl_ratio-1)", t01p,
        f"row: condition={r.condition}; col: ppl_ratio", "{:.3f}", dict(kind="direct", path=str(t01p), row_filter={"condition": r.condition}, col="ppl_ratio", scale=1.0, post="pct_from_ratio"), rid=f"fig2A_{r.condition}")
for r, tag in [(ms, "mean_sparse_softmax"), (mu, "mean_sparse_uniform")]:
    rec("fig2 panelA mean-threshold", f"{(r.ppl_ratio-1)*100:.3f}%", (r.ppl_ratio - 1) * 100, "100*(ppl_ratio-1)", t01p, f"row: condition={tag}; col: ppl_ratio", "{:.3f}", dict(kind="direct", path=str(t01p), row_filter={"condition": tag}, col="ppl_ratio", scale=1.0, post="pct_from_ratio"), rid=f"fig2A_{tag}")

# B: strict same-support weighting control (T01B), 28 layers
ebp = RESULTS_FORMAL / qw / "T01B" / "effects_by_layer.csv"; eb = pd.read_csv(ebp).sort_values("target_layer")
# recompute from blocks (matches stored to <2e-6; see recompute_vs_stored.csv)
est, lo, hi = [], [], []
for L_ in eb.target_layer:
    x, paths = paired_delta(qw, f"strict_mean_L{L_:02d}_uniform", f"strict_mean_L{L_:02d}_softmax"); e, l, h = boot_mean_ci(x)
    est.append(e); lo.append(l); hi.append(h)
    rec("fig2 panelB layer effect", f"{fmt_sig(e)} [{fmt_sig(l)},{fmt_sig(h)}]", e, "paired block bootstrap mean (uniform - exact-relative), same support", paths, "col: delta_nll", "{:.4f}",
        dict(kind="boot", alias=qw, cond_a=f"strict_mean_L{L_:02d}_uniform", cond_b=f"strict_mean_L{L_:02d}_softmax", L=2048, root_a="formal", root_b="formal", which="est"), rid=f"fig2B_L{L_:02d}")
est, lo, hi = map(np.array, (est, lo, hi))
axB.errorbar(eb.target_layer, est, yerr=[est - lo, hi - est], fmt="o", color=C["green"], ecolor="k", ms=3, capsize=1.5, lw=0.6, elinewidth=0.6)
_rel = float(np.max((hi - lo) / est))  # report how wide the widest interval is relative to its estimate
print(f"fig1B: widest 95% interval is {_rel*100:.1f}% of its estimate (markers are wider than most bars)")
axB.set_yscale("log"); axB.axhline(0, color="k", lw=0.5)
axB.set_xlabel("target layer $\\ell$"); axB.set_ylabel("$D_W(\\ell)$", labelpad=1)  # labelpad, else the label sits on the tick text
axB.tick_params(axis="y", pad=1.5)
axB.set_xticks([0, 9, 18, 27])
panel_label(axB, "B")
n_pos_ci = int((lo > 0).sum())  # "28/28 intervals > 0" and the error-bar note are stated in the caption (panel too narrow)
print("fig2B max CI half-width ratio hi/est, est/lo:", float(np.max(hi / est)), float(np.max(est / lo)))
print("fig2B intervals above zero:", n_pos_ci, "of", len(lo))

# C: weighting geometry (T02): exp vs tested linear across K
Ks = [4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024]
for fam, col, mk, lab in [("exp", C["blue"], "o", "exponential"), ("linear", C["vermil"], "s", "linear (tested)")]:
    ys, los, his, ks = [], [], [], []
    for K in Ks:
        c = f"geometry_{fam}_K{K}"
        if not block_path(qw, c).exists(): continue
        x, paths = paired_delta(qw, c); e, l, h = boot_mean_ci(x)
        ys.append(e); los.append(l); his.append(h); ks.append(K)
        rec("fig2 panelC point", fmt_sig(e), e, "paired block bootstrap mean delta_nll", paths, "col: delta_nll", SIG3,
            dict(kind="boot", alias=qw, cond_a=c, cond_b=None, L=2048, root_a="formal", root_b="formal", which="est"), rid=f"fig2C_{c}")
    ys, los, his = map(np.array, (ys, los, his))
    axC.errorbar(ks, ys, yerr=[ys - los, his - ys], fmt=mk + "-", color=col, ecolor="k", ms=3, capsize=1.5, lw=0.6, label=lab, mfc="white" if fam == "exp" else col, elinewidth=0.6)
# breadth: other models at K=16 (T02 available models)
for a in ALL_MODELS:
    if a == qw or not block_path(a, "geometry_linear_K16").exists(): continue
    for fam, col in [("exp", C["blue"]), ("linear", C["vermil"])]:
        x, paths = paired_delta(a, f"geometry_{fam}_K16"); e = float(x.mean())
        axC.plot([16], [e], marker="x", color=col, ms=4, mew=0.8, ls="none")
        rec("fig2 panelC K16 other model", fmt_sig(e), e, "mean delta_nll over blocks", paths, "col: delta_nll", SIG3,
            dict(kind="boot", alias=a, cond_a=f"geometry_{fam}_K16", cond_b=None, L=2048, root_a="formal", root_b="formal", which="est"), rid=f"fig2C_K16_{a}_{fam}")
axC.plot([], [], "x", color="k", ms=4, label="others, $K{=}16$")
axC.set_xscale("log", base=2); axC.set_yscale("log")
axC.set_xlabel("grid intervals $K$ ($R{=}1$)"); axC.set_ylabel("$\\Delta$NLL")
axC.legend(loc="center right", bbox_to_anchor=(1.03, 0.64), handlelength=1.2, labelspacing=0.2, handletextpad=0.3); panel_label(axC, "C")  # legend in the empty region right of the exponential curve

# D: ten-model common precision point K=32, R=4
rows = []
for a in ALL_MODELS:
    x, paths = paired_delta(a, "precision_K32_R4"); e, l, h = boot_mean_ci(x)
    rows.append(dict(alias=a, est=(math.exp(e) - 1) * 100, lo=(math.exp(l) - 1) * 100, hi=(math.exp(h) - 1) * 100,
                     params=roster.set_index("alias").loc[a, "parameter_count"]))
    rec("fig2 panelD bar", f"{(math.exp(e)-1)*100:.3f}%", (math.exp(e) - 1) * 100, "100*(exp(boot mean delta_nll)-1)", paths, "col: delta_nll", "{:.3f}",
        dict(kind="boot", alias=a, cond_a="precision_K32_R4", cond_b=None, L=2048, root_a="formal", root_b="formal", which="ppl_pct"), rid=f"fig2D_{a}")
d = pd.DataFrame(rows).sort_values("params")
xpos = np.arange(len(d))
# horizontal bars (models on the y axis) so that model names fit at 7 pt; same values and value range
axD.barh(xpos, d.est, color=C["sky"], edgecolor="k", lw=0.5, height=0.7)
axD.errorbar(d.est, xpos, xerr=[d.est - d.lo, d.hi - d.est], fmt="none", ecolor="k", capsize=1.5, lw=0.6, elinewidth=0.6)
axD.axvline(1.0, color=C["vermil"], lw=0.9, ls="--"); axD.text(1.03, len(d) - 1, "1%", ha="left", va="center", color=C["vermil"])
axD.set_yticks(xpos); axD.set_yticklabels([DISPLAY[a] for a in d.alias]); axD.invert_yaxis()
axD.set_xlabel("PPL increase (%)"); axD.set_xlim(0, max(1.15, d.hi.max() * 1.15))
panel_label(axD, "D")
v13_save()
flush_manifest("fig2_amount")
