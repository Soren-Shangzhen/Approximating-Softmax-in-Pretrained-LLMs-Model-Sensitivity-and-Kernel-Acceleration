#!/usr/bin/env python3
"""Figure 3 (Section 4.3, Appendix D.3): attention divergence and model loss.

A dNLL against mean row-wise JSD for the 56 interventions of Qwen2.5-1.5B with the in-sample log-log fit (T07);
B flattening-minus-sharpening contrast D_T at J0 and 10 J0 after matching the mean JSD in each model (T08).
Data: results/characterization/<model>/{T07,T08,T09} and condition_blocks."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
qw = DEEP_REF
roster = model_roster(); DISPLAY = dict(zip(roster.alias, roster.display)); PAR = roster.set_index("alias").parameter_count

AX = v13_axes("fig4_structure")
axA, axB, axC = AX["A"], AX["B"], AX["C"]

# A: response map
dpp = RESULTS_FORMAL / qw / "T07" / "distortion_points.csv"; dp = pd.read_csv(dpp)
fsp = RESULTS_FORMAL / qw / "T07" / "fit_summary.csv"; fs = pd.read_csv(fsp)
g = fs[(fs.regime == "all") & (fs.model == "global")].iloc[0]
fam_style = {"exp_quant": (C["blue"], "o", "exp grid"), "pot": (C["green"], "^", "PoT"),
             "temperature_flatten": (C["orange"], "s", "flatten"), "temperature_sharpen": (C["purple"], "D", "sharpen")}
use = dp[dp.usable_loglog.astype(bool)] if "usable_loglog" in dp else dp
for fam, (col, mk, lab) in fam_style.items():
    s = use[use.family == fam]
    axA.plot(s.js_mean, s.delta_nll, mk, color=col, mfc="white", ms=3.5, mew=0.8, label=lab, ls="none")
xs = np.logspace(np.log10(use.js_mean.min()), np.log10(use.js_mean.max()), 50)
axA.plot(xs, np.exp(g.intercept) * xs ** g.slope, "-", color="k", lw=0.8)
axA.text(0.97, 0.04, f"slope {g.slope:.2f}\n$R^2$ {g.r2:.3f}, $n$ = {int(g.n)}", transform=axA.transAxes, va="bottom", ha="right")  # stats lower right, legend upper left (both off the diagonal)
axA.set_xscale("log"); axA.set_yscale("log"); axA.set_xlabel("mean row-wise JSD"); axA.set_ylabel("$\\Delta$NLL")
axA.legend(loc="upper left", handletextpad=0.1, labelspacing=0.15, borderaxespad=0.1); panel_label(axA, "A")
rec("fig4 panelA fit", f"{g.slope:.2f}", g.slope, "direct", fsp, "row: regime=all; model=global; col: slope", "{:.2f}", dict(kind="direct", path=str(fsp), row_filter={"regime": "all", "model": "global"}, col="slope", scale=1.0), rid="fig4A_slope")
rec("fig4 panelA fit", f"{g.r2:.3f}", g.r2, "direct", fsp, "row: regime=all; model=global; col: r2", "{:.3f}", dict(kind="direct", path=str(fsp), row_filter={"regime": "all", "model": "global"}, col="r2", scale=1.0), rid="fig4A_r2")
rec("fig4 panelA fit", f"{int(g.n)}", g.n, "direct", fsp, "row: regime=all; model=global; col: n", "{:.0f}", dict(kind="direct", path=str(fsp), row_filter={"regime": "all", "model": "global"}, col="n", scale=1.0), rid="fig4A_n")
rec("fig4 panelA points", f"{len(use)} points", len(use), "count of usable_loglog rows", dpp, "col: usable_loglog", "{:.0f}", dict(kind="count", what="t07_points"), rid="fig4A_npoints")

# B: ten-model D_T at K0 and 10K0
rows = []
for a in ALL_MODELS:
    for lab in ["K0", "10K0"]:
        x, paths = paired_delta(a, f"jsd_{lab}_Aminus", f"jsd_{lab}_Aplus"); e, l, h = boot_mean_ci(x)
        rows.append(dict(alias=a, lab=lab, est=e, lo=l, hi=h, params=PAR[a], resolved=(l > 0) or (h < 0)))
        for which, v in [("est", e), ("lo", l), ("hi", h)]:
            rec(f"fig4 panelB D_T({lab})", SIG3P.format(v), v, {"est": "paired block bootstrap mean (A- minus A+)", "lo": "bootstrap 2.5%", "hi": "bootstrap 97.5%"}[which], paths, "col: delta_nll", "{:+.5f}",
                dict(kind="boot", alias=a, cond_a=f"jsd_{lab}_Aminus", cond_b=f"jsd_{lab}_Aplus", L=2048, root_a="formal", root_b="formal", which=which), rid=f"fig4B_{a}_{lab}_{which}")
d = pd.DataFrame(rows); order = sorted(ALL_MODELS, key=lambda a: PAR[a]); xp = {a: i for i, a in enumerate(order)}
for lab, off, col, mk in [("K0", -0.18, C["blue"], "o"), ("10K0", 0.18, C["vermil"], "s")]:
    s = d[d.lab == lab]
    for res, mfc in [(True, col), (False, "white")]:
        ss = s[s.resolved == res]
        if len(ss) == 0: continue
        # horizontal layout (models on the y axis); filled = resolved, hollow = unresolved (stated in the caption)
        axB.errorbar(ss.est, [xp[a] + off for a in ss.alias], xerr=[ss.est - ss.lo, ss.hi - ss.est], fmt=mk, color=col, mfc=mfc,
                     ecolor="k", elinewidth=0.6, capsize=1.5, lw=0.6, ms=4,
                     label=(("$J_0$" if lab == "K0" else "$10J_0$") if res else None))
axB.axvline(0, color="k", lw=0.6); axB.set_xscale("symlog", linthresh=2e-3); axB.margins(x=0.08)
axB.set_xticks([-1e-2, 0, 1e-2, 1]); axB.set_xticklabels(["$-10^{-2}$", "0", "$10^{-2}$", "$10^{0}$"])  # sparse ticks at 7 pt
axB.set_yticks(range(len(order))); axB.set_yticklabels([DISPLAY[a] for a in order]); axB.invert_yaxis()
axB.set_xlabel("$D_T$ (symlog)")
# the legend sat on the Llama-3.2-3B row and its samples read as that model's points
# the legend names the two encodings separately -- marker shape = distortion level,
# fill = whether the 95% interval excludes zero.  the fill entry carries both shapes, a grey
# hollow circle and a grey hollow square, so it reads as a convention and not as a third series.
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
_hc, = axB.plot([], [], "o", color="0.45", mfc="white", ms=4, mew=0.8, ls="none")
_hs, = axB.plot([], [], "s", color="0.45", mfc="white", ms=4, mew=0.8, ls="none")
_lh, _ll = axB.get_legend_handles_labels()
axB.legend(_lh + [(_hc, _hs)], _ll + ["CI includes 0"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.45)},
           loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, columnspacing=0.7,
           handletextpad=0.25, borderaxespad=0.1, borderpad=0.2)
panel_label(axB, "B")
axB.set_title(axB.get_title(loc="left"), loc="left", pad=14)  # room for the three-entry legend  # inside, upper right (empty at x > 0.05 for the first rows); panel_label(axB, "B")

# C: matched-TV robustness (qwen15, llama1)
rows = []
for a in ["qwen15", "llama1"]:
    for lab in ["K0", "10K0"]:
        xj, pj = paired_delta(a, f"jsd_{lab}_Aminus", f"jsd_{lab}_Aplus"); ej, lj, hj = boot_mean_ci(xj)
        xt, pt = paired_delta(a, f"tv_{lab}_Aminus", f"tv_{lab}_Aplus"); et, lt, ht = boot_mean_ci(xt)
        rows.append(dict(alias=a, lab=lab, ej=ej, lj=lj, hj=hj, et=et, lt=lt, ht=ht))
        for which, v in [("est", et), ("lo", lt), ("hi", ht)]:
            rec(f"fig4 panelC D_TV({lab})", SIG3P.format(v), v, {"est": "paired block bootstrap mean (TV-matched A- minus A+)", "lo": "bootstrap 2.5%", "hi": "bootstrap 97.5%"}[which], pt, "col: delta_nll", "{:+.5f}",
                dict(kind="boot", alias=a, cond_a=f"tv_{lab}_Aminus", cond_b=f"tv_{lab}_Aplus", L=2048, root_a="formal", root_b="formal", which=which), rid=f"fig4C_{a}_{lab}_{which}")
d = pd.DataFrame(rows); xt_ = np.arange(len(d))
axC.errorbar(xt_ - 0.15, d["ej"], yerr=[d["ej"] - d["lj"], d["hj"] - d["ej"]], fmt="o", color=C["blue"], ecolor="k", capsize=1.5, ms=4, lw=0.6, label="JSD-matched $D_T$", elinewidth=0.6)
axC.errorbar(xt_ + 0.15, d["et"], yerr=[d["et"] - d["lt"], d["ht"] - d["et"]], fmt="^", color=C["green"], mfc="white", ecolor="k", capsize=1.5, ms=4, lw=0.6, label="TV-matched $D_{TV}$", elinewidth=0.6)
SHORT = {"qwen15": "Qwen2.5-1.5B", "llama1": "Llama-3.2-1B"}
axC.axhline(0, color="k", lw=0.6); axC.set_xticks(xt_); axC.set_xticklabels([f"{SHORT[r.alias]}\n{'$J_0$' if r.lab=='K0' else '$10J_0$'}" for r in d.itertuples()], rotation=30, ha="right")
axC.set_ylabel("directional contrast"); axC.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1); panel_label(axC, "C")
v13_save(); flush_manifest("fig4_structure")
