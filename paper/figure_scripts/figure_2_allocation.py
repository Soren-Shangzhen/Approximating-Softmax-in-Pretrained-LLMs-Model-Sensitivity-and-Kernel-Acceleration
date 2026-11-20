#!/usr/bin/env python3
"""Figure 2 (Section 4.2, Appendix D.2): allocation of a fixed number of intervals.

A R sweep at K = 21 in Qwen2.5-1.5B; B unweighted and probability-weighted RMS log-weight error vs dNLL relative
to R = 1; C D_R(21) in ten models.  Data: results/characterization/<model>/T04 and condition_blocks."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
qw = DEEP_REF
roster = model_roster(); DISPLAY = dict(zip(roster.alias, roster.display)); PAR = roster.set_index("alias").parameter_count

AX = v13_axes("fig3_allocation")
axA, axB, axC, axD, axE = AX["A"], AX["B"], AX["C"], AX["D"], AX["E"]

# A: attention-mass concentration (T04 mass_concentration)
mcp = RESULTS_FORMAL / qw / "T04" / "mass_concentration.csv"; mc = pd.read_csv(mcp)
x = np.arange(len(mc)); w = 0.38
axA.bar(x - w / 2, mc.entry_fraction * 100, w, color="white", edgecolor="k", hatch="////", label="entries")
axA.bar(x + w / 2, mc.mass_fraction * 100, w, color=C["orange"], edgecolor="k", label="attention mass")
axA.set_xticks(x); axA.set_xticklabels([str(r) for r in mc.rule], fontsize=6, rotation=60, ha="right")  # 6 pt floor
axA.set_ylabel("%"); axA.set_ylim(0, 135); axA.legend(loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2); panel_label(axA, "A")  # legend inside the free band above the 100% bars  # legend above the axes
for _, r in mc.iterrows():
    rec("fig3 panelA bar", f"{r.entry_fraction*100:.2f}%", r.entry_fraction * 100, "direct x100", mcp, f"row: rule={r.rule}; col: entry_fraction", "{:.2f}", dict(kind="direct", path=str(mcp), row_filter={"rule": r.rule}, col="entry_fraction", scale=100.0), rid=f"fig3A_entry_{r.rule}")
    rec("fig3 panelA bar", f"{r.mass_fraction*100:.2f}%", r.mass_fraction * 100, "direct x100", mcp, f"row: rule={r.rule}; col: mass_fraction", "{:.2f}", dict(kind="direct", path=str(mcp), row_filter={"rule": r.rule}, col="mass_fraction", scale=100.0), rid=f"fig3A_mass_{r.rule}")

# B: controlled R sweep at K=21 (deep reference)
Rs = ["0.25", "0.5", "1", "2", "4", "8"]; est, lo, hi = [], [], []
for R in Rs:
    c = f"alloc_K21_R{R}"; xv, paths = paired_delta(qw, c); e, l, h = boot_mean_ci(xv); est.append(e); lo.append(l); hi.append(h)
    rec("fig3 panelB point", SIG3.format(e), e, "paired block bootstrap mean delta_nll", paths, "col: delta_nll", SIG3, dict(kind="boot", alias=qw, cond_a=c, cond_b=None, L=2048, root_a="formal", root_b="formal", which="est"), rid=f"fig3B_{c}")
est, lo, hi = map(np.array, (est, lo, hi)); Rf = [float(r) for r in Rs]
axB.errorbar(Rf, est, yerr=[est - lo, hi - est], fmt="o-", color=C["blue"], ecolor="k", ms=3.5, capsize=1.5, lw=0.6, elinewidth=0.6)
axB.set_xscale("log", base=2); axB.set_xlabel("ratio $R$"); axB.set_ylabel("$\\Delta$NLL"); panel_label(axB, "B")
axB.set_xticks(Rf); axB.set_xticklabels(["¼", "", "1", "2", "4", "8"])  # compact tick labels

# C: error-geometry negative control (normalised to R=1)
egp = RESULTS_FORMAL / qw / "T04" / "error_geometry.csv"; eg = pd.read_csv(egp).set_index("condition")
g = np.array([eg.loc[f"alloc_K21_R{R}", "rms_eps"] for R in Rs]); p = np.array([eg.loc[f"alloc_K21_R{R}", "p_rms_eps"] for R in Rs])
ref = Rs.index("1")
axC.plot(Rf, g / g[ref], "s--", color=C["grey"], mfc="white", label="unweighted")
axC.plot(Rf, p / p[ref], "^-", color=C["purple"], label="$p$-weighted")
axC.plot(Rf, est / est[ref], "o-", color=C["blue"], label="$\\Delta$NLL")  # filled, as in panel A; hollow is reserved for the crossing intervals of panel C
axC.axhline(1, color="k", lw=0.5, ls=":"); axC.set_xscale("log", base=2); axC.set_xticks(Rf); axC.set_xticklabels(["¼", "", "1", "2", "4", "8"])
axC.set_xlabel("ratio $R$"); axC.set_ylabel("relative to $R{=}1$")
# direct line labels at the right end instead of a legend (the panel is too narrow for a legend at 7 pt)
# colour-coded in-panel labels placed in empty regions (the panel is too narrow for a legend at 8 pt)
axC.text(0.26, 0.35, "$\\Delta$NLL", color=C["blue"], ha="left", va="center")
axC.text(1.2, 2.95, "unweighted\nRMS", color=C["grey"], ha="left", va="center")
axC.text(1.2, 2.05, "$p$-weighted\nRMS", color=C["purple"], ha="left", va="center")
panel_label(axC, "C")
for R in Rs:
    c = f"alloc_K21_R{R}"
    rec("fig3 panelC point", f"{eg.loc[c,'rms_eps']:.4f}", eg.loc[c, "rms_eps"], "direct", egp, f"row: condition={c}; col: rms_eps", "{:.4f}", dict(kind="direct", path=str(egp), row_filter={"condition": c}, col="rms_eps", scale=1.0), rid=f"fig3C_rms_{c}")
    rec("fig3 panelC point", f"{eg.loc[c,'p_rms_eps']:.4f}", eg.loc[c, "p_rms_eps"], "direct", egp, f"row: condition={c}; col: p_rms_eps", "{:.4f}", dict(kind="direct", path=str(egp), row_filter={"condition": c}, col="p_rms_eps", scale=1.0), rid=f"fig3C_prms_{c}")

# D: ten-model D_R(21) with 95% paired intervals
rows = []
for a in ALL_MODELS:
    xv, paths = paired_delta(a, "alloc_K21_R1", "alloc_K21_R4"); e, l, h = boot_mean_ci(xv)
    rows.append(dict(alias=a, est=e, lo=l, hi=h, params=PAR[a]))
    for which, v in [("est", e), ("lo", l), ("hi", h)]:
        rec("fig3 panelD D_R(21)", SIG3P.format(v), v, {"est": "paired block bootstrap mean", "lo": "bootstrap 2.5%", "hi": "bootstrap 97.5%"}[which], paths, "col: delta_nll (R1 minus R4)", "{:+.5f}",
            dict(kind="boot", alias=a, cond_a="alloc_K21_R1", cond_b="alloc_K21_R4", L=2048, root_a="formal", root_b="formal", which=which), rid=f"fig3D_{a}_{which}")
d = pd.DataFrame(rows).sort_values("params"); xp = np.arange(len(d))
unres = (d.lo <= 0).to_numpy()
# horizontal layout (models on the y axis) so that model names fit at 7 pt; values unchanged
# every model uses the same colour and marker shape; only the fill distinguishes an
# interval that excludes zero from one that crosses it, so no model looks like a different series.
axD.errorbar(d.est[~unres], xp[~unres], xerr=[d.est[~unres] - d.lo[~unres], d.hi[~unres] - d.est[~unres]], fmt="o", ecolor="k", color=C["blue"], capsize=1.5, lw=0.6, elinewidth=0.6, ms=4, label="CI excludes 0")
axD.errorbar(d.est[unres], xp[unres], xerr=[d.est[unres] - d.lo[unres], d.hi[unres] - d.est[unres]], fmt="o", ecolor="k", color=C["blue"], mfc="white", capsize=1.5, lw=0.6, elinewidth=0.6, ms=4, label="CI crosses 0")
axD.axvline(0, color="k", lw=0.6); axD.set_yticks(xp); axD.set_yticklabels([DISPLAY[a] for a in d.alias]); axD.invert_yaxis()
# an extra tick at 0.005; labelled in units of 1e-3 so the three labels do not collide
axD.set_xlabel("$D_R(21)$ ($\\times10^{-3}$)"); axD.set_xticks([0, 0.005, 0.01]); axD.set_xticklabels(["0", "5", "10"])
# the legend belongs under the panel title, not floating above the figure
axD.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8, labelspacing=0.15, borderaxespad=0.1, borderpad=0.15, fontsize=6.5)
panel_label(axD, "D")
axD.set_title(axD.get_title(loc="left"), loc="left", pad=14)  # room for the legend under the title  # figure-level legend above panel C (the panel is too narrow for an inside legend)

# E: precision x allocation interaction
rows = []
for a in ALL_MODELS:
    if not (block_path(a, "alloc_K16_R1").exists() and block_path(a, "alloc_K32_R1").exists()): continue
    x16, p16 = paired_delta(a, "alloc_K16_R1", "alloc_K16_R4"); x32, p32 = paired_delta(a, "alloc_K32_R1", "alloc_K32_R4")
    e16, l16, h16 = boot_mean_ci(x16); e32, l32, h32 = boot_mean_ci(x32); ei, li, hi_ = boot_mean_ci(x16 - x32)
    rows.append(dict(alias=a, e16=e16, l16=l16, h16=h16, e32=e32, l32=l32, h32=h32, ei=ei, li=li, hi=hi_, params=PAR[a]))
    rec("fig3 panelE D_R(16)", SIG3P.format(e16), e16, "paired block bootstrap mean", p16, "col: delta_nll", SIG3P, dict(kind="boot", alias=a, cond_a="alloc_K16_R1", cond_b="alloc_K16_R4", L=2048, root_a="formal", root_b="formal", which="est"), rid=f"fig3E_{a}_DR16")
    rec("fig3 panelE D_R(32)", SIG3P.format(e32), e32, "paired block bootstrap mean", p32, "col: delta_nll", SIG3P, dict(kind="boot", alias=a, cond_a="alloc_K32_R1", cond_b="alloc_K32_R4", L=2048, root_a="formal", root_b="formal", which="est"), rid=f"fig3E_{a}_DR32")
    for which, v in [("est", ei), ("lo", li), ("hi", hi_)]:
        rec("fig3 panelE I_R", SIG3P.format(v), v, "boot mean of D_R(16)-D_R(32) (paired blocks)", p16 + p32, "col: delta_nll", SIG3P, dict(kind="interaction", alias=a, which=which), rid=f"fig3E_{a}_IR_{which}")
d = pd.DataFrame(rows).sort_values("params"); xp = np.arange(len(d)); w = 0.36
axE.bar(xp - w / 2, d.e16, w, color=C["orange"], edgecolor="k", lw=0.5, label="$D_R(16)$")
axE.bar(xp + w / 2, d.e32, w, color="white", edgecolor="k", lw=0.5, hatch="///", label="$D_R(32)$")
axE.errorbar(xp - w / 2, d.e16, yerr=[d.e16 - d.l16, d.h16 - d.e16], fmt="none", ecolor="k", capsize=1.5, lw=0.7, elinewidth=0.6)
axE.errorbar(xp + w / 2, d.e32, yerr=[d.e32 - d.l32, d.h32 - d.e32], fmt="none", ecolor="k", capsize=1.5, lw=0.7, elinewidth=0.6)
for i, r in enumerate(d.itertuples()):
    axE.text(i, max(r.h16, r.h32) * 1.05, "$I_R$ CI>0" if r.li > 0 else "$I_R$ unresolved", ha="center", fontsize=6, va="bottom")  # horizontal, 6 pt
axE.set_xticks(xp); axE.set_xticklabels([DISPLAY[a] for a in d.alias], rotation=35, ha="right")
axE.set_ylabel("$D_R(K)$"); axE.set_ylim(0, d[["h16", "h32"]].max().max() * 1.9); axE.legend(loc="upper right"); panel_label(axE, "E")
v13_save()
flush_manifest("fig3_allocation")
