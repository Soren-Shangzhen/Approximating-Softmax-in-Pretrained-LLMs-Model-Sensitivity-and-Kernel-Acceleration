#!/usr/bin/env python3
"""Figure S6 (Appendix D.4): similar probability redistribution, model-dependent losses (ten models).

A signed leakage L_0.9 of matched flattening and sharpening at J0; B local pV RMS change; C original D_T(J0) and
the contrast after two-region mass restoration (paired intervals); D L_0.9 at K = 21 for R = 1 and R = 4.
Data: results/characterization/<model>/T10 and condition_blocks."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
roster = model_roster(); DISPLAY = dict(zip(roster.alias, roster.display)); PAR = roster.set_index("alias").parameter_count
order = sorted(ALL_MODELS, key=lambda a: PAR[a]); xp = np.arange(len(order))

AX = v13_axes("fig5_geometry")
axA, axB, axC, axD = AX["A"], AX["B"], AX["C"], AX["D"]
probe = {a: pd.read_csv(RESULTS_FORMAL / a / "T10" / "probe_summary.csv").set_index("condition") for a in ALL_MODELS}
def ppath(a): return RESULTS_FORMAL / a / "T10" / "probe_summary.csv"

# A: signed leakage q=0.9 at K0
w = 0.38
lm = [probe[a].loc["jsd_K0_Aminus", "q0.9_leakage"] for a in order]; lp = [probe[a].loc["jsd_K0_Aplus", "q0.9_leakage"] for a in order]
axA.bar(xp - w / 2, lm, w, color=C["orange"], edgecolor="k", lw=0.5, label="flattening $A^-$")
axA.bar(xp + w / 2, lp, w, color="white", edgecolor=C["purple"], hatch="///", lw=0.7, label="sharpening $A^+$")
axA.axhline(0, color="k", lw=0.6); axA.set_xticks(xp); axA.tick_params(labelbottom=False)  # sharex, model names on the bottom panel only
axA.set_ylabel("leakage $L_{0.9}$"); axA.set_ylim(min(lp) * 1.15, max(lm) * 1.9); axA.legend(loc="upper left", ncol=2); panel_label(axA, "A")  # headroom for the inside legend
axA.text(0.99, 0.98, "descriptive, no bootstrap interval", transform=axA.transAxes, ha="right", va="top", fontsize=6, color=C["grey"])  # note moved off the A+ bars
for a, m_, p_ in zip(order, lm, lp):
    rec("fig5 panelA leakage", f"{m_:+.4f}", m_, "direct (descriptive)", ppath(a), "row: condition=jsd_K0_Aminus; col: q0.9_leakage", "{:+.4f}", dict(kind="direct", path=str(ppath(a)), row_filter={"condition": "jsd_K0_Aminus"}, col="q0.9_leakage", scale=1.0), rid=f"fig5A_{a}_Aminus")
    rec("fig5 panelA leakage", f"{p_:+.4f}", p_, "direct (descriptive)", ppath(a), "row: condition=jsd_K0_Aplus; col: q0.9_leakage", "{:+.4f}", dict(kind="direct", path=str(ppath(a)), row_filter={"condition": "jsd_K0_Aplus"}, col="q0.9_leakage", scale=1.0), rid=f"fig5A_{a}_Aplus")

# B: local pV RMS (used) negative control, K0
rm = [probe[a].loc["jsd_K0_Aminus", "output_rms_used"] for a in order]; rp = [probe[a].loc["jsd_K0_Aplus", "output_rms_used"] for a in order]
axB.bar(xp - w / 2, rm, w, color=C["orange"], edgecolor="k", lw=0.5, label="$A^-$"); axB.bar(xp + w / 2, rp, w, color="white", edgecolor=C["purple"], hatch="///", lw=0.7, label="$A^+$")
# annotate sign of D_T (recomputed) above bars
for i, a in enumerate(order):
    x, _ = paired_delta(a, "jsd_K0_Aminus", "jsd_K0_Aplus"); e, l, h = boot_mean_ci(x)
    tag = "+" if l > 0 else ("−" if h < 0 else "0")
    axB.text(i, max(rm[i], rp[i]) * 1.09, f"$D_T$ {tag}", ha="center", fontsize=6)
    rec("fig5 panelB pV rms", f"{rm[i]:.4f}", rm[i], "direct (descriptive)", ppath(a), "row: condition=jsd_K0_Aminus; col: output_rms_used", "{:.4f}", dict(kind="direct", path=str(ppath(a)), row_filter={"condition": "jsd_K0_Aminus"}, col="output_rms_used", scale=1.0), rid=f"fig5B_{a}_Aminus")
    rec("fig5 panelB pV rms", f"{rp[i]:.4f}", rp[i], "direct (descriptive)", ppath(a), "row: condition=jsd_K0_Aplus; col: output_rms_used", "{:.4f}", dict(kind="direct", path=str(ppath(a)), row_filter={"condition": "jsd_K0_Aplus"}, col="output_rms_used", scale=1.0), rid=f"fig5B_{a}_Aplus")
axB.set_xticks(xp); axB.tick_params(labelbottom=False)
axB.set_ylabel("$pV$ RMS change"); axB.set_ylim(0, max(max(rm), max(rp)) * 1.45); axB.legend(loc="upper left", ncol=2); panel_label(axB, "B")
# "sign of matched D_T shown above bars" note removed from the panel (stated in the caption)

# C: mass-partition rescue at K0: original vs rescued directional gap
rows = []
for a in order:
    xo, po = paired_delta(a, "jsd_K0_Aminus", "jsd_K0_Aplus"); eo, lo_, ho = boot_mean_ci(xo)
    xr, pr = paired_delta(a, "rescue_K0_Aminus_q0p9", "rescue_K0_Aplus_q0p9"); er, lr, hr = boot_mean_ci(xr)
    rows.append(dict(alias=a, eo=eo, lo=lo_, ho=ho, er=er, lr=lr, hr=hr))
    for which, v in [("est", er), ("lo", lr), ("hi", hr)]:
        rec("fig5 panelC rescued gap", SIG3P.format(v), v, {"est": "paired block bootstrap mean (rescued A- minus rescued A+)", "lo": "bootstrap 2.5%", "hi": "bootstrap 97.5%"}[which], pr, "col: delta_nll", "{:+.5f}",
            dict(kind="boot", alias=a, cond_a="rescue_K0_Aminus_q0p9", cond_b="rescue_K0_Aplus_q0p9", L=2048, root_a="formal", root_b="formal", which=which), rid=f"fig5C_{a}_rescued_{which}")
d = pd.DataFrame(rows)
axC.errorbar(xp - 0.15, d.eo, yerr=[d.eo - d.lo, d.ho - d.eo], fmt="o", ecolor="k", color=C["blue"], capsize=1.5, ms=4, lw=0.6, label="original $D_T(J_0)$", elinewidth=0.6)  # elinewidth was inside the label string
axC.errorbar(xp + 0.15, d.er, yerr=[d.er - d.lr, d.hr - d.er], fmt="s", ecolor="k", color=C["green"], mfc="white", capsize=1.5, ms=4, lw=0.6, label="after two-region mass restoration", elinewidth=0.6)
axC.axhline(0, color="k", lw=0.6); axC.set_yscale("symlog", linthresh=2e-3)
axC.set_xticks(xp); axC.tick_params(labelbottom=False)
axC.set_ylabel("contrast (symlog)"); axC.legend(loc="upper left", ncol=2); panel_label(axC, "C")

# D: allocation-linked leakage negative control
l1 = [probe[a].loc["alloc_K21_R1", "q0.9_leakage"] for a in order]; l4 = [probe[a].loc["alloc_K21_R4", "q0.9_leakage"] for a in order]
axD.bar(xp - w / 2, l1, w, color="white", edgecolor="k", hatch="....", lw=0.6, label="$K{=}21, R{=}1$")
axD.bar(xp + w / 2, l4, w, color=C["sky"], edgecolor="k", lw=0.5, label="$K{=}21, R{=}4$")
axD.axhline(0, color="k", lw=0.6); axD.set_xticks(xp); axD.set_xticklabels([DISPLAY[a] for a in order], rotation=45, ha="right", fontsize=6)
axD.set_ylabel("leakage $L_{0.9}$"); axD.set_ylim(0, max(max(l1), max(l4)) * 1.5); axD.legend(loc="upper left", ncol=2); panel_label(axD, "D")  # legend above the axes
axD.text(0.99, 0.98, "descriptive, no bootstrap interval", transform=axD.transAxes, ha="right", va="top", fontsize=6, color=C["grey"])
for a, v1, v4 in zip(order, l1, l4):
    rec("fig5 panelD leakage", f"{v1:+.4f}", v1, "direct (descriptive)", ppath(a), "row: condition=alloc_K21_R1; col: q0.9_leakage", "{:+.4f}", dict(kind="direct", path=str(ppath(a)), row_filter={"condition": "alloc_K21_R1"}, col="q0.9_leakage", scale=1.0), rid=f"fig5D_{a}_R1")
    rec("fig5 panelD leakage", f"{v4:+.4f}", v4, "direct (descriptive)", ppath(a), "row: condition=alloc_K21_R4; col: q0.9_leakage", "{:+.4f}", dict(kind="direct", path=str(ppath(a)), row_filter={"condition": "alloc_K21_R4"}, col="q0.9_leakage", scale=1.0), rid=f"fig5D_{a}_R4")
v13_save(); flush_manifest("fig5_geometry")
