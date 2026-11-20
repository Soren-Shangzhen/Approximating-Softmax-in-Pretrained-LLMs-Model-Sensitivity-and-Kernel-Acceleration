#!/usr/bin/env python3
"""Figure 4 and Figure S2 (Section 5, Appendix B.4): lattice controls, Rowmax-H15 fidelity, B200 performance.

Figure 4: A anchor x resolution on Qwen2.5-1.5B (8-bit absolute probability grids vs row-maximum score lattices);
B perplexity increase at 2K (BF16 kernel measured in five models; BF16 and FP8 semantic simulations);
C attention-forward speed-up over stock FA4 at 8K by dtype and mask; D FA4 emulation settings vs Rowmax-H15.
Figure S2: A FP8 causal sequence-length sweep (three alternating rounds and the 1K/2K retest);
B Nsight Systems run: call latency and kernel-only speed-up.
Data: results/lattice_operators/qwen15, results/characterization/qwen15, results/kernel/{fidelity,timing,nsys}."""
import sys, math, re; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
qw = DEEP_REF

AX = v13_axes("fig6_h15_b200")
axA, axB, axC, axD, axE, axF = AX["A"], AX["B"], AX["C"], AX["D"], AX["E"], AX["F"]

# ================= A: anchor x resolution =================
cells = {("octave", "absolute"): ("pq_log2_b8", "lattice"), ("octave", "rowmax"): ("pot_m1", "formal"),
         ("half-octave", "absolute"): ("pq_logsqrt2_b8", "lattice"), ("half-octave", "rowmax"): ("response_pot_m2", "formal")}
V = {}
for (lat, anc), (c, root) in cells.items():
    x, paths = paired_delta(qw, c, None, 2048, root); e, l, h = boot_mean_ci(x); V[(lat, anc)] = (e, l, h, x, paths)
    for which, v in [("est", e), ("lo", l), ("hi", h)]:
        rec("fig6 panelA bar", SIG3.format(v), v, {"est": "paired block bootstrap mean delta_nll", "lo": "bootstrap 2.5%", "hi": "bootstrap 97.5%"}[which], paths, f"col: delta_nll ({lat}, {anc} anchor = {c})", "{:.5f}",
            dict(kind="boot", alias=qw, cond_a=c, cond_b=None, L=2048, root_a=root, root_b=root, which=which), rid=f"fig6A_{lat}_{anc}_{which}")
xg = {"octave": 0.0, "half-octave": 1.0}; w = 0.36
for anc, off, fc, hatch, lab in [("absolute", -w / 2, "white", "////", "absolute anchor"), ("rowmax", w / 2, C["blue"], None, "rowmax anchor")]:
    xs = [xg[l] + off for l in xg]; es = [V[(l, anc)][0] for l in xg]; los = [V[(l, anc)][1] for l in xg]; his = [V[(l, anc)][2] for l in xg]
    axA.bar(xs, es, w, color=fc, edgecolor="k", hatch=hatch, lw=0.7, label=lab)
    axA.errorbar(xs, es, yerr=[np.array(es) - los, np.array(his) - es], fmt="none", ecolor="k", capsize=1.5, lw=0.6, elinewidth=0.6)
# absolute drops (computed): anchor change at octave; spacing change at absolute anchor
d_anchor = V[("octave", "absolute")][0] - V[("octave", "rowmax")][0]
d_res = V[("octave", "absolute")][0] - V[("half-octave", "absolute")][0]
xa, xb = xg["octave"] - w / 2, xg["octave"] + w / 2
axA.annotate("", xy=(xb, V[("octave", "rowmax")][0]), xytext=(xa, V[("octave", "absolute")][0]), arrowprops=dict(arrowstyle="->", lw=0.8, color=C["vermil"]))
axA.text(xg["octave"] + w * 0.15, (V[("octave", "absolute")][0] + V[("octave", "rowmax")][0]) / 2, f"$-${fmt_sig(d_anchor)}", color=C["vermil"], va="center", ha="left")  # labels in the free band above the bars
# a short bracket instead of the long arc -- dashed reference at the octave/absolute level, then
# a vertical double arrow down to the half-octave/absolute bar
_y0 = V[("octave", "absolute")][0]; _xh = xg["half-octave"] - w / 2
axA.plot([xa, _xh], [_y0, _y0], ls=(0, (3, 2)), lw=0.7, color=C["green"], zorder=1)
axA.annotate("", xy=(_xh, V[("half-octave", "absolute")][0]), xytext=(_xh, _y0), arrowprops=dict(arrowstyle="->", lw=0.8, color=C["green"]))
axA.text(_xh - 0.06, _y0 * 1.04, f"$-${fmt_sig(d_res)}", color=C["green"], ha="right", va="bottom")
rec("fig6 panelA annotation", "-" + fmt_sig(d_anchor), -d_anchor, "mean delta_nll(pq_log2_b8) - mean delta_nll(pot_m1), paired blocks", [V[("octave", "absolute")][4][0], V[("octave", "rowmax")][4][0]], "col: delta_nll", "{:.5f}", dict(kind="pair_diff_neg", a=("pq_log2_b8", "lattice"), b=("pot_m1", "formal")), rid="fig6A_anchor_drop")
rec("fig6 panelA annotation", "-" + fmt_sig(d_res), -d_res, "mean delta_nll(pq_log2_b8) - mean delta_nll(pq_logsqrt2_b8), paired blocks", [V[("octave", "absolute")][4][0], V[("half-octave", "absolute")][4][0]], "col: delta_nll", "{:.5f}", dict(kind="pair_diff_neg", a=("pq_log2_b8", "lattice"), b=("pq_logsqrt2_b8", "lattice")), rid="fig6A_spacing_drop")
axA.set_xticks(list(xg.values())); axA.set_xticklabels(["octave", "half-octave"]); axA.set_ylabel("$\\Delta$NLL")
axA.set_ylim(0, V[("octave", "absolute")][2] * 1.30)
axA.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1, borderpad=0.15)
panel_label(axA, "A")
axA.set_title(axA.get_title(loc="left"), loc="left", pad=14)  # legend inside the free upper-right band; annotations moved apart

# ================= B: five true-kernel models + two Qwen2.5-1.5B simulations =================
# one unit for the whole panel.  Every point is the perplexity increase
# 100*(exp(dNLL) - 1), computed from the unrounded bootstrap mean (never dNLL x 100), so the axis and
# the reader's question ("how much worse?") use the same scale and no per-point annotation is needed.
# Intervals live in Table 2.  The true-kernel group is five models from three families, ordered by
# parameter count; the simulation group is Qwen2.5-1.5B only, past a dashed rule.  Models are not pooled.
# ordered by parameter count (1.5B, 7.2B, 8.0B, 70.6B, 72.7B), so the two largest sit together
S5 = [("Q1.5B", "ppl_A0.csv", "ppl_A1.csv"), ("M7B", "ppl_mistral7_A0.csv", "ppl_mistral7_A1.csv"),
      ("L8B", "ppl_llama8_A0.csv", "ppl_llama8_A1.csv"), ("L70B", "ppl_llama70_A0.csv", "ppl_llama70_A1.csv"),
      ("Q72B", "ppl_qwen72_A0.csv", "ppl_qwen72_A1.csv")]
pts = []
for _lab, _f0, _f1 in S5:
    _d0 = pd.read_csv(B200_PPL / _f0).sort_values("block"); _d1 = pd.read_csv(B200_PPL / _f1).sort_values("block")
    _e, _l, _h = boot_mean_ci((_d1.nll - _d0.nll).to_numpy())
    pts.append((_lab, _e, True, [B200_PPL / _f0, B200_PPL / _f1], dict(kind="b200_ppl", f0=_f0, f1=_f1)))
xs_, ps_ = paired_delta(qw, "h15tiled_bf16", None, 2048, "lattice"); es_, _, _ = boot_mean_ci(xs_)
xf_, pf_ = paired_delta(qw, "h15tiled", None, 2048, "lattice"); ef_, _, _ = boot_mean_ci(xf_)
pts += [("BF16", es_, False, ps_,
         dict(kind="boot", alias=qw, cond_a="h15tiled_bf16", cond_b=None, L=2048, root_a="lattice", root_b="lattice")),
        ("FP8", ef_, False, pf_,
         dict(kind="boot", alias=qw, cond_a="h15tiled", cond_b=None, L=2048, root_a="lattice", root_b="lattice"))]
_bx = [0, 1, 2, 3, 4, 5.5, 6.5]  # a gap separates the true-kernel group from the two simulations
for i, (lab, e, filled, paths, spec) in enumerate(pts):
    _pct = (math.exp(e) - 1) * 100
    axB.plot([_bx[i]], [_pct], "o", color=C["black"] if filled else C["blue"],
             mfc=(C["black"] if filled else "white"), ms=5, mew=1.1, ls="none")
    rec("fig6 panelB point", f"{_pct:+.3f}%", _pct, "100*(exp(paired block bootstrap mean)-1), 97 blocks",
        paths, "col: nll/delta_nll paired on block", "{:+.3f}", {**spec, "which": "ppl_pct"}, rid=f"fig6B_{i}_pplpct")
axB.plot([], [], "o", color="k", ms=5, label="BF16 kernel, measured")
axB.plot([], [], "o", color=C["blue"], mfc="white", ms=5, label="Q1.5B sim.")
axB.axvline(4.9, color="0.6", lw=0.6, ls=(0, (2, 2)))  # true-kernel group | simulation group
axB.set_xticks(_bx); axB.set_xticklabels([q[0] for q in pts], ha="center")
axB.set_ylabel("PPL increase (%)")
axB.set_yticks([0, 0.2, 0.4, 0.6]); axB.set_ylim(0, 0.62); axB.set_xlim(-0.7, _bx[-1] + 0.7)
axB.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1, borderpad=0.15)
panel_label(axB, "B")
axB.set_title(axB.get_title(loc="left"), loc="left", pad=14)  # legend above the axes, not in the data area

# ================= C: FP8 causal speedup vs sequence length =================
cfg0 = dict(dtype="fp8_e4m3", heads=16, hd=128, batch=1, causal=1)
Ls = [1024, 2048, 4096, 8192, 16384, 32768]
pooled, per_round = [], {}
for L in Ls:
    A0 = pd.concat([load_lat(f"S_seqsweep_r{i}_A0") for i in (1, 2, 3)]); A1 = pd.concat([load_lat(f"S_seqsweep_r{i}_A1") for i in (1, 2, 3)])
    r = speedup_pct(A0, A1, seqlen=L, **cfg0); pooled.append(r["speedup_pct"])
    per_round[L] = [speedup_pct(load_lat(f"S_seqsweep_r{i}_A0"), load_lat(f"S_seqsweep_r{i}_A1"), seqlen=L, **cfg0)["speedup_pct"] for i in (1, 2, 3)]
    pairs = [(f"S_seqsweep_r{i}_A0", f"S_seqsweep_r{i}_A1") for i in (1, 2, 3)]
    rec("fig6 panelC pooled", f"{r['speedup_pct']:+.1f}%", r["speedup_pct"], "100*(median A0 over 3 pooled rounds / median A1 - 1)", [B200_EXT / f"{n}.csv" for pr in pairs for n in pr], f"seqlen={L}; col: latency_ms", "{:+.1f}", dict(kind="multi_round", pairs=pairs, cfg=dict(seqlen=L, **cfg0), sub="ext", which="pooled"), rid=f"fig6C_pooled_{L}")
    for i, v in enumerate(per_round[L], 1):
        rec("fig6 panelC round", f"{v:+.1f}%", v, "100*(median A0/median A1-1) within one ABBA round", [B200_EXT / f"S_seqsweep_r{i}_A0.csv", B200_EXT / f"S_seqsweep_r{i}_A1.csv"], f"seqlen={L}; col: latency_ms", "{:+.1f}", dict(kind="speedup", a0=f"S_seqsweep_r{i}_A0", a1=f"S_seqsweep_r{i}_A1", sub="ext", cfg=dict(seqlen=L, **cfg0)), rid=f"fig6C_r{i}_{L}")
short = {L: speedup_pct(load_lat("S_short_A0"), load_lat("S_short_A1"), seqlen=L, **cfg0)["speedup_pct"] for L in (1024, 2048)}
for L, v in short.items():
    rec("fig6 panelC short retest", f"{v:+.1f}%", v, "100*(median A0/median A1-1), n=9 per arm", [B200_EXT / "S_short_A0.csv", B200_EXT / "S_short_A1.csv"], f"seqlen={L}; col: latency_ms", "{:+.1f}", dict(kind="speedup", a0="S_short_A0", a1="S_short_A1", sub="ext", cfg=dict(seqlen=L, **cfg0)), rid=f"fig6C_short_{L}")
lx = np.log2(Ls)
# read from results/kernel/nsys/kernel_vs_wall.csv (never hard-coded); both are ratios of the same run
KVW = B200_RES / "nsys" / "kernel_vs_wall.csv"; kvw = pd.read_csv(KVW); kvw = kvw[kvw.causal == 1]
ko, wc3 = [], []
for L in Ls:
    a0 = kvw[(kvw.seqlen == L) & (kvw.arm == "A0")].iloc[0]; a1 = kvw[(kvw.seqlen == L) & (kvw.arm == "A1")].iloc[0]
    ko.append(100 * (a0.kern_med_ns / a1.kern_med_ns - 1)); wc3.append(100 * (a0.wall_per_iter_us / a1.wall_per_iter_us - 1))
    rec("fig6 panelC kernel-only (traced run)", f"{ko[-1]:+.1f}%", ko[-1], "100*(median kernel A0 / median kernel A1 - 1), nsys cuda_gpu_kern_sum", KVW, f"seqlen={L}, causal=1; col: kern_med_ns", "{:+.1f}",
        dict(kind="csv_expr", path=str(KVW), cells={"a0": dict(filter={"seqlen": L, "arm": "A0", "causal": 1}, col="kern_med_ns"), "a1": dict(filter={"seqlen": L, "arm": "A1", "causal": 1}, col="kern_med_ns")}, expr="100*(a0/a1-1)"), rid=f"fig6C_s3_kernel_{L}")
    rec("fig6 panelC wall-clock (traced run)", f"{wc3[-1]:+.1f}%", wc3[-1], "100*(wall A0 / wall A1 - 1), 200 host-timed iterations of the same nsys run", KVW, f"seqlen={L}, causal=1; col: wall_per_iter_us", "{:+.1f}",
        dict(kind="csv_expr", path=str(KVW), cells={"a0": dict(filter={"seqlen": L, "arm": "A0", "causal": 1}, col="wall_per_iter_us"), "a1": dict(filter={"seqlen": L, "arm": "A1", "causal": 1}, col="wall_per_iter_us")}, expr="100*(a0/a1-1)"), rid=f"fig6C_s3_wall_{L}")
# the sequence-length evidence is an appendix figure of two panels that share the length axis.
# separate higher-precision measurement and is never merged into the pooled curve.
axC.plot(lx, pooled, "o-", color=C["blue"], ms=4.5, zorder=3, label="pooled (3 ABBA rounds)")
axC.plot([np.log2(1024), np.log2(2048)], [short[1024], short[2048]], "^", color=C["vermil"], mfc="white", ms=6, ls="none", zorder=4, label="retest (1K, 2K only)")
for L in Ls: axC.plot([np.log2(L)] * 3, per_round[L], ".", color=C["sky"], ms=3.8, zorder=5, label="single round" if L == Ls[0] else None)
axC.axhline(0, color="k", lw=0.7); axC.set_xticks(lx); axC.set_xticklabels(["1K", "2K", "4K", "8K", "16K", "32K"])
sd4 = np.std(per_round[4096], ddof=SD_DDOF); sd8 = np.std(per_round[8192], ddof=SD_DDOF)
axC.text(np.log2(4096) - 0.12, max(per_round[4096]) + 0.6, f"sd {sd4:.2f}", ha="right", va="bottom"); axC.text(np.log2(8192), max(per_round[8192]) + 1.6, f"sd {sd8:.2f}", ha="center")  # 4K label left of the points, clear of the rising line
axC.set_xlabel("sequence length"); axC.set_ylabel("speed-up (%)")
axC.set_ylim(min(min(v) for v in per_round.values()) - 2.0, max(max(pooled), max(ko)) + 4.5)
axC.legend(loc="lower right", ncol=1, handlelength=1.0, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.2, borderpad=0.25)
panel_label(axC, "C")

# difference is drawn: the two are ratios, and their difference is in percentage points, not a time.
xb = np.arange(len(Ls)); wb = 0.38
axF.bar(xb - wb / 2, wc3, wb, color=C["blue"], edgecolor="k", lw=0.5, label="call latency")
axF.bar(xb + wb / 2, ko, wb, color=C["orange"], edgecolor="k", lw=0.5, label="kernel-only")
for x, v, v_other, right in [(x, v, o, False) for x, v, o in zip(xb - wb / 2, wc3, ko)] + [(x, v, o, True) for x, v, o in zip(xb + wb / 2, ko, wc3)]:
    # horizontal 6 pt labels; when the pair's labels would touch, the kernel-only label is lifted one text height
    lift = 3.4 if (right and v >= 0 and abs(v - v_other) < 3.2) else 0.0
    axF.text(x, v + (0.6 + lift if v >= 0 else -0.6), f"{v:.1f}".replace("-", "\u2212"), ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
axF.axhline(0, color="k", lw=0.7); axF.set_xticks(xb); axF.set_xticklabels(["1K", "2K", "4K", "8K", "16K", "32K"])
axF.set_xlim(-0.6, len(Ls) - 0.4); axF.set_ylim(min(wc3) - 10.5, max(max(ko), max(wc3)) + 22.0)
axF.set_xlabel("sequence length"); axF.set_ylabel("speed-up (%)")
axF.legend(loc="upper left", ncol=2, handlelength=1.0, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
panel_label(axF, "F")
rec("fig6 panelC sd annotation", f"{sd4:.2f}", sd4, "population sd (ddof=0) over the 3 per-round speed-ups ", [B200_EXT / f"S_seqsweep_r{i}_A{a}.csv" for i in (1, 2, 3) for a in (0, 1)], "seqlen=4096", "{:.2f}", dict(kind="multi_round", pairs=[(f"S_seqsweep_r{i}_A0", f"S_seqsweep_r{i}_A1") for i in (1, 2, 3)], cfg=dict(seqlen=4096, **cfg0), sub="ext", which="sd"), rid="fig6C_sd4k")
rec("fig6 panelC sd annotation", f"{sd8:.2f}", sd8, "population sd (ddof=0) over the 3 per-round speed-ups ", [B200_EXT / f"S_seqsweep_r{i}_A{a}.csv" for i in (1, 2, 3) for a in (0, 1)], "seqlen=8192", "{:.2f}", dict(kind="multi_round", pairs=[(f"S_seqsweep_r{i}_A0", f"S_seqsweep_r{i}_A1") for i in (1, 2, 3)], cfg=dict(seqlen=8192, **cfg0), sub="ext", which="sd"), rid="fig6C_sd8k")
# shorter x label (head dim in caption); legend moved above the axes, clear of the 1K/2K points

# ================= D: dtype x mask at 8K (main-figure panel C) =================
# grouped by dtype, coloured by mask; FP16 non-causal was not measured and is left as a gap
# rather than a zero bar.  Round counts are shown, and the per-round points are drawn only where three
# rounds exist -- no error bar is invented for a single-round configuration.
base = dict(seqlen=8192, heads=16, hd=128, batch=1)
# FP16 causal, FP16 non-causal and BF16 non-causal were raised to the
# same three-round ABBA protocol, so every cell of the ladder now carries 21 windows per arm.
CELLS = [("BF16", "causal", [(f"A_main_r{i}_A0", f"A_main_r{i}_A1") for i in (1, 2, 3)], dict(dtype="bf16", causal=1)),
         ("BF16", "non-causal", [(f"L_bf16_noncausal_r{i}_A0", f"L_bf16_noncausal_r{i}_A1") for i in (1, 2, 3)], dict(dtype="bf16", causal=0)),
         ("FP16", "causal", [(f"J_fp16_causal_r{i}_A0", f"J_fp16_causal_r{i}_A1") for i in (1, 2, 3)], dict(dtype="fp16", causal=1)),
         ("FP16", "non-causal", [(f"K_fp16_noncausal_r{i}_A0", f"K_fp16_noncausal_r{i}_A1") for i in (1, 2, 3)], dict(dtype="fp16", causal=0)),
         ("FP8", "causal", [(f"A_main_r{i}_A0", f"A_main_r{i}_A1") for i in (1, 2, 3)], dict(dtype="fp8_e4m3", causal=1)),
         ("FP8", "non-causal", [(f"C2_noncausal_r{i}_A0", f"C2_noncausal_r{i}_A1") for i in (1, 2, 3)], dict(dtype="fp8_e4m3", causal=0))]
DT = ["BF16", "FP16", "FP8"]; MASKS = [("causal", C["blue"]), ("non-causal", C["sky"])]
wd = 0.34; _rounds = []
for k, (dt, mask, pairs, cfg) in enumerate(CELLS):
    gx = DT.index(dt) + (-wd / 2 if mask == "causal" else wd / 2)
    A0 = pd.concat([load_lat(a) for a, _ in pairs]); A1 = pd.concat([load_lat(b) for _, b in pairs])
    r = speedup_pct(A0, A1, **cfg, **base)
    col = dict(MASKS)[mask]
    axD.bar(gx, r["speedup_pct"], wd, color=col, edgecolor="k", lw=0.6, label=mask if dt == "BF16" else None)
    rr = [speedup_pct(load_lat(a), load_lat(b), **cfg, **base)["speedup_pct"] for a, b in pairs]
    _sd = float(np.std(rr, ddof=SD_DDOF))  # every cell is three rounds, so each bar carries its round-to-round sd
    axD.errorbar([gx], [r["speedup_pct"]], yerr=[_sd], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5, zorder=4)
    axD.plot([gx] * len(rr), rr, ".", color="k", ms=3.0, zorder=5)
    rec("fig6 panelD sd", f"{_sd:.2f}", _sd, "population sd (ddof=0) over the three per-round speed-ups ", [B200_EXT / f"{n}.csv" for pr in pairs for n in pr], "; ".join(f"{kk}={vv}" for kk, vv in {**cfg, **base}.items()), "{:.2f}", dict(kind="multi_round", pairs=pairs, cfg={**cfg, **base}, sub="ext", which="sd"), rid=f"fig6D_sd_{k}")
    axD.text(gx - (0.05 if mask == "causal" else -0.05), r["speedup_pct"] + (2.6 if mask == "non-causal" else 0.8), f"{r['speedup_pct']:.1f}", ha="center", va="bottom")  # non-causal labels raised, else the FP16 and FP8 pair touch
    _rounds.append((gx, len(rr)))
    rec("fig6 panelD bar", f"{r['speedup_pct']:+.1f}%", r["speedup_pct"], "100*(pooled median A0/pooled median A1-1)", [B200_EXT / f"{n}.csv" for pr in pairs for n in pr], "; ".join(f"{kk}={vv}" for kk, vv in {**cfg, **base}.items()), "{:+.1f}", dict(kind="multi_round", pairs=pairs, cfg={**cfg, **base}, sub="ext", which="pooled"), rid=f"fig6D_{k}")
axD.set_xticks(range(len(DT))); axD.set_xticklabels(DT); axD.set_xlim(-0.55, len(DT) - 0.45)
assert {n for _, n in _rounds} == {3}, _rounds  # the ladder is uniform, so the caption states the protocol once
axD.axhline(0, color="k", lw=0.7); axD.set_ylabel("speed-up at 8K (%)"); axD.set_ylim(0, 34)
axD.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.0, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1, borderpad=0.15)
panel_label(axD, "D")
axD.set_title(axD.get_title(loc="left"), loc="left", pad=14)  # legend under the title, clear of the bars

# ================= E: stock-FA4 tuning control (main-figure panel D) =================
# the stock settings sit on a real numeric beta axis (0, 12.5, 25, 37.5, 50, 75), with the two
# pairs that share a beta separated by a small cosmetic offset and identified on a second tick row by
# their emulation period f.  Rowmax-H15 keeps its own region to the right of a rule, sharing the y
# axis: it replaces the primitive instead of emulating more of it, so it is given no beta at all.
# The zero line is the shipping default (freq 8 / start_frg 1, beta = 25%), named in the caption.
def _beta_pct(freq, start_frg, res=4, frg_cnt=4, kmod=16):
    if freq == 0:
        return 0.0
    emu = sum(1 for j in range(frg_cnt) for k in range(kmod)
              if not (k % freq < freq - res or j >= frg_cnt - 1 or j < start_frg))
    return 100.0 * emu / (frg_cnt * kmod)

E_CFG = [("E_freq0", 0, None), ("E_freq16_s1", 16, 1), ("E_freq8_s1", 8, 1), ("E_freq8_s0", 8, 0),
         ("E_freq2_s1", 2, 1), ("E_freq4_s1", 4, 1), ("E_freq2_s0", 2, 0), ("E_freq4_s0", 4, 0)]
stock = float(load_lat("E_freq8_s1").latency_ms.median())
bx, by, bf = [], [], []
for n, f, sf in E_CFG:
    med = float(load_lat(n).latency_ms.median()); v = 100 * (stock / med - 1)
    beta = 0.0 if f == 0 else _beta_pct(f, sf)
    bx.append(beta); by.append(v); bf.append(f)
    rec("fig6 panelE point", f"{v:+.1f}%", v, "100*(median(E_freq8_s1)/median(this config)-1)", [B200_EXT / "E_freq8_s1.csv", B200_EXT / f"{n}.csv"], "col: latency_ms", "{:+.1f}", dict(kind="E_rel", name=n), rid=f"fig6E_{n}")
    rec("fig6 panelE beta", f"{beta:g}%", beta, "emulated fraction from the kernel predicate (res=4, frg_cnt=4)", B200_EXT / f"{n}.csv", "col: freq,start_frg", "{:g}", dict(kind="direct_const", value=beta), rid=f"fig6E_beta_{n}")
h15 = 100 * (stock / float(load_lat("E_h15").latency_ms.median()) - 1)
rec("fig6 panelE H15", f"{h15:+.1f}%", h15, "100*(median(E_freq8_s1)/median(E_h15)-1)", [B200_EXT / "E_freq8_s1.csv", B200_EXT / "E_h15.csv"], "col: latency_ms", "{:+.1f}", dict(kind="E_rel", name="E_h15"), rid="fig6E_h15")
# the four settings that share a beta get four evenly spaced slots of their own, so each is a
# labelled tick instead of two markers crowding one.  Unique betas keep their true coordinate; the four
# slots are drawing positions, and the tick under each gives its beta.
_dup = {b for b in bx if bx.count(b) > 1}
_slots = list(np.linspace(50.0, 82.5, sum(1 for b in bx if b in _dup)))
_px, _k = [], 0
for b in bx:
    if b in _dup:
        _px.append(_slots[_k]); _k += 1
    else:
        _px.append(b)
_hx = 100.0  # Rowmax-H15's column, past the rule; it is not a beta value and carries no beta tick
axE.axhline(0, color="k", lw=0.6, ls=(0, (1, 2)))  # the shipping default
axE.axvline(88.0, color="0.6", lw=0.6, ls=(0, (2, 2)))  # stock beta axis | Rowmax-H15
axE.plot(_px, by, "x", color=C["green"], ms=5, mew=0.9, ls="none", label="stock FA4 settings")
axE.plot([_hx], [h15], "*", color=C["vermil"], ms=10, ls="none", label="Rowmax-H15")
# only the two values the panel is about: the best stock setting measured, and Rowmax-H15
_best = int(np.argmax(by))
axE.annotate(f"{by[_best]:+.1f}", (_px[_best], by[_best]), textcoords="offset points", xytext=(0, 7), ha="center", color=C["green"])
axE.annotate(f"{h15:+.1f}", (_hx, h15), textcoords="offset points", xytext=(7, 0), ha="left", va="center", color=C["vermil"])
_lbl = [f"{b:g}\n " if b not in _dup else f"{b:g}\nf={f}" for b, f in zip(bx, bf)]
axE.set_xticks(_px + [_hx]); axE.set_xticklabels(_lbl + ["H15\n "], linespacing=1.15)
axE.set_ylabel("speed-up vs\ndefault (%)", labelpad=1, linespacing=1.05)
axE.set_xlim(-8, _hx + 26); axE.set_ylim(min(by) - 6, h15 + 12)
axE.set_xlabel("$\\beta$ (%)")
axE.xaxis.label.set_horizontalalignment("center")
axE.xaxis.label.set_x(0.42)  # centred under the stock region: the beta axis does not extend to Rowmax-H15
axE.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1, borderpad=0.15)
panel_label(axE, "E")
axE.set_title(axE.get_title(loc="left"), loc="left", pad=14)

v13_save(); flush_manifest("fig6_h15_b200")
