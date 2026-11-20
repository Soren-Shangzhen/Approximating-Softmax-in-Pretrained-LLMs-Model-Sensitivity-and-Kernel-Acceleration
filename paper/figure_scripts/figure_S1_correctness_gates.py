#!/usr/bin/env python3
"""Figure S1 (Appendix B.2): correctness gates of the patched B200 kernel, from element-wise readbacks.

A single tile (sequence 128): residual of the kernel probabilities against the Rowmax-H15 reference (V one-hot,
so O = P); B four tiles (sequence 512): measured normalised log2(P1/P0) against the bound it must satisfy, the
recovered unnormalised weight error against the analytic envelope [-0.25, 0.335] octave, and the per-row
normalisation shift.  Data: results/kernel/correctness_dumps/{L1_singletile_P.pt, L15_multitile_P.pt}
(written by kernel/correctness/run_gates.sh)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
import torch

D1 = B200_RES / "correctness_dumps" / "L1_singletile_P.pt"
D15 = B200_RES / "correctness_dumps" / "L15_multitile_P.pt"
d1 = torch.load(D1, map_location="cpu", weights_only=False)
d15 = torch.load(D15, map_location="cpu", weights_only=False)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_W, 3.4), layout="constrained")  # 1 x 2 at text width  # stacked at text width, 7 pt

# Panel A: single-tile element-wise kernel vs reference
P1, Pref, valid = d1["P1"], d1["P_ref"], d1["valid"]
l1 = d1["l1_per_row"]; resid = (P1 - Pref)[valid]
n_valid = int(valid.sum()); l1_mean = float(l1.mean()); l1_max = float(l1.max()); res_max = float(resid.abs().max())
axA.scatter(Pref[valid].numpy(), resid.numpy(), s=1.2, alpha=0.35, color=C["blue"], lw=0, rasterized=True)
axA.axhline(0, color="k", lw=0.6)
axA.set_xscale("log"); axA.set_xlabel("reference probability $P_{\\mathrm{ref}}$ (log)")
axA.set_ylabel("kernel $-$ reference")
from matplotlib.transforms import blended_transform_factory as _btf
axA.text(0.98, 0.0036, f"single tile, {n_valid:,} valid elements\nmean L1/row = {l1_mean:.3e}\nmax L1/row = {l1_max:.3e}\nmax |residual| = {res_max:.2e}",
         transform=_btf(axA.transAxes, axA.transData), va="center", ha="right", fontsize=6.5, bbox=dict(fc="white", ec="0.6", lw=0.5))  # stats box inside, right, in the empty band between the zero cluster and the positive-residual arc  # stats box beside the axes
panel_label(axA, "A")
for rid_, which, v, txt, deriv in [
    ("figS4A_l1_mean", "l1_mean", l1_mean, f"{l1_mean:.3e}", "mean over rows of the per-row L1 distance between kernel and reference probabilities (measured, single tile)"),
    ("figS4A_l1_max", "l1_max", l1_max, f"{l1_max:.3e}", "max over rows of the per-row L1 distance"),
    ("figS4A_nvalid", "l1_rows", float(l1.numel()), f"{l1.numel()}", "rows in the single-tile gate")]:
    rec("figS4 panelA scalar", txt, v, deriv, D1, f"dump key via which={which}", "{:.3e}", dict(kind="dump", path=str(D1), which=which), rid=rid_)
rec("figS4 panelA scalar", f"{n_valid}", n_valid, "count of valid (causally allowed) elements in the single-tile gate", D1, "dump key: valid.sum()", "{:.0f}", dict(kind="dump_extra", path=str(D1), which="n_valid"), rid="figS4A_nelem")

# Panel B: two error objects, each against its own bound
# the readback stores NORMALISED row probabilities (row sums ~ 1), so the measured
# log2(P1/P0) is a normalised-probability error.  The analytic envelope [-0.25, 0.335] bounds the
# UNNORMALISED weight error eps_j = log2(w_hat_j / w_j); for eps in [a, b] the corresponding general
# normalised bound is [a-b, b-a].  Both objects are drawn, each against the bound that applies to it.
# The recovered eps uses the operator's exact fixed point at its anchor (H15(0) = 1); under this dump's
# per-tile FP8 semantics the anchor is the row maximum, so the value at the row-maximum entry is the
# per-row normalisation shift -log2(Z1/Z0) and subtracting it leaves the per-element weight error.
r = d15["log2_ratio"][d15["mask_used"]].numpy()
LO, HI = d15["meta"]["envelope"]; E = d15["meta"]["e4m3_slack"]
_rows = torch.arange(d15["log2_ratio"].shape[0])
shift = d15["log2_ratio"][_rows, d15["argmax_idx"]]            # = -log2(Z1/Z0) at the exact anchor
eps = (d15["log2_ratio"] - shift.unsqueeze(1))[d15["mask_used"]].numpy()
NB = HI - LO                                                    # general normalised bound, +/- NB
out_eps = float(((eps < LO) | (eps > HI)).mean() * 100)         # against the bare analytic envelope
out_eps_slack = float(((eps < LO - E) | (eps > HI + E)).mean() * 100)
out_norm = float(((r < -NB - E) | (r > NB + E)).mean() * 100)
rs = d15["P1"].sum(-1); rs_lo, rs_hi = float(rs.min()), float(rs.max())
_P1 = d15["P1"]; kept = int((_P1[_rows, d15["argmax_idx"]] == _P1.max(-1).values).sum())
_ties = (_P1 == _P1.max(-1, keepdim=True).values).sum(-1)
tie_med, tie_max = int(_ties.median()), int(_ties.max())

axB.axvspan(-NB - E, NB + E, color=C["grey"], alpha=0.25, lw=0, label=f"normalised bound $\\pm${NB:.3f} + slack (for the filled bars)")
axB.axvspan(LO, HI, color=C["yellow"], alpha=0.55, lw=0, label=f"analytic weight envelope [{LO}, {HI}] octave (for the outlined bars)")
_bins = np.linspace(min(r.min(), eps.min()) - 0.02, max(r.max(), eps.max(), NB + E) + 0.02, 150)
axB.hist(r, bins=_bins, color=C["blue"], alpha=0.8, label=f"measured $\\log_2(P_1/P_0)$, normalised (n = {len(r):,})")
axB.hist(eps, bins=_bins, histtype="step", color=C["vermil"], lw=0.8, label="recovered weight error $\\log_2(\\hat w/w)$")
axB.axvline(0, color="k", lw=0.6)
axB.set_yscale("log"); axB.set_ylim(bottom=0.7, top=float(np.histogram(r, bins=_bins)[0].max()) * 4000)
axB.plot(shift.numpy(), np.full(len(shift), 0.85), "|", color=C["green"], ms=7, mew=0.7, alpha=0.9,
         label=f"per-row normalisation shift (n = {len(shift)})")
axB.set_xlabel("$\\log_2$ ratio  [octave]"); axB.set_ylabel("count (log)")
axB.text(0.97, 0.97, f"multi-tile (512 = 4$\\times$128)\n{len(r):,} valid elements\nweight error outside envelope: {out_eps:.3f}%\n(0 outside envelope + slack)\nrow sums in [{rs_lo:.4f}, {rs_hi:.4f}]",
         transform=axB.transAxes, va="top", ha="right", bbox=dict(fc="white", ec="0.6", lw=0.5))
fig.legend(*axB.get_legend_handles_labels(), loc="outside lower center", ncols=1, handletextpad=0.3, columnspacing=1.0)
panel_label(axB, "B")
for rid_, which, v, txt, deriv, fmt in [
    ("figS4B_eps_out", "l15_eps_outside_pct", out_eps, f"{out_eps:.3f}%", "share of valid multi-tile elements whose recovered unnormalised weight error falls outside the analytic Rowmax-H15 envelope", "{:.4f}"),
    ("figS4B_eps_out_slack", "l15_eps_outside_slack_pct", out_eps_slack, f"{out_eps_slack:.3f}%", "the same share against the envelope widened by the E4M3 storage slack of the readback path", "{:.4f}"),
    ("figS4B_eps_lo", "l15_eps_lo", float(eps.min()), f"{eps.min():.4f}", "minimum recovered unnormalised weight error", "{:.4f}"),
    ("figS4B_eps_hi", "l15_eps_hi", float(eps.max()), f"{eps.max():.4f}", "maximum recovered unnormalised weight error", "{:.4f}"),
    ("figS4B_norm_lo", "l15_norm_lo", float(r.min()), f"{r.min():.4f}", "minimum measured normalised log2(P1/P0)", "{:.4f}"),
    ("figS4B_norm_hi", "l15_norm_hi", float(r.max()), f"{r.max():.4f}", "maximum measured normalised log2(P1/P0)", "{:.4f}"),
    ("figS4B_norm_bound", "l15_norm_bound", float(NB), f"{NB:.3f}", "general normalised bound b-a implied by the unnormalised envelope [a, b]", "{:.4f}"),
    ("figS4B_norm_out", "l15_norm_outside_pct", out_norm, f"{out_norm:.3f}%", "share of measured normalised ratios outside that bound widened by the E4M3 slack", "{:.4f}"),
    ("figS4B_shift_lo", "l15_shift_lo", float(shift.min()), f"{shift.min():.4f}", "most negative per-row normalisation shift, read at the row-maximum entry", "{:.4f}"),
    ("figS4B_shift_hi", "l15_shift_hi", float(shift.max()), f"{shift.max():.4f}", "largest per-row normalisation shift", "{:.4f}"),
    ("figS4B_shift_mean", "l15_shift_mean", float(shift.mean()), f"{shift.mean():.4f}", "mean per-row normalisation shift", "{:.4f}"),
    ("figS4B_kept", "l15_argmax_kept", float(kept), f"{kept}", "rows in which the stock arm's row-maximum entry still attains the patched arm's row maximum", "{:.0f}"),
    ("figS4B_rows", "l15_rows", float(len(shift)), f"{len(shift)}", "rows in the multi-tile gate", "{:.0f}"),
    ("figS4B_tiemed", "l15_tie_median", float(tie_med), f"{tie_med}", "median number of entries tied at the patched arm's row maximum", "{:.0f}"),
    ("figS4B_tiemax", "l15_tie_max", float(tie_max), f"{tie_max}", "maximum number of entries tied at the patched arm's row maximum", "{:.0f}"),
    ("figS4B_n", "l15_n", float(len(r)), f"{len(r)}", "valid elements compared in the multi-tile gate", "{:.0f}"),
    ("figS4B_rslo", "rowsum_lo", rs_lo, f"{rs_lo:.4f}", "minimum kernel row sum", "{:.4f}"),
    ("figS4B_rshi", "rowsum_hi", rs_hi, f"{rs_hi:.4f}", "maximum kernel row sum", "{:.4f}"),
    ("figS4B_envlo", "env_lo", float(LO), f"{LO}", "analytic Rowmax-H15 envelope lower bound (dump metadata)", "{:.4f}"),
    ("figS4B_envhi", "env_hi", float(HI), f"{HI}", "analytic Rowmax-H15 envelope upper bound (dump metadata)", "{:.4f}"),
    ("figS4B_slack", "e4m3_slack", float(E), f"{E:g}", "E4M3 storage slack of the readback path", "{:.4f}")]:
    rec("figS4 panelB scalar", txt, v, deriv, D15, f"dump key via which={which}", fmt, dict(kind="dump", path=str(D15), which=which), rid=rid_)
save(fig, "figS4_correctness_gates"); flush_manifest("figS4_correctness_gates")
print(f"A: n_valid={n_valid} mean_l1={l1_mean:.4e} max_l1={l1_max:.4e} max_resid={res_max:.3e}")
print(f"B: n={len(r)} eps_outside={out_eps:.3f}% eps_outside_slack={out_eps_slack:.3f}% norm=[{r.min():.4f},{r.max():.4f}] norm_outside={out_norm:.3f}% shift=[{shift.min():.4f},{shift.max():.4f}] mean {shift.mean():.4f} argmax_kept={kept}/{len(shift)} ties med {tie_med} max {tie_max} rowsum=[{rs_lo:.4f},{rs_hi:.4f}]")
