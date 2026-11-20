#!/usr/bin/env python3
"""Figure S5 (Appendix D.4): local attention-output change by layer as a negative control (Qwen2.5-1.5B).

A RMS change and B cosine of the local output pV for matched flattening and sharpening at J0 and 10 J0
(descriptive, no intervals).  Data: results/characterization/qwen15/T10/output_probe.csv."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
plt = setup_style()
qw = DEEP_REF
p = RESULTS_FORMAL / qw / "T10" / "output_probe.csv"; d = pd.read_csv(p)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.5), layout="constrained")  # side by side, 7 pt
style = {"jsd_K0_Aminus": (C["orange"], "o", "-", "$A^-$ at $J_0$"), "jsd_K0_Aplus": (C["purple"], "s", "-", "$A^+$ at $J_0$"),
         "jsd_10K0_Aminus": (C["orange"], "o", "--", "$A^-$ at $10J_0$"), "jsd_10K0_Aplus": (C["purple"], "s", "--", "$A^+$ at $10J_0$")}
for c, (col, mk, ls, lab) in style.items():
    s = d[d.condition == c].sort_values("layer")
    axA.plot(s.layer, s.output_rms_used, marker=mk, color=col, ls=ls, ms=2.5, lw=0.8, mfc="white" if "Aplus" in c else col, label=lab)
    axB.plot(s.layer, s.cosine_used, marker=mk, color=col, ls=ls, ms=2.5, lw=0.8, mfc="white" if "Aplus" in c else col, label=lab)
    for _, r in s.iterrows():
        rec("figS1 panelA point", f"{r.output_rms_used:.4f}", r.output_rms_used, "direct (descriptive)", p, f"row: condition={c}; layer={int(r.layer)}; col: output_rms_used", "{:.4f}", dict(kind="direct", path=str(p), row_filter={"condition": c, "layer": int(r.layer)}, col="output_rms_used", scale=1.0), rid=f"figS1A_{c}_L{int(r.layer)}")
        rec("figS1 panelB point", f"{r.cosine_used:.4f}", r.cosine_used, "direct (descriptive)", p, f"row: condition={c}; layer={int(r.layer)}; col: cosine_used", "{:.4f}", dict(kind="direct", path=str(p), row_filter={"condition": c, "layer": int(r.layer)}, col="cosine_used", scale=1.0), rid=f"figS1B_{c}_L{int(r.layer)}")
axA.set_xlabel("layer"); axA.set_ylabel("$pV$ RMS change"); axA.set_yscale("log"); panel_label(axA, "A")
axB.set_xlabel("layer"); axB.set_ylabel("$pV$ cosine"); panel_label(axB, "B")
fig.legend(*axA.get_legend_handles_labels(), loc="outside upper center", ncols=4, handletextpad=0.2, columnspacing=0.8)  # one figure-level legend
# the in-axes note "descriptive, no bootstrap interval" (6 pt, overlapping the panel titles side by side) is stated in the caption instead
save(fig, "figS1_local_pv"); flush_manifest("figS1_local_pv")
