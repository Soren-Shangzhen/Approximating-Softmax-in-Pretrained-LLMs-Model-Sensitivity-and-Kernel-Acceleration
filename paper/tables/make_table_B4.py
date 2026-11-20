#!/usr/bin/env python3
"""Write paper/tables/table_B4_nsys.tex (Table B4) from results/kernel/nsys/kernel_vs_wall.csv and
launch_breakdown.csv; no value is typed by hand."""
import os
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1]
NSYS = Path(os.environ.get("APPROX_SOFTMAX_RESULTS", D.parent / "results")) / "kernel" / "nsys"
kvw = pd.read_csv(NSYS / "kernel_vs_wall.csv"); lbd = pd.read_csv(NSYS / "launch_breakdown.csv")
def seq(L): return f"{L // 1024}K"
out = [r"\begingroup",
       r"\needspace{12\baselineskip}",
       r"\refstepcounter{table}\label{tab:appx-nsys-kernel}\label{tab:appx-nsys-launch}",
       r"\par\medskip\noindent{\small\textbf{Table~\thetable.} Session-3 timeline (FP8, head dimension 128, 16 heads, batch 1): median kernel duration over all 250 traced calls and host-side call latency over the 200 timed calls, per kernel, with the resulting kernel-only and call-latency speed-ups of A1 over A0 and the share of the timed interval outside the kernel (100 minus the kernel share; launch and host work overlap kernel execution, so the share is not an additive time budget).}\par\smallskip",
       r"\addcontentsline{lot}{table}{\protect\numberline{\thetable}Session-3 timeline: kernel duration and call latency}",
       r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{@{}llrrrrrrl@{}}", r"\toprule",
       r"\textbf{seq} & \textbf{mask} & \multicolumn{2}{c}{\textbf{kernel (\(\mu\)s) A0 / A1}} & \multicolumn{2}{c}{\textbf{call latency (\(\mu\)s) A0 / A1}} & \textbf{kernel-only} & \textbf{call latency} & \textbf{non-kernel \% (A0 / A1)} \\", r"\midrule", r"\endhead"]
rows = []
for (L, c), g in kvw.groupby(["seqlen", "causal"], sort=False):
    a0 = g[g.arm == "A0"].iloc[0]; a1 = g[g.arm == "A1"].iloc[0]
    ko = 100 * (a0.kern_med_ns / a1.kern_med_ns - 1); wc = 100 * (a0.wall_per_iter_us / a1.wall_per_iter_us - 1)
    b0 = lbd[(lbd.seqlen == L) & (lbd.causal == c) & (lbd.arm == "A0")].iloc[0]; b1 = lbd[(lbd.seqlen == L) & (lbd.causal == c) & (lbd.arm == "A1")].iloc[0]
    rows.append((seq(int(L)), "causal" if c == 1 else "non-causal", f"{a0.kern_med_ns/1000:.3f}", f"{a1.kern_med_ns/1000:.3f}",
                 f"{a0.wall_per_iter_us:.3f}", f"{a1.wall_per_iter_us:.3f}", f"{ko:+.1f}\\%", f"{wc:+.1f}\\%",
                 f"{100-b0.kern_pct_of_wall:.1f} / {100-b1.kern_pct_of_wall:.1f}"))
out += [" & ".join(r) + r" \\" for r in rows]
out += [r"\bottomrule", r"\end{longtable}", r"\addtocounter{table}{-1}", r"}", r"\endgroup"]
(D / "tables" / "table_B4_nsys.tex").write_text("\n".join(out) + "\n")
print("wrote paper/tables/table_B4_nsys.tex:", len(rows), "rows")
