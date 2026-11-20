#!/usr/bin/env python3
"""Write paper/tables/table_D3_ggap.tex (Table D3): the primary endpoint G_gap(10J0) and the rescued contrast D_rescue(10J0) for the ten models,
read from results/characterization/<model>/T10/rescue_effects.csv (row target_label=10K0; columns rescued_gap*, gap_reduction*).
Those intervals are the pipeline's paired block bootstrap of the per-block series (softmax_approx/characterization.py, T10:
Bootstrapper.mean_ci(dorig - dres), 5,000 replicates, seed 0); nothing is recomputed here.  Values shown to three
significant figures.  Also checks that original_gap equals the D_T(10J0) of Table D2."""
import csv, os, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
DRAFT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("APPROX_SOFTMAX_RESULTS", DRAFT.parent / "results")) / "characterization"
MODELS = [("qwen05", "Qwen2.5-0.5B"), ("llama1", "Llama-3.2-1B"), ("qwen15", "Qwen2.5-1.5B"), ("gemma2", "Gemma-2-2B"),
          ("qwen3", "Qwen2.5-3B"), ("llama3", "Llama-3.2-3B"), ("mistral7", "Mistral-7B-v0.3"), ("llama8", "Llama-3.1-8B"),
          ("llama70", "Llama-3.1-70B"), ("qwen72", "Qwen2.5-72B")]
def s3(x):
    d = Decimal(repr(float(x)))
    if d == 0: return "0"
    return format(d.quantize(Decimal(1).scaleb(d.adjusted() - 2), rounding=ROUND_HALF_UP), "f")
def fmt(r, k, lo, hi):
    return f"{s3(r[k])} [{s3(r[lo])}, {s3(r[hi])}]".replace("-", "$-$")
dt = (DRAFT / "tables/table_D2_direction.tex").read_text().replace("$-$", "-").replace("\\textbf{", "")
rows = []
for alias, name in MODELS:
    r = next(x for x in csv.DictReader(open(RES / alias / "T10/rescue_effects.csv")) if x["target_label"] == "10K0")
    orig = s3(r["original_gap"])
    line = next(l for l in dt.splitlines() if l.startswith(name + " "))
    assert orig in line, (name, orig, "original D_T(10J0) not found in Table D2 row")
    rows.append(f"{name} & {fmt(r, 'rescued_gap', 'rescued_ci_low', 'rescued_ci_high')} & "
                f"{fmt(r, 'gap_reduction', 'gap_reduction_ci_low', 'gap_reduction_ci_high')} \\\\")
out = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize",
       r"\caption{\textbf{Primary endpoint \(G_{\mathrm{gap}}(10J_0)\).}  Rescued contrast \(D_{\mathrm{rescue}}(10J_0)\) and the"
       r" signed change \(G_{\mathrm{gap}}(10J_0)=D_T(10J_0)-D_{\mathrm{rescue}}(10J_0)\) (original \(D_T(10J_0)\) in"
       r" Table~\ref{tab:appx-dt}); 97 blocks, paired block bootstrap of the per-block difference, 5,000 replicates, seed 0."
       r"  A positive \(G_{\mathrm{gap}}\) is not necessarily a smaller \(|D|\).}",
       r"\label{tab:appx-ggap}", r"\begin{tabular}{@{}lll@{}}", r"\toprule",
       r"Model & \(D_{\mathrm{rescue}}(10J_0)\) [95\% CI] & \(G_{\mathrm{gap}}(10J_0)\) [95\% CI] \\", r"\midrule",
       *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(DRAFT / "tables/table_D3_ggap.tex").write_text("\n".join(out) + "\n")
print("wrote paper/tables/table_D3_ggap.tex:", len(rows), "models; original D_T(10J0) matches Table D2 for all")
