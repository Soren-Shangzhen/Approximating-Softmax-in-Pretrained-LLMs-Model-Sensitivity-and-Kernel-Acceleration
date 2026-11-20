#!/usr/bin/env python3
"""Recompute every number in the paper's tables from the result files and compare it with the LaTeX table.

    python paper/tables/check_tables.py            # prints a summary, writes paper/tables/check_report.csv

The tables of paper/tables/*.tex are those of the arXiv source.  For each numeric cell of Tables 2, 3, B1, B2 (with
the exponential-map sub-table), B5, C1, C2, D1 and D2, this script re-derives the value from the per-block and
per-run files under results/ and checks it against the printed number at the printed precision: a cell passes
when the recomputed value, rounded to the number of decimals shown, equals the printed number (half-even or
half-up rounding).  Integers must match exactly; short text cells (e.g. 'yes', 'unresolved') must be present.
Tables B4 and D3 are generated from results/ by make_table_B4.py and make_table_D3.py.

Statistics follow softmax_approx/characterization.py: the point estimate is the mean over the 97 aligned
2048-token blocks of the per-block NLL difference, the 95% interval is the percentile paired block bootstrap of
Bootstrapper.mean_ci (5,000 replicates, seed 0; replicated in boot() below), and a perplexity change is
100 (exp(dNLL) - 1).  B200 speed-ups are 100 (median A0 / median A1 - 1) on the pooled windows of all rounds; the
per-round spread is the population sd of the per-round speed-ups.  Needs numpy and pandas only (no GPU, no torch).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables"
RES = Path(os.environ.get("APPROX_SOFTMAX_RESULTS", ROOT / "results"))
RF = RES / "characterization"
RI = RES / "lattice_operators"
RL = RES / "characterization_large_T11_T12"
B200 = RES / "kernel"
BEXT = B200 / "timing"
BPPL = B200 / "fidelity"
EXPABL = RES / "exp_map_ablation"
LEAK = RI / "qwen15" / "leakage" / "leakage_summary.csv"
REPS, SEED = 5000, 0

DISPLAY = {"qwen05": "Qwen2.5-0.5B", "qwen15": "Qwen2.5-1.5B", "qwen3": "Qwen2.5-3B", "qwen72": "Qwen2.5-72B",
           "llama1": "Llama-3.2-1B", "llama3": "Llama-3.2-3B", "llama8": "Llama-3.1-8B", "llama70": "Llama-3.1-70B",
           "gemma2": "Gemma-2-2B", "mistral7": "Mistral-7B-v0.3"}
ALIAS = {v: k for k, v in DISPLAY.items()}

# ----------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------
_IDX: dict[int, np.ndarray] = {}


def boot(x) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, identical to softmax_approx.characterization.Bootstrapper.mean_ci(x, 5000, 0)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) not in _IDX:
        _IDX[len(x)] = np.random.default_rng(SEED).integers(0, len(x), size=(REPS, len(x)), endpoint=False)
    b = x[_IDX[len(x)]].mean(axis=1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def ppl(dnll: float) -> float:
    return 100.0 * math.expm1(dnll)


def blocks(alias: str, cond: str, root: str = "formal", L: int = 2048) -> pd.DataFrame:
    base = {"formal": RF / alias, "lattice": RI / alias, "70b": RL / alias, "ablation": EXPABL / alias}[root]
    df = pd.read_csv(base / "condition_blocks" / f"{cond}__L{L}.csv").sort_values("block").reset_index(drop=True)
    assert not df.block.duplicated().any()
    return df


def delta(alias, a, b=None, ra="formal", rb=None, L=2048) -> np.ndarray:
    """Per-block delta_nll of condition a, or the paired difference delta_nll(a) - delta_nll(b)."""
    A = blocks(alias, a, ra, L)
    if b is None:
        return A.delta_nll.to_numpy(float)
    B = blocks(alias, b, rb or ra, L)
    m = A.merge(B, on="block", suffixes=("_a", "_b"))
    assert len(m) == len(A) == len(B), (alias, a, b)
    assert np.allclose(m.baseline_nll_a, m.baseline_nll_b, rtol=0, atol=1e-9), f"baseline mismatch {alias} {a} {b}"
    return (m.delta_nll_a - m.delta_nll_b).to_numpy(float)


def bci(*args, **kw) -> list[float]:
    return list(boot(delta(*args, **kw)))


def b200_ppl(f0: str, f1: str) -> np.ndarray:
    d0 = pd.read_csv(BPPL / f0).sort_values("block"); d1 = pd.read_csv(BPPL / f1).sort_values("block")
    assert (d0.block.values == d1.block.values).all() and len(d0) == 97
    return (d1.nll.to_numpy() - d0.nll.to_numpy())


def lat(name: str, sub: str = "ext") -> pd.DataFrame:
    return pd.read_csv((BEXT if sub == "ext" else B200) / f"{name}.csv")


def med(df: pd.DataFrame, **cfg) -> float:
    m = np.ones(len(df), bool)
    for k, v in cfg.items():
        if k in df.columns:
            m &= (df[k] == v).to_numpy()
    assert m.any(), cfg
    return float(df[m].latency_ms.median())


def rounds(prefix: str, n: int) -> list[tuple[str, str]]:
    return [(f"{prefix}_r{i}_A0", f"{prefix}_r{i}_A1") for i in range(1, n + 1)]


def multi(pairs, **cfg) -> dict:
    """Pooled-median speed-up over all rounds, per-round speed-ups and their population sd."""
    sp = np.array([100 * (med(lat(a), **cfg) / med(lat(b), **cfg) - 1) for a, b in pairs])
    A0 = pd.concat([lat(a) for a, _ in pairs]); A1 = pd.concat([lat(b) for _, b in pairs])
    a0, a1 = med(A0, **cfg), med(A1, **cfg)
    return dict(a0=a0, a1=a1, pooled=100 * (a0 / a1 - 1), sd=float(sp.std(ddof=0)), min=float(sp.min()),
                max=float(sp.max()), n=len(pairs))


BASE = dict(heads=16, hd=128, batch=1)

# ----------------------------------------------------------------------------------------------------------------
# LaTeX access and comparison
# ----------------------------------------------------------------------------------------------------------------
NUM = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")


def norm(s: str) -> str:
    s = s.replace("$-$", "-").replace("\\(-", "-").replace("−", "-").replace("\\(", "").replace("\\)", "")
    s = s.replace("---", " n/a ").replace("--", " to ").replace("\\ ", " ")
    return s.replace("\\%", "%").replace("\\textbf{", "").replace("\\quad", "")


_TEX: dict[str, list[str]] = {}


def tex(name: str) -> list[str]:
    if name not in _TEX:
        _TEX[name] = [l for l in (TAB / name).read_text().splitlines() if not l.lstrip().startswith("%")]
    return _TEX[name]


def cells_of(line: str) -> list[str]:
    line = re.sub(r"\\\\\s*$", "", line.strip())
    return [c.strip() for c in line.split("&")]


def row(name: str, *keys: str, after: str | None = None) -> list[str]:
    lines = tex(name)
    start = 0
    if after is not None:
        start = next(i for i, l in enumerate(lines) if after in l)
    for l in lines[start:]:
        if "&" not in l:
            continue
        c = cells_of(l)
        if len(c) >= len(keys) and all(norm(c[i]).strip().startswith(k) for i, k in enumerate(keys)):
            return c
    raise KeyError(f"{name}: row {keys} (after {after!r}) not found")


REPORT: list[dict] = []


def _match_num(v, tok: str) -> bool:
    s = tok.replace(",", "")
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return "." not in s and int(s) == int(v)
    d = len(s.split(".")[1]) if "." in s else 0
    q = Decimal(repr(float(v)))
    return any(q.quantize(Decimal(1).scaleb(-d), rounding=r) == Decimal(s) for r in (ROUND_HALF_EVEN, ROUND_HALF_UP))


def check_cell(table: str, label: str, cell: str, expected: list, source: str):
    """expected: values for the numeric tokens of the cell, in order (None = token not checked); a str is a phrase
    that must occur in the cell."""
    toks = NUM.findall(norm(cell))
    words = [e for e in expected if isinstance(e, str)]
    nums = [e for e in expected if not isinstance(e, str)]
    for w in words:
        REPORT.append(dict(table=table, item=label, printed=cell, recomputed=w, ok=w in norm(cell), source=source))
    if not nums:
        return
    if len(toks) != len(nums):
        REPORT.append(dict(table=table, item=label, printed=cell, recomputed=repr(nums), ok=False,
                           source=source + f" [token count {len(toks)} != {len(nums)}]"))
        return
    for t, v in zip(toks, nums):
        if v is None:
            continue
        REPORT.append(dict(table=table, item=label, printed=t, recomputed=float(v), ok=_match_num(v, t), source=source))


def check_row(table: str, keys, values: dict, source: str, after: str | None = None):
    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    c = row(table, *keys, after=after)
    for col, exp in values.items():
        check_cell(table, " | ".join(keys) + f" [col {col}]", c[col], exp, source)


def check_after(table: str, literal: str, value, source: str, ntok: int = 0, occurrence: int = 0):
    """Check the ntok-th number after the occurrence-th `literal` in the table file (footnotes, inline sentences)."""
    t = norm("\n".join(tex(table)))
    lit = norm(literal)
    i = -1
    for _ in range(occurrence + 1):
        i = t.find(lit, i + 1)
        if i < 0:
            break
    if i < 0:
        REPORT.append(dict(table=table, item=literal, printed="(literal not found)", recomputed=value, ok=False, source=source))
        return
    tok = NUM.findall(t[i + len(lit):])[ntok]
    REPORT.append(dict(table=table, item=literal + (f" #{ntok}" if ntok else ""), printed=tok,
                       recomputed=float(value) if not isinstance(value, (int, np.integer)) else int(value),
                       ok=_match_num(value, tok), source=source))


# ----------------------------------------------------------------------------------------------------------------
# Table 2 (tab:m-h15): Rowmax-H15 fidelity, call latency, energy
# ----------------------------------------------------------------------------------------------------------------
S5 = [("Qwen2.5-1.5B", "ppl_A0.csv", "ppl_A1.csv"), ("Qwen2.5-72B", "ppl_qwen72_A0.csv", "ppl_qwen72_A1.csv"),
      ("Mistral-7B-v0.3", "ppl_mistral7_A0.csv", "ppl_mistral7_A1.csv"),
      ("Llama-3.1-8B", "ppl_llama8_A0.csv", "ppl_llama8_A1.csv"),
      ("Llama-3.1-70B", "ppl_llama70_A0.csv", "ppl_llama70_A1.csv")]


def energy():
    f0 = pd.read_csv(BEXT / "F_power_A0.csv").iloc[0]; f1 = pd.read_csv(BEXT / "F_power_A1.csv").iloc[0]
    return f0, f1


def table2():
    T = "table_2_rowmax_h15.tex"
    for model, f0, f1 in S5:
        e, lo, hi = boot(b200_ppl(f0, f1))
        check_row(T, ("BF16 kernel, measured", model), {2: [e, lo, hi], 3: [ppl(e)]},
                  f"kernel/fidelity/{f0}, {f1}: paired nll(A1)-nll(A0)")
    for key, cond in [("BF16 semantic simulation", "h15tiled_bf16"), ("FP8 semantic simulation", "h15tiled")]:
        e, lo, hi = bci("qwen15", cond, ra="lattice")
        check_row(T, (key, "Qwen2.5-1.5B"), {2: [e, lo, hi], 3: [ppl(e)]}, f"lattice_operators/qwen15/condition_blocks/{cond}")
    m = multi(rounds("A_main", 3), dtype="fp8_e4m3", seqlen=8192, causal=1, **BASE)
    check_row(T, "FP8 causal 8K, hd 128: speed-up", {1: [m["a0"]], 2: [m["a1"]], 3: [m["pooled"]]}, "timing/A_main_r{1,2,3}")
    m = multi(rounds("C2_noncausal", 3), dtype="fp8_e4m3", seqlen=8192, causal=0, **BASE)
    check_row(T, "FP8 non-causal 8K, hd 128: speed-up", {1: [m["a0"]], 2: [m["a1"]], 3: [m["pooled"]]}, "timing/C2_noncausal_r{1,2,3}")
    f0, f1 = energy()
    check_row(T, "FP8 causal 16K: energy per forward", {1: [f0.energy_per_fwd_mJ], 2: [f1.energy_per_fwd_mJ],
              3: [100 * (f1.energy_per_fwd_mJ / f0.energy_per_fwd_mJ - 1)]}, "timing/F_power_A{0,1}.csv")


# ----------------------------------------------------------------------------------------------------------------
# Table 3 (tab:crossmodel-summary)
# ----------------------------------------------------------------------------------------------------------------
def cond_row(alias: str, task: str, fname: str, cond: str) -> pd.Series:
    df = pd.read_csv(RF / alias / task / fname)
    r = df[df.condition == cond]
    assert len(r) == 1, (alias, task, cond)
    return r.iloc[0]


def table3():
    T = "table_3_crossmodel.tex"
    summ = pd.read_csv(RF / "crossmodel_summary.csv").set_index("alias")
    for alias, name in DISPLAY.items():
        e, lo, hi = bci(alias, "alloc_K21_R1", "alloc_K21_R4")
        eff = pd.read_csv(RF / alias / "T04" / "effects.csv").set_index("effect").loc["D_R_K21_R1_minus_R4"]
        assert abs(eff.estimate - e) < 1e-12 and abs(eff.ci_low - lo) < 1e-12 and abs(eff.ci_high - hi) < 1e-12, alias
        k32 = cond_row(alias, "T03", "conditions.csv", "precision_K32_R4")
        k20 = cond_row(alias, "T06", "rowmax_pot_conditions.csv", "rowmax_pot_K20_clamp")
        check_row(T, name, {1: [summ.loc[alias, "parameter_count"] / 1e9], 2: [100 * (k32.ppl_ratio - 1)],
                            3: [e, lo, hi], 4: [100 * (k20.ppl_ratio - 1)], 5: [100 * k20.top1_agreement]},
                  f"characterization/{alias}: crossmodel_summary.csv, T03/conditions.csv, alloc_K21_R{{1,4}} blocks "
                  f"(= T04/effects.csv D_R_K21), T06/rowmax_pot_conditions.csv rowmax_pot_K20_clamp")


# ----------------------------------------------------------------------------------------------------------------
# Table B1 (tab:appx-anchor)
# ----------------------------------------------------------------------------------------------------------------
def tableB1():
    T = "table_B1_anchor_resolution.tex"
    lk = pd.read_csv(LEAK).set_index("condition")
    log2 = bci("qwen15", "pq_log2_b8", ra="lattice"); lsq = bci("qwen15", "pq_logsqrt2_b8", ra="lattice")
    m1 = bci("qwen15", "pot_m1"); m2 = bci("qwen15", "response_pot_m2")
    src = "lattice_operators/qwen15/condition_blocks/pq_log2_b8, pq_logsqrt2_b8; characterization/qwen15/condition_blocks/pot_m1, response_pot_m2"
    check_row(T, "octave", {1: [None, None] + log2, 2: [None] + m1,
                            3: [lk.loc["pq_log2_b8", "pearson_r"], lk.loc["pot_m1", "pearson_r"]]},
              src + "; lattice_operators/qwen15/leakage/leakage_summary.csv")
    check_row(T, "half-octave", {1: [None, None] + lsq, 2: [None] + m2, 3: [lk.loc["pq_logsqrt2_b8", "pearson_r"]]},
              src + "; lattice_operators/qwen15/leakage/leakage_summary.csv")
    pairs = [("anchor: absolute $\\rightarrow$ rowmax, octave spacing", ("pq_log2_b8", "lattice"), ("pot_m1", "formal")),
             ("anchor: absolute $\\rightarrow$ rowmax, half-octave spacing", ("pq_logsqrt2_b8", "lattice"), ("response_pot_m2", "formal")),
             ("spacing: octave $\\rightarrow$ half-octave, absolute anchor", ("pq_log2_b8", "lattice"), ("pq_logsqrt2_b8", "lattice")),
             ("spacing: octave $\\rightarrow$ half-octave, rowmax anchor", ("pot_m1", "formal"), ("response_pot_m2", "formal"))]
    for key, (a, ra), (b, rb) in pairs:
        check_row(T, norm(key), {1: bci("qwen15", a, b, ra=ra, rb=rb)}, f"paired blocks {a} - {b} (qwen15)")
    v1 = bci("qwen15", "ilpot_v1", ra="lattice")[0]
    for lit, val in [("log2-p / PoT m=1 = ", log2[0] / m1[0]), ("absolute-lattice ILPoT / log2-p = ", v1 / log2[0]),
                     ("log$\\surd$2-p / PoT m=2 = ", lsq[0] / m2[0]), ("log2-p / PoT m=2 = ", log2[0] / m2[0])]:
        check_after(T, lit, val, "ratio of block means (condition_blocks)")


# ----------------------------------------------------------------------------------------------------------------
# Table B2 (tab:appx-h15-full) and the exponential-map sub-table (appx_expabl.tex)
# ----------------------------------------------------------------------------------------------------------------
def tableB2():
    T = "table_B2_rowmax_h15_fidelity.tex"
    k = boot(b200_ppl("ppl_A0.csv", "ppl_A1.csv"))
    rows = [("B200 patched FlashAttention-4 kernel", list(k), "kernel/fidelity/ppl_A{0,1}.csv"),
            ("Semantic simulation of the BF16", bci("qwen15", "h15tiled_bf16", ra="lattice"), "lattice_operators h15tiled_bf16"),
            ("Semantic simulation of the FP8", bci("qwen15", "h15tiled", ra="lattice"), "lattice_operators h15tiled"),
            ("Offline whole-row Rowmax-H15", bci("qwen15", "h15rowmax", ra="lattice"), "lattice_operators h15rowmax"),
            ("Absolute-anchored", bci("qwen15", "ilpot_h15_v1", ra="lattice"), "lattice_operators ilpot_h15_v1"),
            ("PoT m = 2 (half-octave reference)", bci("qwen15", "response_pot_m2"), "characterization response_pot_m2"),
            ("ILPoT-H-A (half-octave reference)", bci("qwen15", "ilpot_ha_v1", ra="lattice"), "lattice_operators ilpot_ha_v1")]
    for key, v, src in rows:
        check_row(T, key, {1: v, 2: [ppl(v[0])]}, src + " (qwen15 condition_blocks)")
    con = [("FP8 semantic Rowmax-H15 $-$ PoT m = 2", ("h15tiled", "lattice"), ("response_pot_m2", "formal")),
           ("FP8 semantic Rowmax-H15 $-$ ILPoT-H-A", ("h15tiled", "lattice"), ("ilpot_ha_v1", "lattice")),
           ("FP8 semantic Rowmax-H15 $-$ offline whole-row", ("h15tiled", "lattice"), ("h15rowmax", "lattice")),
           ("BF16 semantic Rowmax-H15 $-$ FP8 semantic", ("h15tiled_bf16", "lattice"), ("h15tiled", "lattice"))]
    for key, (a, ra), (b, rb) in con:
        check_row(T, norm(key), {1: bci("qwen15", a, b, ra=ra, rb=rb)}, f"paired blocks {a} - {b} (qwen15)")
    sim = bci("qwen15", "h15tiled_bf16", ra="lattice")[0]
    check_row(T, "Simulation vs measured kernel", {1: [k[0], sim], 2: [100 * (k[0] - sim) / k[0]]},
              "ppl_A{0,1}.csv and lattice_operators h15tiled_bf16")
    tb = blocks("qwen15", "h15tiled", "lattice")
    top1 = float(np.average(tb.top1_agreement, weights=tb.n_pred))
    check_after(T, "for the FP8 semantic operator: ", top1, "lattice_operators h15tiled blocks: prediction-weighted top-1")
    lk = pd.read_csv(LEAK).set_index("condition")
    check_after(T, "Pearson r for the FP8 semantic operator: ", lk.loc["h15tiled", "pearson_r"], "leakage_summary.csv h15tiled")
    check_after(T, "phase-aligned; compare ", lk.loc["ilpot_v1", "pearson_r"], "leakage_summary.csv ilpot_v1")


def table_expabl():
    T = "table_B2_exp_map_ablation.tex"
    cond = {"exact": "baseline", "h15": "h15rowmax", "potm2": "response_pot_m2", "sx": "schraudolph_rowmax",
            "sk2": "schraudolph_rowmax_k2", "sk3": "schraudolph_rowmax_k3"}
    bl = {m: blocks("qwen15", c, "ablation").set_index("block") for m, c in cond.items()}
    nll = pd.DataFrame({m: d.nll for m, d in bl.items()}).sort_index()
    npred = pd.DataFrame({m: d.n_pred for m, d in bl.items()}).sort_index()
    assert len(nll) == 97 and (npred.nunique(axis=1) == 1).all()
    ex, h15 = nll["exact"].to_numpy(), nll["h15"].to_numpy()

    def c(m, ref):
        return list(boot(nll[m].to_numpy() - ref))
    src = "exp_map_ablation/qwen15/condition_blocks (nll of each operator minus the softmax run of the same blocks)"
    v = c("h15", ex)
    check_row(T, "Rowmax-H15, H(x)", {1: v, 2: [ppl(v[0])]}, src)
    for key, m in [("Rowmax-PoT m=2", "potm2"), ("Unshifted Schraudolph", "sx"), ("S(Q_{2^{-2}}(x)), k=2", "sk2"),
                   ("S(Q_{2^{-3}}(x)), k=3", "sk3")]:
        v = c(m, ex)
        check_row(T, key, {1: v, 2: [ppl(v[0])], 3: c(m, h15)}, src)


# ----------------------------------------------------------------------------------------------------------------
# Table B5 (tab:appx-b200)
# ----------------------------------------------------------------------------------------------------------------
def tableB5():
    T = "table_B5_b200.tex"
    A = "(a) Main matrix"
    for dt, lab in [("fp8_e4m3", "FP8 E4M3"), ("bf16", "BF16")]:
        for S, s in [(4096, "4K"), (8192, "8K"), (16384, "16K")]:
            m = multi(rounds("A_main", 3), dtype=dt, seqlen=S, causal=1, **BASE)
            check_row(T, (lab, s), {2: [m["a0"]], 3: [m["a1"]], 4: [m["pooled"]], 5: [m["sd"]], 6: [3]}, "timing/A_main_r{1,2,3}", after=A)
    m = multi(rounds("J_fp16_causal", 3), dtype="fp16", seqlen=8192, causal=1, **BASE)
    check_row(T, ("FP16", "8K"), {2: [m["a0"]], 3: [m["a1"]], 4: [m["pooled"]], 5: [m["sd"]], 6: [3]}, "timing/J_fp16_causal_r{1,2,3}", after=A)
    m = multi([("Y_fp16_A0", "Y_fp16_A1")], dtype="fp16", seqlen=16384, causal=1, **BASE)
    check_row(T, ("FP16", "16K"), {2: [m["a0"]], 3: [m["a1"]], 4: [m["pooled"]], 6: [1]}, "timing/Y_fp16_A{0,1}", after=A)

    Bm = "(b) FP8 causal sequence-length sweep"
    for S, s in [(1024, "1K"), (2048, "2K"), (4096, "4K"), (8192, "8K"), (16384, "16K"), (32768, "32K")]:
        m = multi(rounds("S_seqsweep", 3), dtype="fp8_e4m3", seqlen=S, causal=1, **BASE)
        vals = {1: [m["pooled"]], 2: [m["min"], m["max"]], 3: [m["sd"]]}
        if S <= 2048:
            vals[4] = [multi([("S_short_A0", "S_short_A1")], dtype="fp8_e4m3", seqlen=S, causal=1, **BASE)["pooled"]]
        check_row(T, s, vals, "timing/S_seqsweep_r{1,2,3}" + (", ext/S_short_A{0,1}" if S <= 2048 else ""), after=Bm)
    sw = pd.concat([lat(f"S_seqsweep_r{i}_A{a}").assign(arm=f"A{a}") for i in (1, 2, 3) for a in (0, 1)])
    small = sorted({float(sw[(sw.arm == a) & (sw.seqlen == S)].sm_clock.median()) for a in ("A0", "A1") for S in (1024, 2048, 4096, 8192)})
    assert small == [1965.0], small
    check_after(T, "Recorded SM clock: ", 1965, "timing/S_seqsweep: median sm_clock, both arms, seq <= 8K")
    check_after(T, "at 16K A1's median was ", float(sw[(sw.arm == "A1") & (sw.seqlen == 16384)].sm_clock.median()), "timing/S_seqsweep sm_clock")
    a1 = sw[(sw.arm == "A1") & (sw.seqlen == 32768)].sm_clock; a0 = sw[(sw.arm == "A0") & (sw.seqlen == 32768)].sm_clock
    for j, v in enumerate([a1.median(), a1.min(), a1.max(), a0.median(), a0.min(), a0.max()]):
        check_after(T, "at 32K A1 ran at a median of ", float(v), "timing/S_seqsweep sm_clock at 32K", ntok=j)

    Cm = "(c) Non-causal cells"
    for lab, pre, dt in [("FP8", "C2_noncausal", "fp8_e4m3"), ("BF16", "L_bf16_noncausal", "bf16"), ("FP16", "K_fp16_noncausal", "fp16")]:
        m = multi(rounds(pre, 3), dtype=dt, seqlen=8192, causal=0, **BASE)
        check_row(T, (lab, "non-causal", "8K"), {3: [m["pooled"], m["sd"]], 4: [3]}, f"timing/{pre}_r{{1,2,3}}", after=Cm)
    for lab, dt in [("FP8", "fp8_e4m3"), ("BF16", "bf16")]:
        for S, s in [(4096, "4K"), (16384, "16K")]:
            m = multi([("C_noncausal_A0", "C_noncausal_A1")], dtype=dt, seqlen=S, causal=0, **BASE)
            check_row(T, (lab, "non-causal", s), {3: [m["pooled"]], 4: [1]}, "timing/C_noncausal_A{0,1}", after=Cm)

    Dm = "(d) Stock-FA4 polynomial-emulation"

    def beta(freq, start, res=4, frg=4, kmod=16):
        if freq == 0:
            return 0.0
        emu = sum(1 for j in range(frg) for k in range(kmod) if not (k % freq < freq - res or j >= frg - 1 or j < start))
        return 100.0 * emu / (frg * kmod)
    stock = float(lat("E_freq8_s1").latency_ms.median())
    for key, name, f, s in [("freq 0 (hardware MUFU only)", "E_freq0", 0, 1), ("freq 16, start 1", "E_freq16_s1", 16, 1),
                            ("freq 8, start 1 (stock)", "E_freq8_s1", 8, 1), ("freq 8, start 0", "E_freq8_s0", 8, 0),
                            ("freq 2, start 1", "E_freq2_s1", 2, 1), ("freq 4, start 1", "E_freq4_s1", 4, 1),
                            ("freq 2, start 0", "E_freq2_s0", 2, 0), ("freq 4, start 0", "E_freq4_s0", 4, 0)]:
        rel = 100 * (stock / float(lat(name).latency_ms.median()) - 1)
        check_row(T, key, {1: [beta(f, s)], 2: [rel]}, f"timing/{name}.csv vs ext/E_freq8_s1.csv; beta from the kernel predicate", after=Dm)
    check_row(T, "A1 Rowmax-H15 (full replacement)", {2: [100 * (stock / float(lat("E_h15").latency_ms.median()) - 1)]},
              "timing/E_h15.csv vs ext/E_freq8_s1.csv", after=Dm)

    Em = "(e) Shape robustness"
    for key, tag, cfg in [("8K, heads 4", "G_heads4", dict(heads=4)), ("8K, heads 8", "G_heads8", dict(heads=8)),
                          ("8K, heads 32", "G_heads32", dict(heads=32))]:
        m = multi([(f"{tag}_A0", f"{tag}_A1")], dtype="fp8_e4m3", seqlen=8192, causal=1, hd=128, batch=1, **cfg)
        check_row(T, key, {1: [m["pooled"]]}, f"timing/{tag}_A{{0,1}}", after=Em)
    m = multi([("H_b4_A0", "H_b4_A1")], dtype="fp8_e4m3", seqlen=8192, causal=1, heads=16, hd=128, batch=4)
    check_row(T, "8K, batch 4", {1: [m["pooled"]]}, "timing/H_b4_A{0,1}", after=Em)
    m = multi(rounds("I_gqa", 3), dtype="fp8_e4m3", seqlen=8192, hq=32, hkv=8, hd=128, batch=1, causal=1)
    check_row(T, "8K, GQA", {0: [None, None, None, m["a0"], m["a1"]], 1: [m["pooled"], m["sd"]]}, "timing/I_gqa_r{1,2,3}", after=Em)
    for lab, dt in [("head dim 64, FP8", "fp8_e4m3"), ("head dim 64, BF16", "bf16")]:
        v = [multi([("B_hd64_A0", "B_hd64_A1")], dtype=dt, seqlen=S, causal=1, heads=16, hd=64, batch=1)["pooled"] for S in (4096, 8192, 16384)]
        check_row(T, lab, {1: v}, "timing/B_hd64_A{0,1}", after=Em)
    v = [multi([("D_seqext_A0", "D_seqext_A1")], dtype="fp8_e4m3", seqlen=S, causal=1, **BASE)["pooled"] for S in (2048, 32768)]
    check_row(T, "2K / 32K (extension run)", {1: v}, "timing/D_seqext_A{0,1}", after=Em)

    Fm = "(f) Sustained power and energy"
    f0, f1 = energy()
    check_row(T, "A0 stock FA4", {1: [int(f0.n_forwards)], 2: [f0.mean_power_w], 3: [f0.per_fwd_ms], 4: [f0.energy_per_fwd_mJ]}, "timing/F_power_A0.csv", after=Fm)
    check_row(T, "A1 FA4 + Rowmax-H15", {1: [int(f1.n_forwards)], 2: [f1.mean_power_w], 3: [f1.per_fwd_ms], 4: [f1.energy_per_fwd_mJ]}, "timing/F_power_A1.csv", after=Fm)
    check_after(T, "Energy and time per forward both change by ", 100 * (f1.energy_per_fwd_mJ / f0.energy_per_fwd_mJ - 1), "F_power energy_per_fwd_mJ")
    check_after(T, "Energy and time per forward both change by ", 100 * (f1.per_fwd_ms / f0.per_fwd_ms - 1), "F_power per_fwd_ms")

    Gm = "(g) External references"
    for S, s in [(4096, "4K"), (8192, "8K"), (16384, "16K")]:
        x = {b: float(lat(b)[lat(b).seqlen == S].latency_ms.median()) for b in ("X_torch_flash_bf16", "X_mem_eff_bf16", "X_cudnn_bf16")}
        m = multi(rounds("A_main", 3), dtype="bf16", seqlen=S, causal=1, **BASE)
        cu = x["X_cudnn_bf16"]
        check_row(T, s, {1: [x["X_torch_flash_bf16"]], 2: [x["X_mem_eff_bf16"]], 3: [cu], 4: [m["a0"]], 5: [m["a1"]],
                         6: [100 * (cu / m["a1"] - 1)], 7: [100 * (cu / m["a0"] - 1)]},
                  "timing/X_{torch_flash,mem_eff,cudnn}_bf16.csv; ext/A_main_r{1,2,3}", after=Gm)


# ----------------------------------------------------------------------------------------------------------------
# Tables C1 (70B per-context matching) and C2 (zero-shot)
# ----------------------------------------------------------------------------------------------------------------
def tableC1():
    T = "table_C1_long_context_matching.tex"
    for alias in ("qwen72", "llama70"):
        d = RL / alias / "T11"
        man = json.loads((d / "match_manifest.json").read_text())["contexts"]
        cond = pd.read_csv(d / "conditions.csv")
        for L, s in [(2048, "2K"), (8192, "8K"), (16384, "16K")]:
            c = man[str(L)]
            b = cond[(cond.condition == "baseline") & (cond.seq_len == L)].iloc[0]
            check_row(T, (DISPLAY[alias], s), {2: [int(b.n_blocks)], 3: [int(b.n_pred)], 4: [c["K0_jsd"]],
                                                5: [c["Aminus"]["achieved"], 100 * c["Aminus"]["rel_err"]],
                                                6: [c["Aplus"]["achieved"], 100 * c["Aplus"]["rel_err"]]},
                      f"{d.relative_to(RES)}/match_manifest.json, conditions.csv")


ZS = [("lambada_openai", "acc"), ("hellaswag", "acc_norm"), ("arc_easy", "acc_norm"), ("arc_challenge", "acc_norm"),
      ("piqa", "acc_norm"), ("winogrande", "acc")]
ZS_COND = [("Softmax", "baseline"), ("Mean-threshold support", "mean_sparse_softmax"), ("Exp. $K=32", "zero_exp_K32_R4"),
           ("Exp. $K=64", "zero_exp_K64_R4"), ("Rowmax-PoT", "zero_rowmax_pot_K20")]


def tableC2():
    T = "table_C2_zero_shot.tex"
    for alias in ("qwen72", "llama70", "qwen15"):
        p = (RF / alias / "T12" / "zero_shot_results.csv") if alias == "qwen15" else \
            (RL / alias / "T12" / "zero_shot_results.csv")
        z = pd.read_csv(p).set_index(["condition", "task", "metric"]).value
        for key, cond in ZS_COND:
            vals = {}
            for j, (task, met) in enumerate(ZS, start=1):
                v, b = z[(cond, task, met)], z[("baseline", task, met)]
                vals[j] = [100 * v, 100 * (v - b)]
            if alias != "qwen15" and cond != "baseline":
                vals[7] = [100 * (z[(cond, "lambada_openai", "perplexity")] / z[("baseline", "lambada_openai", "perplexity")] - 1)]
            check_row(T, norm(key).replace("\\ ", " "), vals, str(p.relative_to(RES)), after=f"\\textbf{{{DISPLAY[alias]}}}")


# ----------------------------------------------------------------------------------------------------------------
# Tables D1 (allocation contrast) and D2 (matched-distortion direction)
# ----------------------------------------------------------------------------------------------------------------
def tableD1():
    T = "table_D1_allocation.tex"
    summ = pd.read_csv(RF / "crossmodel_summary.csv").set_index("alias")
    pos = res = 0
    for alias, name in DISPLAY.items():
        dr = bci(alias, "alloc_K21_R1", "alloc_K21_R4")
        pos += dr[0] > 0; res += dr[1] > 0 or dr[2] < 0
        vals = {1: [summ.loc[alias, "parameter_count"] / 1e9], 2: dr, 3: ["yes" if (dr[1] > 0 or dr[2] < 0) else "no"]}
        have = (RF / alias / "condition_blocks" / "alloc_K16_R1__L2048.csv").exists()
        if have:
            d16 = delta(alias, "alloc_K16_R1", "alloc_K16_R4"); d32 = delta(alias, "alloc_K32_R1", "alloc_K32_R4")
            vals[4] = [d16.mean()]; vals[5] = [d32.mean()]; vals[6] = list(boot(d16 - d32))
        check_row(T, name, vals, f"characterization/{alias}/condition_blocks/alloc_K{{16,21,32}}_R{{1,4}}")
    check_after(T, "Point estimates positive in ", int(pos), "count over the ten D_R(21) estimates")
    check_after(T, "intervals exclude zero in ", int(res), "count over the ten D_R(21) intervals")


def tableD2():
    T = "table_D2_direction.tex"
    for alias, name in DISPLAY.items():
        j0 = bci(alias, "jsd_K0_Aminus", "jsd_K0_Aplus"); j10 = bci(alias, "jsd_10K0_Aminus", "jsd_10K0_Aplus")

        def mark(v):
            return "+" if v[1] > 0 else ("resolved negative" if v[2] < 0 else "unresolved")
        check_row(T, name, {1: j0, 2: [mark(j0)], 3: j10, 4: [mark(j10)]},
                  f"characterization/{alias}/condition_blocks/jsd_{{K0,10K0}}_{{Aminus,Aplus}}")
    for alias, lit in (("llama1", "[95\\% CI]: Llama-3.2-1B"), ("qwen15", "]; Qwen2.5-1.5B")):
        t0 = bci(alias, "tv_K0_Aminus", "tv_K0_Aplus"); t10 = bci(alias, "tv_10K0_Aminus", "tv_10K0_Aplus")
        for j, v in enumerate(t0 + t10):
            check_after(T, lit, v, f"characterization/{alias}/condition_blocks/tv_{{K0,10K0}}_{{Aminus,Aplus}}", ntok=j)
    for j, v in enumerate(bci("qwen15", "jsd_3K0_Aminus", "jsd_3K0_Aplus")):
        check_after(T, "Qwen2.5-1.5B 3J$_0$ contrast: ", v, "characterization/qwen15/condition_blocks/jsd_3K0_{Aminus,Aplus}", ntok=j)


def main():
    for f in (table2, table3, tableB1, tableB2, table_expabl, tableB5, tableC1, tableC2, tableD1, tableD2):
        f()
    rep = pd.DataFrame(REPORT)
    rep.to_csv(TAB / "check_report.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(rep.groupby("table").ok.agg(["sum", "count"]).rename(columns={"sum": "pass", "count": "checked"}).to_string())
    bad = rep[~rep.ok]
    print(f"\n{int(rep.ok.sum())}/{len(rep)} checks pass")
    if len(bad):
        print("\nFAILURES:")
        print(bad.to_string(max_colwidth=70))
        sys.exit(1)


if __name__ == "__main__":
    main()
