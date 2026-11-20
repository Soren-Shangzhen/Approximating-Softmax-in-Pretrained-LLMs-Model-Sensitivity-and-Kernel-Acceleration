"""Shared loaders, statistics, plot layout and number registry of the paper's figure scripts.

* Every plotted number is read from a result file under results/ at run time (no literal data arrays).
* The paired block bootstrap replicates softmax_approx.characterization.Bootstrapper exactly:
  idx = default_rng(seed).integers(0, n, size=(reps, n)), boots = x[idx].mean(axis=1), 2.5% / 97.5% quantiles,
  reps = 5000, seed = 0; the statistical unit is the aligned 2048-token block.
* rec() registers each plotted number with its source file and a recompute spec; set PAPER_NUMBER_REGISTRY=1 to
  write the registry to paper/figure_scripts/number_registry/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Paths (read-only inputs) and output roots
BASE = Path(__file__).resolve().parents[2]                                   # repository root
RESULTS = Path(os.environ.get("APPROX_SOFTMAX_RESULTS", BASE / "results"))   # or a directory of reruns
RESULTS_FORMAL = RESULTS / "characterization"
RESULTS_LATTICE = RESULTS / "lattice_operators"
RESULTS_LARGE = RESULTS / "characterization_large_T11_T12"
B200_RES = RESULTS / "kernel"
B200_EXT = B200_RES / "timing"
B200_PPL = B200_RES / "fidelity"

FIG_DIR = BASE / "paper" / "figures"
MANIFEST_PARTS = BASE / "paper" / "figure_scripts" / "number_registry"
WRITE_REGISTRY = os.environ.get("PAPER_NUMBER_REGISTRY", "0") == "1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# output files: paper figure numbering (None = panel set not used in the paper)
OUTPUT_NAMES = {
    "fig1_amount": "figure_1", "figS_amount_supp": "figure_S4", "fig2_allocation": "figure_2",
    "figS_allocation_supp": None, "fig3_structure": "figure_3", "figS_structure_supp": None,
    "fig4_h15_b200": "figure_4", "figS_seqsweep_supp": "figure_S2", "figS4_correctness_gates": "figure_S1",
    "fig7_longcontext": "figure_S3", "figS1_local_pv": "figure_S5", "fig5_geometry": "figure_S6",
    "figS3_rowmax_pot_crossmodel": "figure_S7",
}
# figure script -> key of its layout and panel titles
SCRIPT_STEMS = {'figure_1_S4_support_weighting_grid': 'fig2_amount', 'figure_2_allocation': 'fig3_allocation', 'figure_3_divergence': 'fig4_structure', 'figure_4_S2_rowmax_h15': 'fig6_h15_b200', 'figure_S1_correctness_gates': 'figS4_correctness_gates', 'figure_S3_long_context': 'fig7_longcontext', 'figure_S5_local_pv': 'figS1_local_pv', 'figure_S6_redistribution': 'fig5_geometry', 'figure_S7_rowmax_pot': 'figS3_rowmax_pot_crossmodel'}

BOOT_REPS = 5000
BOOT_SEED = 0


def rel(p: Path | str) -> str:
    """Repository-relative path string used in manifest source_file."""
    p = Path(p)
    for root in (BASE,):
        try:
            return str(p.relative_to(root))
        except ValueError:
            pass
    return str(p)


# Model roster (read from results/characterization/crossmodel_summary.csv, not hard-coded)
def model_roster() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_FORMAL / "crossmodel_summary.csv",
                     usecols=["model", "alias", "family", "parameter_count", "num_layers"])
    df["display"] = df["model"].str.split("/").str[-1]
    df["display"] = df["display"].replace({"gemma-2-2b": "Gemma-2-2B", "Meta-Llama-3.1-8B": "Llama-3.1-8B"})
    df["params_B"] = df["parameter_count"] / 1e9
    return df.sort_values("parameter_count").reset_index(drop=True)


ALL_MODELS = ["qwen05", "qwen15", "qwen3", "qwen72", "llama1", "llama3", "llama8", "llama70",
              "gemma2", "mistral7"]
DEEP_REF = "qwen15"

# Zero-shot metric set: acc for LAMBADA and WinoGrande, acc_norm for HellaSwag, ARC-Easy, ARC-Challenge, PIQA.
ZS_TASKS = ["lambada_openai", "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
ZS_METRIC = {"lambada_openai": "acc", "hellaswag": "acc_norm", "arc_easy": "acc_norm",
             "arc_challenge": "acc_norm", "piqa": "acc_norm", "winogrande": "acc"}
ZS_LABEL = {"lambada_openai": "LAMBADA", "hellaswag": "HellaSwag", "arc_easy": "ARC-e",
            "arc_challenge": "ARC-c", "piqa": "PIQA", "winogrande": "WinoGrande"}
ZS_SET_NOTE = "metric set: acc for LAMBADA and WinoGrande; acc_norm for HellaSwag, ARC-Easy, ARC-Challenge and PIQA"

# B200 speed-ups: pooled median over all rounds; the spread is the population sd (ddof=0) of the per-round speed-ups.
SD_DDOF = 0

# Display precision: dNLL and interval endpoints to 3 significant figures, rendered from the unrounded value.
def fmt_sig(v: float, n: int = 3, plus: bool = False) -> str:
    import decimal
    if v == 0:
        s = "0." + "0" * (n - 1)
    else:
        d = decimal.Decimal(repr(float(v)))
        q = d.quantize(decimal.Decimal(1).scaleb(d.adjusted() - n + 1), rounding=decimal.ROUND_HALF_EVEN)
        if q.adjusted() != d.adjusted():  # rounding carried into the next decade (e.g. 0.0009996 -> 0.00100)
            q = d.quantize(decimal.Decimal(1).scaleb(q.adjusted() - n + 1), rounding=decimal.ROUND_HALF_EVEN)
        s = format(q, "f")
    return ("+" + s) if (plus and not s.startswith("-")) else s


class SigFig:
    """Drop-in for a '{:...}' format string: fmt.format(v) renders `n` significant figures."""
    def __init__(self, n: int = 3, plus: bool = False):
        self.n, self.plus = n, plus
    def format(self, v):
        return fmt_sig(v, self.n, self.plus)
    def __str__(self):
        return f"{self.n} significant figures" + (" (signed)" if self.plus else "")


SIG3, SIG3P = SigFig(3), SigFig(3, plus=True)
NLL_KEY = re.compile(r"^(t01b\.aggregate\.(est|lo|hi)|t02\..*dnll|t04\.[^.]+\.(DR\d+|IR)\.(est|lo|hi)|t04\.[^.]+\.R[\d.]+\.dnll"
                     r"|t08\..*\.(est|lo|hi)|t09\..*\.(est|lo|hi)|t10\.[^.]+\.(rescued|Ggap)_[^.]+\.(est|lo|hi)"
                     r"|t11\.[^.]+\.L\d+\.(DR|DT)\.(est|lo|hi)|t11\.[^.]+\.L\d+\.R\d\.dnll"
                     r"|anchor\.[^.]+\.(est|lo|hi)|step2d\.[^.]+\.(est|lo|hi)|b200\.ppl\.dnll\.(est|lo|hi))$")


# Block-level loaders
def block_path(alias: str, condition: str, L: int = 2048, root: str = "formal") -> Path:
    if root == "formal":
        return RESULTS_FORMAL / alias / "condition_blocks" / f"{condition}__L{L}.csv"
    if root == "lattice":
        return RESULTS_LATTICE / alias / "condition_blocks" / f"{condition}__L{L}.csv"
    if root == "70b":
        return RESULTS_LARGE / alias / "condition_blocks" / f"{condition}__L{L}.csv"
    raise ValueError(root)


def load_blocks(alias: str, condition: str, L: int = 2048, root: str = "formal") -> pd.DataFrame:
    p = block_path(alias, condition, L, root)
    df = pd.read_csv(p)
    df = df.sort_values("block").reset_index(drop=True)
    df.attrs["path"] = p
    return df


def paired_delta(alias: str, cond_a: str, cond_b: str | None = None, L: int = 2048,
                 root_a: str = "formal", root_b: str | None = None) -> tuple[np.ndarray, list[Path]]:
    """Per-block delta_nll of cond_a, or per-block (delta_a - delta_b) aligned on block id."""
    a = load_blocks(alias, cond_a, L, root_a)
    paths = [a.attrs["path"]]
    if cond_b is None:
        return a["delta_nll"].to_numpy(float), paths
    b = load_blocks(alias, cond_b, L, root_b or root_a)
    paths.append(b.attrs["path"])
    m = a.merge(b, on="block", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), f"block misalignment {cond_a} vs {cond_b}"
    assert np.allclose(m["baseline_nll_a"], m["baseline_nll_b"]), "baseline mismatch across arms"
    return (m["delta_nll_a"] - m["delta_nll_b"]).to_numpy(float), paths


def boot_mean_ci(x, reps: int = BOOT_REPS, seed: int = BOOT_SEED, level: float = 0.95):
    """Exact replica of softmax_approx.characterization.Bootstrapper.mean_ci."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)), endpoint=False)
    boots = x[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(x.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


# B200 latency helpers
LAT_KEYS = ["dtype", "seqlen", "heads", "hd", "batch", "causal"]


def load_lat(name: str, sub: str = "ext") -> pd.DataFrame:
    p = (B200_EXT if sub == "ext" else B200_RES) / f"{name}.csv"
    df = pd.read_csv(p)
    df.attrs["path"] = p
    return df


def cell_median(df: pd.DataFrame, **cfg) -> tuple[float, int]:
    m = np.ones(len(df), bool)
    for k, v in cfg.items():
        if k in df.columns:
            m &= (df[k] == v).to_numpy()
    sub = df[m]
    return float(sub["latency_ms"].median()), int(len(sub))


def speedup_pct(a0: pd.DataFrame, a1: pd.DataFrame, **cfg) -> dict:
    m0, n0 = cell_median(a0, **cfg)
    m1, n1 = cell_median(a1, **cfg)
    return dict(a0_ms=m0, a1_ms=m1, n0=n0, n1=n1, speedup_pct=100.0 * (m0 / m1 - 1.0))


# Number manifest recorder
_MANIFEST: list[dict] = []
_RECOMPUTE: dict[str, dict] = {}


def rec(location: str, text_as_written: str, value, derivation: str, source_file, source_field: str,
        precision_shown: str, recompute: dict | None = None, rid: str | None = None) -> str:
    """Register one number with its source file and a machine-readable recompute spec."""
    if isinstance(source_file, (list, tuple)):
        source_file = " + ".join(rel(p) for p in source_file)
    else:
        source_file = rel(source_file)
    rid = rid or f"n{len(_MANIFEST):04d}_{re.sub(r'[^a-zA-Z0-9]+', '_', location)[:40]}"
    _MANIFEST.append(dict(id=rid, location=location, text_as_written=text_as_written,
                          value_numeric=float(value), derivation=derivation, source_file=source_file,
                          source_field=source_field, precision_shown=precision_shown))
    if recompute is not None:
        _RECOMPUTE[rid] = recompute
    return rid


def flush_manifest(part_name: str):
    if not WRITE_REGISTRY:
        _MANIFEST.clear(); _RECOMPUTE.clear()
        return
    MANIFEST_PARTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_MANIFEST).to_csv(MANIFEST_PARTS / f"{part_name}.csv", index=False)
    with open(MANIFEST_PARTS / f"{part_name}.recompute.json", "w") as f:
        json.dump(_RECOMPUTE, f, indent=1)
    _MANIFEST.clear()
    _RECOMPUTE.clear()


# Convenience recorders
def rec_boot(location: str, label: str, alias: str, cond_a: str, cond_b: str | None, L: int,
             root_a: str, root_b: str | None, fmt=SIG3P, part: str = "estimate"):
    """Record estimate / ci_low / ci_high of a paired-block bootstrap; returns (est, lo, hi)."""
    x, paths = paired_delta(alias, cond_a, cond_b, L, root_a, root_b)
    est, lo, hi = boot_mean_ci(x)
    spec = dict(kind="boot", alias=alias, cond_a=cond_a, cond_b=cond_b, L=L, root_a=root_a,
                root_b=root_b or root_a)
    field = (f"block-paired mean of delta_nll ({cond_a})" if cond_b is None
             else f"block-paired mean of delta_nll({cond_a}) - delta_nll({cond_b})")
    rec(location, fmt.format(est), est, "paired block bootstrap mean (reps=5000, seed=0)", paths,
        field + "; col: delta_nll", fmt, {**spec, "which": "est"}, rid=f"{label}__est")
    rec(location, fmt.format(lo), lo, "bootstrap 2.5% quantile", paths, field + "; col: delta_nll",
        fmt, {**spec, "which": "lo"}, rid=f"{label}__lo")
    rec(location, fmt.format(hi), hi, "bootstrap 97.5% quantile", paths, field + "; col: delta_nll",
        fmt, {**spec, "which": "hi"}, rid=f"{label}__hi")
    return est, lo, hi


def rec_direct(location: str, label: str, path: Path, row_filter: dict, col: str, fmt: str = "{:.6g}",
               text: str | None = None, scale: float = 1.0, derivation: str = "direct"):
    """Record a value read directly from a CSV cell (optionally scaled, e.g. ×100 for %)."""
    df = pd.read_csv(path)
    m = np.ones(len(df), bool)
    for k, v in row_filter.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    sub = df[m]
    assert len(sub) == 1, f"{path} filter {row_filter} matched {len(sub)} rows"
    v = float(sub.iloc[0][col]) * scale
    field = "row: " + "; ".join(f"{k}={vv}" for k, vv in row_filter.items()) + f"; col: {col}"
    if scale != 1.0:
        derivation = f"{derivation} x {scale:g}"
    rec(location, text or fmt.format(v), v, derivation, path, field, fmt,
        dict(kind="direct", path=str(path), row_filter=row_filter, col=col, scale=scale), rid=label)
    return v


def rec_speedup(location: str, label: str, a0_name: str, a1_name: str, cfg: dict, sub: str = "ext",
                fmt: str = "{:+.1f}%"):
    a0 = load_lat(a0_name, sub); a1 = load_lat(a1_name, sub)
    r = speedup_pct(a0, a1, **cfg)
    field = "median(latency_ms) over runs with " + "; ".join(f"{k}={v}" for k, v in cfg.items())
    rec(location, fmt.format(r["speedup_pct"]), r["speedup_pct"],
        "100*(median A0 latency / median A1 latency - 1)", [a0.attrs["path"], a1.attrs["path"]],
        field, fmt, dict(kind="speedup", a0=a0_name, a1=a1_name, sub=sub, cfg=cfg), rid=label)
    return r


# Plot style
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paperstyle as ps
    ps.apply()
    plt.rcParams.update({"hatch.linewidth": 0.5, "lines.linewidth": 1.0, "lines.markersize": 4})
    return plt


# Okabe–Ito colour-blind-safe palette
C = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
         blue="#0072B2", vermil="#D55E00", purple="#CC79A7", grey="#7f7f7f")
SINGLE_W, DOUBLE_W = 3.4, 5.5


def _lift_titles_above_legends(fig):
    """v12: if a panel title/label overlaps a legend anchored above the axes, move both above the legend."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = getattr(ax, "_v12_panel_texts", [])
        leg = ax.get_legend()
        if not texts or leg is None:
            continue
        lb = leg.get_window_extent(r)
        if not any(t.get_window_extent(r).overlaps(lb) for t in texts):
            continue
        top = ax.transAxes.inverted().transform((0, lb.y1))[1]
        for t in texts:
            t.set_y(top + 0.02)


def save(fig, name: str):
    import paperstyle as ps
    name = OUTPUT_NAMES[name]
    ps.save(fig, FIG_DIR / f"{name}.pdf")  # lint + exact physical size (no bbox tightening)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    print(f"saved {name}.pdf/.png")


PANEL_TITLES = {  # short panel titles, by figure script
    "fig2_amount": {"A": "Support", "B": "Same support", "C": "Weighting geometry vs. K", "D": "Common K=32, R=4"},
    "fig3_allocation": {"A": "Mass concentration", "B": "R sweep", "C": "Error vs. loss", "D": "Allocation contrast", "E": "Precision x allocation"},
    "fig4_structure": {"A": "JSD response map", "B": "Matched-JSD contrast", "C": "Matched-TV check"},
    "fig5_geometry": {"A": "Signed leakage", "B": "Local pV control", "C": "Mass-partition rescue", "D": "Allocation-linked leakage"},
    "fig6_h15_b200": {"A": "Anchor and resolution", "B": "Fidelity at 2K", "C": "Session 1 sweep", "D": "Speed-up by dtype and mask", "E": "FA4 tuning control", "F": "Session 3 traced run"},
    "fig7_longcontext": {"A": "Within-context allocation", "B": "Within-context direction"},
    "figS1_local_pv": {"A": "Local pV change by layer", "B": "Local pV cosine by layer"},
    "figS2_pot_audit": {"A": "Octave-rounding bound", "B": "Mass and entries vs. distance"},
    "figS3_rowmax_pot_crossmodel": {"A": "Common K_max = 20", "B": "Deep-reference K_max sweep"},
    "figS4_correctness_gates": {"A": "Single-tile gate", "B": "Multi-tile gate"},
}


def panel_label(ax, s: str):
    import __main__, os
    import paperstyle as ps
    shown = getattr(ax, "_v13_letter", s)
    stem = os.path.splitext(os.path.basename(getattr(__main__, "__file__", "")))[0]
    stem = SCRIPT_STEMS.get(stem, stem)
    title = PANEL_TITLES.get(stem, {}).get(s) or ""
    ps.panel_label(ax, shown, title)  # bold letter + title as a left-aligned axes title (layout-aware)
    ax._v12_panel_texts = [ax.title]


# Figure layout: built at the text width (5.5 in), every text element >= 7 pt; some panels of a script go
# to a supplementary figure.
TEXT_W = 5.5
MIN_FONT = 7.0

V13_LAYOUTS = {
    # script stem: list of (output name, panels in order, grid spec)
    "fig2_amount": [("fig1_amount", "ABD", dict(shape=(1, 3), size=(TEXT_W, 1.90), width_ratios=[1, 1, 1], wspace=0.4, min_font=8)),
                    ("figS_amount_supp", "C", dict(shape=(1, 1), size=(3.2, 2.3)))],  # panel C moved to the appendix
    "fig3_allocation": [("fig2_allocation", "BCD", dict(shape=(1, 3), size=(TEXT_W, 2.12), width_ratios=[1, 1, 1], wspace=0.4, min_font=8)),
                        ("figS_allocation_supp", "AE", dict(shape=(1, 2), size=(TEXT_W, 2.4), width_ratios=[1, 1.2], wspace=0.35))],
    "fig4_structure": [("fig3_structure", "AB", dict(shape=(1, 2), size=(TEXT_W, 1.95), width_ratios=[1, 1.3], wspace=0.3, min_font=8)),
                       ("figS_structure_supp", "C", dict(shape=(1, 1), size=(3.4, 2.4)))],
    "fig5_geometry": [("fig5_geometry", "ABCD", dict(shape=(4, 1), size=(TEXT_W, 6.6), hspace=0.3, sharex=True))],  # shared model axis, labels on the bottom panel only  # appendix figure, stacked  # shared model axis, labels on the bottom panel only  # appendix figure, stacked
    # main figure = A anchor, B fidelity, D dtype x mask (shown as C), E FA4 tuning control (shown as D)
    "fig6_h15_b200": [("fig4_h15_b200", "ABDE", dict(shape=(2, 2), size=(TEXT_W, 2.18), wspace=0.5, hspace=0.45, width_ratios=[0.86, 1.14], min_font=8)),  # B carries seven categories
                      ("figS_seqsweep_supp", "CF", dict(shape=(1, 2), size=(5.5, 2.5), wspace=0.32, min_font=7))],
}


def v13_axes(stem: str):
    """Create the v13 figures for `stem`; return {original panel letter: axes}."""
    import matplotlib.pyplot as plt
    axes = {}
    _V13_FIGS.clear()
    for out, panels, g in V13_LAYOUTS[stem]:
        nr, nc = g["shape"]
        fig = plt.figure(figsize=g["size"], layout="constrained")  # final print size, constrained layout
        gs = fig.add_gridspec(nr, nc, wspace=g.get("wspace", 0.4) * 0.25, hspace=g.get("hspace", 0.6) * 0.25,
                              width_ratios=g.get("width_ratios"), height_ratios=g.get("height_ratios"))
        shown = 0; first = None
        for i, letter in enumerate(panels):
            if letter == "_":  # empty spacer slot (room for long tick labels of the next panel)
                continue
            ax = fig.add_subplot(gs[i // nc, i % nc], sharex=first if g.get("sharex") else None)  # optional shared x
            first = first or ax
            ax._v13_letter = "ABCDEFGH"[shown] if len(panels.replace("_", "")) > 1 else ""
            shown += 1
            axes[letter] = ax
        fig._v13_min_font = g.get("min_font", MIN_FONT)
        _V13_FIGS.append((out, fig))
    return axes


_V13_FIGS = []


def _enforce_min_font(fig):
    import matplotlib.text as mtext
    floor = getattr(fig, "_v13_min_font", MIN_FONT)  # main-text figures use 8 pt
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < floor:
            t.set_fontsize(floor)


def v13_save():
    import paperstyle as ps
    for out, fig in _V13_FIGS:
        out = OUTPUT_NAMES[out]
        if out is None:          # supplementary panel set that is not a figure of the paper
            continue
        ps.save(fig, FIG_DIR / f"{out}.pdf")  # lint + exact physical size
        fig.savefig(FIG_DIR / f"{out}.png", dpi=150)
        print(f"saved {out}.pdf/.png")
