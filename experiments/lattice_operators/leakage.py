"""Leakage-phase probe of Section 5.1 / Appendix B (Tables B1 and B2).

For each probed condition the high-mass probe of the characterization pipeline (App. A.8; first four aligned
blocks, rows with at least 64 valid keys, S_0.9 recomputed from the softmax of the current trajectory at every
layer and row) is run with a row-level export: the signed leakage L_0.9 = p(S_0.9) - p_hat(S_0.9) of every row
together with the lattice phase of the row maximum that an absolute lattice would see:

  r_oct  = floor(sigma_max + 1/2) - sigma_max                in [-1/2, 1/2] octave   (octave score lattice)
  r_half = (floor(2 sigma_max + 1/2) - 2 sigma_max) / 2      in [-1/4, 1/4]          (half-octave score lattice)
  r_p2   = x - round(x),  x = -log2 max_j p_j                                      (log2-p probability grid)
  r_ps2  = (y - round(y)) / 2,  y = -2 log2 max_j p_j                              (log-sqrt2-p probability grid)
  r_h15  = log2 w_argmax - (sigma_max - M/2),  M = 2 ceil(floor(2 sigma_max + 1/2) / 2)   ({1,1.5} x 2^k lattice)

with sigma = s log2(e).  The leakage-phase Pearson r of Tables B1/B2 is the correlation between L_0.9 and the
phase of the lattice the operator actually uses (PHASE_OF).  A row-maximum anchor has zero phase by
construction; its correlation with r_oct is reported as a control.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from softmax_approx import characterization as cm

LEAK_Q = 0.9
LOG2E = 1.4426950408889634

# phase of the lattice used by each operator (rowmax anchors: control against the absolute octave phase)
PHASE_OF = {
    "ilpot_v1": "r_oct", "pot_m1": "r_oct",
    "ilpot_ha_v1": "r_half",
    "pq_log2_b8": "r_p2", "pq_logsqrt2_b8": "r_ps2",
    "ilpot_h15_v1": "r_h15", "h15tiled": "r_h15", "h15rowmax": "r_h15", "h15tiled_bf16": "r_h15",
}
ROWMAX_ANCHORED = {"pot_m1", "h15tiled", "h15rowmax", "h15tiled_bf16"}


class PhaseStash:
    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.r_oct = self.r_half = self.r_p2 = self.r_ps2 = self.r_h15 = None
        self.layer = -1
        self.q_start = 0
        self.cursor = 0
        self.block = -1


STASH = PhaseStash()


def install_phase_stash(attention_module) -> None:
    """Wrap _alt_probs (inside the pipeline's probe patch) to record the per-row phases; read-only."""
    if getattr(attention_module, "_phase_stash_installed", False):
        return
    inner = attention_module._alt_probs

    def wrapped(scores, valid, mode, q_len, layer_idx=-1, q_start=0, offset_full=0):
        if STASH.enabled:
            with torch.no_grad():
                sig = (scores.float() * LOG2E).masked_fill(~valid, float("-inf"))
                smax = sig.amax(-1)
                STASH.r_oct = torch.floor(smax + 0.5) - smax
                STASH.r_half = (torch.floor(2.0 * smax + 0.5) - 2.0 * smax) / 2.0
                pmax = torch.softmax(scores.float().masked_fill(~valid, float("-inf")), -1).amax(-1)
                x = -torch.log2(pmax)
                STASH.r_p2 = x - torch.round(x)
                y = 2.0 * x
                STASH.r_ps2 = (y - torch.round(y)) / 2.0
                n_max = torch.floor(2.0 * smax + 0.5)
                M = 2.0 * torch.ceil(n_max / 2.0)
                lw = torch.where(n_max == M, torch.zeros_like(smax), torch.full_like(smax, math.log2(0.75)))
                STASH.r_h15 = lw - (smax - M / 2.0)
            if int(layer_idx) == 0 and int(q_start) == 0:
                STASH.block += 1
            STASH.layer = int(layer_idx); STASH.q_start = int(q_start); STASH.cursor = 0
        return inner(scores, valid, mode, q_len, layer_idx, q_start, offset_full)

    attention_module._alt_probs = wrapped
    attention_module._phase_stash_installed = True


_ORIG_P4 = cm.P4Accumulator
LAST_ACC = {}


class RowLeakAccumulator(_ORIG_P4):
    """The pipeline's P4Accumulator (aggregates unchanged) plus a per-row export of L_0.9 and the phases."""

    def __init__(self, qs, primary_q, min_n):
        super().__init__(qs, primary_q, min_n)
        self.cols = defaultdict(list)
        LAST_ACC["acc"] = self

    def add_mass(self, layer, q, keep, pm, hm, frac):
        super().add_mass(layer, q, keep, pm, hm, frac)
        if abs(float(q) - LEAK_Q) > 1e-12:
            return
        if STASH.layer != int(layer):
            raise RuntimeError(f"phase stash out of sync: stash layer {STASH.layer} vs {layer}")
        hc = pm.shape[1]; hs = STASH.cursor; STASH.cursor += hc
        if not bool(keep.any()):
            return
        b, _, Q = pm.shape
        dev = pm.device
        head = torch.arange(hs, hs + hc, device=dev).view(1, hc, 1).expand(b, hc, Q)
        qpos = (torch.arange(Q, device=dev) + STASH.q_start).view(1, 1, Q).expand(b, hc, Q)
        leak = pm.double() - hm.double()
        c = self.cols
        c["leak"].append(leak[keep].float().cpu())
        c["ref_mass"].append(pm[keep].float().cpu())
        for key in ("r_oct", "r_half", "r_p2", "r_ps2", "r_h15"):
            c[key].append(getattr(STASH, key)[:, hs:hs + hc][keep].float().cpu())
        c["head"].append(head[keep].to(torch.int16).cpu())
        c["qpos"].append(qpos[keep].to(torch.int16).cpu())
        n = int(keep.sum())
        c["layer"].append(torch.full((n,), int(layer), dtype=torch.int16))
        c["block"].append(torch.full((n,), STASH.block, dtype=torch.int16))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: torch.cat(v).numpy() for k, v in self.cols.items()}


def run_leak_probe(runner, spec: cm.ConditionSpec, out_dir: Path) -> dict:
    """Run the probe for one condition; writes leak_rows_<id>.npz and per-layer aggregates to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cm.P4Accumulator = RowLeakAccumulator
    STASH.reset(); STASH.enabled = True
    try:
        t0 = time.time()
        mass, out, _, agg = runner.p4_probe_condition(spec, [LEAK_Q], rescue=False)
        secs = time.time() - t0
    finally:
        cm.P4Accumulator = _ORIG_P4
        STASH.enabled = False
    acc = LAST_ACC.pop("acc")
    arr = acc.arrays()
    n_expected = int(agg.get(f"q{LEAK_Q:g}_n_rows", 0))
    if len(arr["leak"]) != n_expected:
        raise RuntimeError(f"row export {len(arr['leak'])} != aggregate rows {n_expected}")
    if STASH.block + 1 != int(runner.args.p4_probe_windows):
        raise RuntimeError(f"block counter {STASH.block + 1} != probe windows")
    np.savez_compressed(out_dir / f"leak_rows_{spec.id}.npz", **arr)
    cm.write_csv(out_dir / f"leak_mass_by_layer_{spec.id}.csv", mass)
    cm.write_csv(out_dir / f"leak_output_by_layer_{spec.id}.csv", out)
    cm.write_json(out_dir / f"leak_probe_{spec.id}.json", dict(
        condition=spec.serializable(), aggregate=agg,
        protocol=dict(q=LEAK_Q, p4_probe_windows=int(runner.args.p4_probe_windows),
                      p4_min_n=int(runner.args.p4_min_n), p4_head_chunk=int(runner.args.p4_head_chunk),
                      definition="L_q = p(S_q) - p_hat(S_q); S_q from the softmax of the current trajectory"),
    ))
    print(f"[leakage] {spec.id}: rows={len(arr['leak'])} mean L={arr['leak'].mean():+.3e} ({secs:.0f}s)")
    return arr


def leak_stats(arr: dict[str, np.ndarray], phase_key: str) -> tuple[dict, list]:
    """Summary of L_0.9 and its correlation with one lattice phase (pooled, per block, per layer)."""
    L = arr["leak"].astype(np.float64)
    r = arr[phase_key].astype(np.float64)
    out = dict(n_rows=int(L.size), mean_L=float(L.mean()), mean_abs_L=float(np.abs(L).mean()),
               rms_L=float(np.sqrt((L ** 2).mean())), positive_fraction=float((L > 0).mean()), phase=phase_key)
    pr, pp = stats.pearsonr(r, L)
    sr, sp = stats.spearmanr(r, L)
    per_layer = []
    for li in sorted(np.unique(arr["layer"])):
        m = arr["layer"] == li
        rr, p = stats.pearsonr(r[m], L[m])
        per_layer.append((int(li), float(rr), float(p), float(np.abs(L[m]).mean())))
    per_block = []
    for blk in sorted(np.unique(arr["block"])):
        m = arr["block"] == blk
        per_block.append(float(stats.pearsonr(r[m], L[m])[0]))
    out.update(pearson_r=float(pr), pearson_p=float(pp), spearman_rho=float(sr), spearman_p=float(sp),
               ols_slope_L_per_octave=float(np.polyfit(r, L, 1)[0]), per_block_pearson=per_block,
               layers_negative=int(sum(1 for _, rr, _, _ in per_layer if rr < 0)), n_layers=len(per_layer))
    return out, per_layer


def summarize_leakage(leak_dir: Path) -> pd.DataFrame | None:
    """leakage_summary.csv from the leak_rows_<id>.npz files of a run (None if there are none)."""
    files = {cid: leak_dir / f"leak_rows_{cid}.npz" for cid in PHASE_OF}
    data = {cid: dict(np.load(p)) for cid, p in files.items() if p.exists()}
    if not data:
        return None
    pot_abs = float(np.abs(data["pot_m1"]["leak"].astype(np.float64)).mean()) if "pot_m1" in data else None
    rows = []
    for cid, arr in data.items():
        st, _ = leak_stats(arr, PHASE_OF[cid])
        st = dict(condition=cid, rowmax_anchored=cid in ROWMAX_ANCHORED, **st)
        if pot_abs:
            st["mean_abs_L_ratio_to_pot_m1"] = st["mean_abs_L"] / pot_abs
        rows.append(st)
    return pd.DataFrame(rows)
