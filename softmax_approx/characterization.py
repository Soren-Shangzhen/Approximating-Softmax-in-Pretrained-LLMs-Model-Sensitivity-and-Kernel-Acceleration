#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization pipeline of Section 4 (Appendices A, C and D): tasks T00-T13 for one model.

Paper: "Approximating Softmax in Pretrained LLMs: Model Sensitivity and Kernel Acceleration" (arXiv:2609.33586).

One code path evaluates every condition on the frozen model: per-block NLL on the 97 aligned 2048-token
WikiText-103 blocks against the softmax baseline recomputed through the same attention backend,
vocabulary-output fidelity (KL, JSD, top-1) and paired block-bootstrap intervals (5,000 replicates, seed 0).

Tasks (dependencies are added automatically):
  T00  engineering gates: query chunking, eager-implementation agreement, padding, quantizer mirror,
       probe identity                                                            (App. A.2)
  T01  support sweep (top-k, mean threshold) with softmax or uniform weights     (Sec. 4.1, Fig. 1A, App. D.1)
  T01B strict single-layer same-support control D_W(l)                           (Sec. 4.1, Fig. 1B, App. A.3)
  T02  weighting geometry: exponential vs linear level-to-weight map             (Fig. S4, App. D.1)
  T03  full-range precision grid (K=32, R=4 across models)                       (Fig. 1C, Table 3, App. D.1)
  T04  allocation: R sweep at K=21, D_R(K), I_R, error geometry, mass concentration
                                                                                 (Sec. 4.2, Fig. 2, Table D1)
  T05  reconstruction: upper edge, nearest boundary, weight-domain interpolation (Sec. 4.4, App. A.5)
  T06  PoT perturbation and its error audit, J0, Rowmax-PoT K_max sweep          (Sec. 4.4, Fig. S7, App. A.6)
  T07  distortion-response map (56 interventions)                                (Fig. 3A, App. D.3)
  T08  matched-JSD flattening/sharpening contrasts D_T(J0), D_T(10 J0)           (Sec. 4.3, Fig. 3B, Table D2)
  T09  matched-TV robustness check                                               (App. D.3)
  T10  high-mass leakage, local pV output, mass-partition rescue G_gap           (Sec. 4.3, Figs. S5-S6, Table D3)
  T11  long context (2K/8K/16K): within-context D_R and D_T                       (App. C.2, Fig. S3, Table C1)
  T12  zero-shot lm-evaluation-harness tasks                                     (App. C.3, Table C2)
  T13  vocabulary-output fidelity                                                (App. C.4)

Model profiles (MODEL_REGISTRY / PROFILE_TASKS): deep_reference for Qwen2.5-1.5B (all tasks); core for
Qwen2.5-0.5B/3B, Llama-3.2-1B/3B and Gemma-2-2B; modern_core for Llama-3.1-8B and Mistral-7B-v0.3.  The 70B/72B
models use softmax_approx.multigpu.characterization (same tasks, multi-GPU placement).

Usage (from the repository root):

    python -m softmax_approx.characterization --model qwen15 --dry-run        # print the task plan
    python -m softmax_approx.characterization --model qwen15 --out-root runs/characterization --resume
    python -m softmax_approx.characterization --model llama1 --tasks T10 --out-root runs/characterization --resume

Naming in the code: BTA is the intervention attention backend (softmax_approx/attention.py); K0 in condition ids
is the matched-divergence reference level J0 of the paper (jsd_10K0_Aminus = flattening matched to 10 J0); P4 is
the high-mass redistribution probe (leakage L_q, local pV output, mass-partition rescue; App. A.8).

Notes
-----
* The score path is FP32 QK^T.  NLL and vocabulary-distribution metrics are computed from final hidden states
  with a chunked LM head, avoiding an L x vocab logits allocation.
* Matching is divergence-only: each alpha is solved on calibration blocks, frozen, then evaluated downstream.
* P4's high-mass set is always defined from the softmax of the current condition trajectory, never from a
  cached clean-model token set.
* q_chunk is a memory device of the simulation, not a property of any method; no speed or memory claim is
  made by this pipeline.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# 0. Frozen scientific configuration
# =============================================================================

LN2 = math.log(2.0)
CROSSMODEL_RELEASE = "2026-08-20-v2"
FORWARD_CACHE_SCHEMA = "crossmodel-forward-v2"

# Ideal PoT has |eps| <= ln(2)/(2m) in real arithmetic.  The implementation
# intentionally mirrors fp32 score/rounding arithmetic, so an empirical maximum
# over tens of millions of entries can exceed the ideal bound by a few 1e-5 nats
# from floating-point roundoff/cancellation.  Keep the theorem audit strict while
# allowing this tiny implementation-level excess.  This does NOT change any
# forward semantics or cached model outputs.
POT_FP32_BOUND_ABS_TOL = 5e-5


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "qwen05": dict(
        hf="Qwen/Qwen2.5-0.5B",
        family="Qwen",
        approx_params="0.5B",
        profile="core",
        extras=["allocation_extended"],
    ),
    "llama1": dict(
        hf="meta-llama/Llama-3.2-1B",
        family="Llama",
        approx_params="1.2B",
        profile="core",
        extras=["tv_robustness"],
    ),
    "qwen15": dict(
        hf="Qwen/Qwen2.5-1.5B",
        family="Qwen",
        approx_params="1.5B",
        profile="deep_reference",
        extras=[],
    ),
    "gemma2": dict(
        hf="google/gemma-2-2b",
        family="Gemma",
        approx_params="2B",
        profile="core",
        extras=[],
    ),
    "qwen3": dict(
        hf="Qwen/Qwen2.5-3B",
        family="Qwen",
        approx_params="3B",
        profile="core",
        extras=["allocation_extended"],
    ),
    "llama3": dict(
        hf="meta-llama/Llama-3.2-3B",
        family="Llama",
        approx_params="3.2B",
        profile="core",
        extras=["longctx_8k"],
    ),
    "llama8": dict(
        hf="meta-llama/Meta-Llama-3.1-8B",
        family="Llama",
        approx_params="8B",
        profile="modern_core",
        extras=[],
    ),
    "mistral7": dict(
        hf="mistralai/Mistral-7B-v0.3",
        family="Mistral",
        approx_params="7B",
        profile="modern_core",
        extras=[],
    ),
    "qwen7": dict(
        hf="Qwen/Qwen2.5-7B",
        family="Qwen",
        approx_params="7B",
        profile="modern_core",
        extras=[],
    ),
}

PROFILE_TASKS: dict[str, list[str]] = {
    "deep_reference": [
        "T00", "T01", "T01B", "T02", "T03", "T04", "T05", "T06",
        "T07", "T08", "T09", "T10", "T11", "T12", "T13",
    ],
    "core": ["T00", "T01", "T02", "T03", "T04", "T06", "T08", "T10", "T13"],
    "modern_core": ["T00", "T03", "T04", "T06", "T08", "T10", "T13"],
}

# Task dependencies are scientific dependencies, not merely import dependencies.
TASK_DEPS: dict[str, list[str]] = {
    "T00": [],
    "T01": ["T00"],
    "T01B": ["T00"],
    "T02": ["T00"],
    "T03": ["T00"],
    "T04": ["T00"],
    "T05": ["T00"],
    "T06": ["T00"],
    "T07": ["T00", "T04", "T05", "T06"],
    "T08": ["T00", "T06"],
    "T09": ["T00", "T06", "T08"],
    "T10": ["T00", "T04", "T06", "T08"],
    "T11": ["T00", "T04", "T06", "T08"],
    "T12": ["T00", "T01", "T03", "T06"],
    "T13": ["T00"],
}

# Deep-reference T07 keeps the historical deterministic 56-point membership.
# The new R<1 anti-allocation points belong to T04 and are intentionally not
# injected post hoc into this fixed response-map membership.
T07_POT_M = [
    1.0, 0.5, 2.0, 4.0,
    0.1826, 0.3162, 0.5773,
    0.1641, 0.3047,
    0.5723609436, 0.3032529266,
]
T07_TEMP_FLAT = [
    0.9, 0.95, 0.98, 0.97, 0.99, 0.5, 0.75,
    0.6146, 0.7775, 0.8781, 0.9296,
    0.6574, 0.7931, 0.8822,
    0.93, 0.8819982377, 0.7921182333,
]
T07_TEMP_SHARP = [
    1.1, 1.02, 1.05, 1.01, 1.03, 1.25, 1.5, 2.0,
    1.0742, 1.1285, 1.2346, 1.4063,
    1.1345, 1.2645, 1.5348,
    1.075, 1.13514649, 1.268321296,
]


# =============================================================================
# 1. Utility helpers
# =============================================================================


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_")
    return s or "unnamed"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def short_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def file_sha256(path: str | Path) -> str | None:
    try:
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20),b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def python_ast_sha256(path: str | Path) -> str | None:
    """Hash Python semantics while ignoring comments/whitespace."""
    try:
        text=Path(path).read_text(encoding="utf-8")
        tree=ast.parse(text, filename=str(path))
        payload=ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_str(s: str | None) -> list[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_csv_int(s: str | None) -> list[int]:
    return [int(x) for x in parse_csv_str(s)]


def parse_csv_float(s: str | None) -> list[float]:
    return [float(x) for x in parse_csv_str(s)]


def mean_or_nan(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    return float(np.nanmean(a)) if a.size else float("nan")


def exp_safe(x: float) -> float:
    try:
        return float(math.exp(x))
    except OverflowError:
        return float("inf")


def model_param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def resolve_model(name_or_alias: str) -> tuple[str, str | None, dict[str, Any]]:
    if name_or_alias in MODEL_REGISTRY:
        info = dict(MODEL_REGISTRY[name_or_alias])
        return info["hf"], name_or_alias, info
    # Raw HF repo. Default to core; user may override --profile.
    return name_or_alias, None, dict(
        hf=name_or_alias,
        family="unknown",
        approx_params="unknown",
        profile="core",
        extras=[],
    )


def resolve_task_order(tasks: Iterable[str]) -> list[str]:
    requested = [t.upper() for t in tasks]
    bad = sorted(set(requested) - set(TASK_DEPS))
    if bad:
        raise ValueError(f"unknown task ids: {bad}")
    seen: set[str] = set()
    out: list[str] = []

    def add(t: str):
        if t in seen:
            return
        for d in TASK_DEPS[t]:
            add(d)
        seen.add(t)
        out.append(t)

    for t in requested:
        add(t)
    return out


class Bootstrapper:
    """Shared paired block bootstrap indices, cached by (n,reps,seed)."""

    _cache: dict[tuple[int, int, int], np.ndarray] = {}

    @classmethod
    def indices(cls, n: int, reps: int, seed: int) -> np.ndarray:
        key = (int(n), int(reps), int(seed))
        if key not in cls._cache:
            rng = np.random.default_rng(seed)
            cls._cache[key] = rng.integers(0, n, size=(reps, n), endpoint=False)
        return cls._cache[key]

    @classmethod
    def mean_ci(cls, x: np.ndarray, reps: int, seed: int, level: float = 0.95):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            return float("nan"), float("nan"), float("nan")
        idx = cls.indices(len(x), reps, seed)
        boots = x[idx].mean(axis=1)
        a = (1.0 - level) / 2.0
        return float(x.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1-a))


# =============================================================================
# 2. Condition specification and BTA configuration
# =============================================================================


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    operator: str
    params: dict[str, Any] = field(default_factory=dict)
    role: str = ""
    family: str = ""
    rescue_q: float | None = None

    def serializable(self) -> dict[str, Any]:
        """Full descriptive condition record used in manifests/tables."""
        return dict(
            id=self.id,
            operator=self.operator,
            params=self.params,
            role=self.role,
            family=self.family,
            rescue_q=self.rescue_q,
        )

    def forward_serializable(self) -> dict[str, Any]:
        """Only fields that can change the model forward pass.

        ``role`` and ``family`` are analysis annotations.  The same numerical
        condition is intentionally reused by multiple tasks under different
        roles (e.g. T04 allocation vs effect calculation, T06 PoT anchor vs
        T07 response-map membership).  Including those annotations in the
        expensive condition-cache signature makes a valid cache appear stale
        and can cause downstream ``missing blocks`` failures.
        """
        return dict(
            id=self.id,
            operator=self.operator,
            params=self.params,
            rescue_q=self.rescue_q,
        )


def c_baseline() -> ConditionSpec:
    return ConditionSpec("baseline", "baseline", role="baseline")


def c_binary(soft: bool = True) -> ConditionSpec:
    return ConditionSpec(
        "mean_sparse_softmax" if soft else "mean_sparse_uniform",
        "binary_soft" if soft else "binary",
        role="support",
        family="support",
    )


def c_strict_same_support(layer: int, soft: bool = True) -> ConditionSpec:
    """Single-layer mean-threshold intervention for the strict same-support control.

    Only ``target_layer`` is perturbed.  All earlier layers therefore follow the
    exact baseline trajectory in both arms, so the score tensor and deterministic
    mean-threshold mask at the target layer are identical by construction.  The
    two arms differ only in weighting *within* that mask: learned relative
    softmax weights (``binary_soft``) versus uniform weights (``binary``).
    """
    layer = int(layer)
    return ConditionSpec(
        f"strict_mean_L{layer:02d}_{'softmax' if soft else 'uniform'}",
        "strict_binary_soft" if soft else "strict_binary_uniform",
        params=dict(target_layer=layer),
        role="strict_same_support_weighting",
        family="support_weighting_control",
    )


def c_topk(ratio: float, soft: bool = True) -> ConditionSpec:
    pct = f"{100*ratio:g}".replace(".", "p")
    return ConditionSpec(
        f"topk{pct}_{'softmax' if soft else 'uniform'}",
        "topk_soft" if soft else "topk",
        params=dict(topk_ratio=float(ratio)),
        role="support_sweep",
        family="support",
    )


def c_multibit(K: int, R: float = 1.0, reconstruction: str = "nearest",
               level_weight: str = "exp", *, role: str = "",
               cid: str | None = None) -> ConditionSpec:
    if cid is None:
        cid = f"mb_{level_weight}_K{K}_R{R:g}_{reconstruction}"
    return ConditionSpec(
        slugify(cid), "multibit",
        params=dict(
            K=int(K), R=float(R), reconstruction=str(reconstruction),
            level_weight=str(level_weight), threshold_mode="none", coverage="max",
        ),
        role=role,
        family="exp_quant" if level_weight == "exp" else f"{level_weight}_quant",
    )


def c_temp(alpha: float, *, role: str = "", cid: str | None = None,
           rescue_q: float | None = None) -> ConditionSpec:
    fam = "temperature_flatten" if alpha < 1 else (
        "temperature_sharpen" if alpha > 1 else "temperature_identity")
    if cid is None:
        cid = f"temp_a{alpha:.10g}"
    return ConditionSpec(
        slugify(cid), "perturb_temp", params=dict(alpha=float(alpha)),
        role=role, family=fam, rescue_q=rescue_q,
    )


def c_pot(m: float, *, role: str = "", cid: str | None = None) -> ConditionSpec:
    if cid is None:
        cid = f"pot_m{m:.10g}"
    return ConditionSpec(
        slugify(cid), "perturb_pot", params=dict(m=float(m)),
        role=role, family="pot",
    )


def c_rowmax_pot(K: int, *, tail_mode: str = "clamp", m: int = 1,
                 cid: str | None = None, role: str = "") -> ConditionSpec:
    if cid is None:
        cid = f"rowmax_pot_K{K}_m{m}_{tail_mode}"
    return ConditionSpec(
        slugify(cid), "rowmax_pot",
        params=dict(K=int(K), fine_m=int(m), tail_mode=str(tail_mode)),
        role=role, family="rowmax_pot",
    )


def configure_condition(bta, spec: ConditionSpec, n_layers: int, q_chunk: int) -> None:
    """Map a serializable ConditionSpec to the project's BTA operator."""
    bta.CFG.reset()
    bta.CFG.q_chunk = int(q_chunk)
    all_layers = list(range(n_layers))
    op = spec.operator
    p = spec.params

    if op == "baseline":
        return

    if op in {"strict_binary_soft", "strict_binary_uniform"}:
        li = int(p["target_layer"])
        if not (0 <= li < int(n_layers)):
            raise ValueError(
                f"target_layer={li} outside [0,{int(n_layers)-1}] for {spec.id}"
            )
        # Reuse the existing audited BTA operators and perturb only one layer.
        # All preceding layers remain baseline, so both arms reach the target
        # layer with identical hidden state / QK scores and therefore the same
        # deterministic mean-threshold support mask.
        mode = "binary_soft" if op == "strict_binary_soft" else "binary"
        bta.CFG.layer_mode = {li: mode}
        return

    if op in {"binary_soft", "binary", "topk_soft", "topk"}:
        bta.CFG.layer_mode = {i: op for i in all_layers}
        if op.startswith("topk"):
            bta.CFG.topk_ratio = float(p["topk_ratio"])
        return

    if op == "multibit":
        bta.CFG.layer_mode = {i: "multibit" for i in all_layers}
        bta.CFG.n_levels = int(p["K"])
        bta.CFG.n_levels_per_layer = {}
        bta.CFG.threshold_mode = str(p.get("threshold_mode", "none"))
        bta.CFG.coverage = str(p.get("coverage", "max"))
        bta.CFG.width_ratio = float(p.get("R", 1.0))
        bta.CFG.level_weight = str(p.get("level_weight", "exp"))
        bta.CFG.quant_round = str(p.get("reconstruction", "nearest"))
        # Full-range final protocol never needs row std.
        if bta.CFG.threshold_mode == "none" and bta.CFG.coverage == "max":
            bta.CFG.use_std = False
        return

    if op == "perturb_temp":
        bta.CFG.layer_mode = {i: "perturb" for i in all_layers}
        bta.CFG.pert_arm = "temp"
        bta.CFG.alpha = float(p["alpha"])
        bta.CFG.pert_m = 1.0
        bta.CFG.pert_c = 0.0
        bta.CFG.pert_seed = 0
        return

    if op == "perturb_pot":
        bta.CFG.layer_mode = {i: "perturb" for i in all_layers}
        bta.CFG.pert_arm = "pot"
        bta.CFG.pert_m = float(p["m"])
        bta.CFG.alpha = 1.0
        bta.CFG.pert_c = 0.0
        bta.CFG.pert_seed = 0
        return

    if op == "rowmax_pot":
        bta.CFG.layer_mode = {i: "pow2_max" for i in all_layers}
        bta.CFG.n_levels = int(p["K"])
        bta.CFG.n_levels_per_layer = {}
        bta.CFG.alpha = 1.0
        bta.CFG.round_mode = "round"
        bta.CFG.fine_m = int(p.get("fine_m", 1))
        bta.CFG.fine_T = float(p.get("fine_T", 0.0))
        bta.CFG.tail_mode = str(p.get("tail_mode", "clamp"))
        return

    raise ValueError(f"unknown operator {op!r} for condition {spec.id}")


# =============================================================================
# 3. Quantization geometry: exact analytical mirror of BTA multibit-exp
# =============================================================================


def secant_log_bound(h):
    if isinstance(h, torch.Tensor):
        hs = h.clamp(min=1e-12)
        small = hs < 1e-3
        E = torch.expm1(hs)
        g = E / hs
        tstar = (g - 1.0) / E.clamp(min=1e-30)
        exact = (torch.log(g) - tstar * hs).clamp(min=0.0)
        return torch.where(small, hs * hs / 8.0, exact)
    h = float(h)
    if h < 1e-3:
        return h*h/8.0
    E = math.expm1(h)
    g = E/h
    return math.log(g) - ((g-1.0)/E)*h


class Grid:
    """Per-row full-range geometric grid used by T04/T05 error probes."""

    def __init__(self, cov: torch.Tensor, K: int, R: float, dtype=torch.float64):
        self.cov = cov.to(dtype)
        self.K = int(K)
        self.R = float(R)
        self.uniform = abs(self.R - 1.0) < 1e-9
        if self.uniform:
            self.step = self.cov / self.K
            self.rho = 1.0
            self.den = None
        else:
            self.rho = self.R ** (-1.0 / max(self.K - 1, 1))
            self.den = 1.0 - self.rho ** self.K

    def edge(self, level):
        if self.uniform:
            return level * self.step
        return self.cov * (1.0 - self.rho ** level) / self.den

    def lcont(self, u):
        if self.uniform:
            return u / self.step.clamp(min=1e-12)
        frac = (u / self.cov.clamp(min=1e-12)).clamp(0.0, 1.0)
        arg = (1.0 - frac*self.den).clamp(min=1e-12)
        return torch.log(arg) / math.log(self.rho)

    def _lo_hi(self, u):
        lc = self.lcont(u)
        hi = torch.ceil(lc).clamp(1.0, float(self.K))
        lo = (hi - 1.0).clamp(0.0, float(self.K))
        return lo, hi

    def recon(self, u, mode: str):
        u = u.to(self.cov.dtype)
        if mode == "up":
            if self.uniform:
                lv = (u / self.step.clamp(min=1e-12)).floor() + 1.0
                lv = lv.clamp(0.0, float(self.K))
            else:
                lv = torch.ceil(self.lcont(u)).clamp(1.0, float(self.K))
            e = self.edge(lv)
            h = e - self.edge((lv - 1.0).clamp(min=0.0))
            return e-u, h, lv

        lo, hi = self._lo_hi(u)
        elo, ehi = self.edge(lo), self.edge(hi)
        h = (ehi-elo).clamp(min=1e-12)
        if mode == "nearest":
            pick_lo = (u-elo) <= (ehi-u)
            lv = torch.where(pick_lo, lo, hi)
            e = torch.where(pick_lo, elo, ehi)
            return e-u, h, lv
        if mode == "lerp":
            s = u-elo
            t = (s/h).clamp(0.0, 1.0)
            eps = torch.log1p(t*torch.expm1(h)) - s
            return eps, h, hi
        raise ValueError(mode)

    @staticmethod
    def bound(h, mode: str):
        if mode == "up":
            return h
        if mode == "nearest":
            return 0.5*h
        if mode == "lerp":
            return secant_log_bound(h)
        raise ValueError(mode)


def pot_eps(D: torch.Tensor, m: float):
    return D - torch.round(D * float(m) / LN2) * LN2 / float(m)


# =============================================================================
# 4. Streaming probe accumulators
# =============================================================================


class DivAccumulator:
    """Mean row-wise JSD/TV/KL on local exact-softmax vs intervention output."""

    def __init__(self, min_n: int = 1):
        self.min_n = int(min_n)
        self.n = 0
        self.sjs = self.stv = self.sklf = self.sklr = 0.0

    @torch.no_grad()
    def add(self, scores: torch.Tensor, valid: torch.Tensor, ph: torch.Tensor):
        p = F.softmax(scores, dim=-1)
        keep = valid.sum(-1) >= self.min_n
        if not bool(keep.any()):
            return
        tiny = 1e-30
        lp = torch.log(p.clamp_min(tiny))
        lq = torch.log(ph.clamp_min(tiny))
        m = 0.5*(p+ph)
        lm = torch.log(m.clamp_min(tiny))
        z = torch.zeros_like(p)
        klf = torch.where(valid, p*(lp-lq), z).sum(-1)
        klr = torch.where(valid, ph*(lq-lp), z).sum(-1)
        js = 0.5*torch.where(valid, p*(lp-lm), z).sum(-1) \
           + 0.5*torch.where(valid, ph*(lq-lm), z).sum(-1)
        tv = 0.5*torch.where(valid, (p-ph).abs(), z).sum(-1)
        self.n += int(keep.sum().item())
        self.sjs += float(js[keep].double().sum())
        self.stv += float(tv[keep].double().sum())
        self.sklf += float(klf[keep].double().sum())
        self.sklr += float(klr[keep].double().sum())

    def report(self) -> dict[str, Any]:
        if self.n <= 0:
            raise RuntimeError("divergence accumulator saw zero valid rows")
        return dict(
            js_mean=self.sjs/self.n,
            tv_mean=self.stv/self.n,
            kl_fwd_mean=self.sklf/self.n,
            kl_rev_mean=self.sklr/self.n,
            n_rows=self.n,
        )


class SupportAccumulator:
    def __init__(self):
        self.kept = 0.0
        self.valid = 0.0
        self.per_layer: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])

    @torch.no_grad()
    def add(self, layer: int, M: torch.Tensor, valid: torch.Tensor):
        k = M.sum(dtype=torch.float64)
        v = valid.sum(dtype=torch.float64)
        self.kept += float(k)
        self.valid += float(v)
        a = self.per_layer[int(layer)]
        a[0] += float(k); a[1] += float(v)

    def report(self):
        return dict(
            density=self.kept/max(self.valid, 1.0),
            kept=self.kept,
            valid=self.valid,
            per_layer={str(k): a[0]/max(a[1], 1.0) for k,a in sorted(self.per_layer.items())},
        )


class ErrorGeometryAccumulator:
    """T04/T05/T06 error geometry without storing entrywise tensors."""

    DELTA_W = 0.5
    DELTA_NB = 80

    def __init__(self, spec: ConditionSpec, min_n: int = 64, min_span: float = 1e-3,
                 row_chunk: int = 192):
        self.spec = spec
        self.min_n = int(min_n)
        self.min_span = float(min_span)
        self.row_chunk = int(row_chunk)
        self.n_rows = 0
        self.n_rows_seen = 0
        self.n_entries = 0
        self.sum_e = self.sum_abs = self.sum_e2 = 0.0
        self.sum_p = self.sum_pabs = self.sum_pe2 = 0.0
        self.sum_ctr_abs = self.sum_pctr2 = 0.0
        self.neg = 0
        self.max_abs = 0.0
        self.max_over_bound = 0.0
        self.sum_d = self.sum_d2 = self.sum_ed = 0.0
        nb = self.DELTA_NB + 1
        self.bin_count = np.zeros(nb, np.float64)
        self.bin_abs = np.zeros(nb, np.float64)
        self.bin_p = np.zeros(nb, np.float64)
        self.bin_pabs = np.zeros(nb, np.float64)

    @torch.no_grad()
    def add(self, scores: torch.Tensor, valid: torch.Tensor):
        s = scores.float()
        shp = s.shape
        s2 = s.reshape(-1, shp[-1])
        v2 = valid.reshape(-1, shp[-1])
        for st in range(0, s2.shape[0], self.row_chunk):
            self._add_rows(s2[st:st+self.row_chunk], v2[st:st+self.row_chunk])

    def _add_rows(self, s: torch.Tensor, valid: torch.Tensor):
        self.n_rows_seen += int(s.shape[0])
        neginf = torch.finfo(torch.float32).min
        posinf = torch.finfo(torch.float32).max
        mx = torch.where(valid, s, torch.full_like(s, neginf)).amax(-1, keepdim=True)
        mn = torch.where(valid, s, torch.full_like(s, posinf)).amin(-1, keepdim=True)
        nval = valid.sum(-1, keepdim=True)
        keep = (nval.squeeze(-1) >= self.min_n) & ((mx-mn).squeeze(-1) > self.min_span)
        if not bool(keep.any()):
            return
        s = s[keep]; valid = valid[keep]; mx = mx[keep]; mn = mn[keep]
        D = (mx-s).clamp(min=0.0)
        p = F.softmax(s.masked_fill(~valid, float("-inf")), -1)
        self.update_prepared(s,valid,mx,mn,D,p)

    @torch.no_grad()
    def update_prepared(self, s: torch.Tensor, valid: torch.Tensor,
                        mx: torch.Tensor, mn: torch.Tensor,
                        D: torch.Tensor, p: torch.Tensor):
        """Update one condition from row-filtered quantities shared across conditions."""
        if self.spec.operator == "multibit":
            K = int(self.spec.params["K"])
            R = float(self.spec.params["R"])
            mode = str(self.spec.params["reconstruction"])
            cov = (mx-mn).clamp(min=1e-6)
            u = (s-mn).clamp(min=0.0)
            eps, h, _ = Grid(cov, K, R).recon(u, mode)
            bnd = Grid.bound(h, mode)
        elif self.spec.operator == "perturb_pot":
            m = float(self.spec.params["m"])
            # Mirror implementation arithmetic: BTA computes D/rounding in
            # fp32, exposing the tiny roundoff excess possible relative to the
            # ideal-arithmetic analytical bound.
            eps = pot_eps(D, m)
            bnd = torch.full_like(eps, LN2/(2*m))
        else:
            return

        ev = eps[valid].double()
        pv = p[valid].double()
        dv = D[valid].double()
        bv = bnd[valid].double().clamp_min(1e-300)

        # Row-centered error: eps - E_p eps.
        ebar = (p.double()*eps.double()*valid).sum(-1, keepdim=True) / \
               (p.double()*valid).sum(-1, keepdim=True).clamp_min(1e-300)
        ectr = (eps.double()-ebar)[valid]

        n = int(ev.numel())
        self.n_rows += int(s.shape[0])
        self.n_entries += n
        self.sum_e += float(ev.sum())
        self.sum_abs += float(ev.abs().sum())
        self.sum_e2 += float((ev*ev).sum())
        self.sum_p += float(pv.sum())
        self.sum_pabs += float((pv*ev.abs()).sum())
        self.sum_pe2 += float((pv*ev*ev).sum())
        self.sum_ctr_abs += float(ectr.abs().sum())
        self.sum_pctr2 += float((pv*ectr*ectr).sum())
        self.neg += int((ev < 0).sum().item())
        self.max_abs = max(self.max_abs, float(ev.abs().max()))
        self.max_over_bound = max(self.max_over_bound, float((ev.abs()/bv).max()))
        self.sum_d += float(dv.sum())
        self.sum_d2 += float((dv*dv).sum())
        self.sum_ed += float((ev*dv).sum())

        bi = torch.floor(dv/self.DELTA_W).long().clamp(0, self.DELTA_NB)
        nb = self.DELTA_NB+1
        self.bin_count += torch.bincount(bi, minlength=nb).double().cpu().numpy()
        self.bin_abs += torch.bincount(bi, weights=ev.abs(), minlength=nb).double().cpu().numpy()
        self.bin_p += torch.bincount(bi, weights=pv, minlength=nb).double().cpu().numpy()
        self.bin_pabs += torch.bincount(bi, weights=pv*ev.abs(), minlength=nb).double().cpu().numpy()

    def report(self) -> dict[str, Any]:
        n = max(self.n_entries, 1)
        sp = max(self.sum_p, 1e-300)
        me = self.sum_e/n
        md = self.sum_d/n
        ve = max(self.sum_e2/n-me*me, 0.0)
        vd = max(self.sum_d2/n-md*md, 0.0)
        cov = self.sum_ed/n-me*md
        corr = cov/math.sqrt(max(ve*vd, 1e-300))
        theo = None
        if self.spec.operator == "perturb_pot":
            theo = LN2/(2*float(self.spec.params["m"]))
        return dict(
            condition=self.spec.id,
            n_rows=self.n_rows,
            n_rows_seen=self.n_rows_seen,
            n_rows_skipped=max(self.n_rows_seen-self.n_rows,0),
            n_entries=self.n_entries,
            mean_eps=self.sum_e/n,
            mean_abs_eps=self.sum_abs/n,
            rms_eps=math.sqrt(max(self.sum_e2/n, 0.0)),
            p_mean_abs_eps=self.sum_pabs/sp,
            p_rms_eps=math.sqrt(max(self.sum_pe2/sp, 0.0)),
            mean_abs_eps_centered=self.sum_ctr_abs/n,
            p_rms_eps_centered=math.sqrt(max(self.sum_pctr2/sp, 0.0)),
            frac_negative=self.neg/n,
            max_abs_eps=self.max_abs,
            theoretical_bound=theo,
            max_over_bound=(self.max_abs/theo if theo else self.max_over_bound),
            corr_eps_delta=corr,
        )

    def bin_rows(self) -> list[dict[str, Any]]:
        rows = []
        total_n = max(self.bin_count.sum(), 1.0)
        total_p = max(self.bin_p.sum(), 1e-300)
        for i,c in enumerate(self.bin_count):
            if c <= 0:
                continue
            lo = i*self.DELTA_W
            hi = (i+1)*self.DELTA_W if i < self.DELTA_NB else float("inf")
            rows.append(dict(
                condition=self.spec.id,
                bin_index=i, delta_lo=lo, delta_hi=hi,
                n_entries=int(c), entries_fraction=c/total_n,
                mean_abs_eps=self.bin_abs[i]/c,
                mass_fraction=self.bin_p[i]/total_p,
                p_mean_abs_eps=self.bin_pabs[i]/max(self.bin_p[i], 1e-300),
            ))
        return rows


class ErrorGeometryGroup:
    """Share row filtering, D, and exact-softmax p across diagnostic conditions."""

    def __init__(self, accs: list[ErrorGeometryAccumulator]):
        if not accs:
            raise ValueError("ErrorGeometryGroup requires at least one accumulator")
        self.accs=accs
        self.min_n=accs[0].min_n
        self.min_span=accs[0].min_span
        self.row_chunk=accs[0].row_chunk
        for a in accs[1:]:
            if a.min_n!=self.min_n or a.min_span!=self.min_span or a.row_chunk!=self.row_chunk:
                raise ValueError("grouped error accumulators must share row-filter settings")

    @torch.no_grad()
    def add(self, scores: torch.Tensor, valid: torch.Tensor):
        s=scores.float(); nk=s.shape[-1]
        s2=s.reshape(-1,nk); v2=valid.reshape(-1,nk)
        neginf=torch.finfo(torch.float32).min
        posinf=torch.finfo(torch.float32).max
        for st in range(0,s2.shape[0],self.row_chunk):
            ss=s2[st:st+self.row_chunk]; vv=v2[st:st+self.row_chunk]
            for a in self.accs:
                a.n_rows_seen += int(ss.shape[0])
            mx=torch.where(vv,ss,torch.full_like(ss,neginf)).amax(-1,keepdim=True)
            mn=torch.where(vv,ss,torch.full_like(ss,posinf)).amin(-1,keepdim=True)
            nv=vv.sum(-1,keepdim=True)
            keep=(nv.squeeze(-1)>=self.min_n)&((mx-mn).squeeze(-1)>self.min_span)
            if not bool(keep.any()):
                continue
            ss=ss[keep]; vv=vv[keep]; mx=mx[keep]; mn=mn[keep]
            D=(mx-ss).clamp(min=0.0)
            pp=F.softmax(ss.masked_fill(~vv,float("-inf")),-1)
            for a in self.accs:
                a.update_prepared(ss,vv,mx,mn,D,pp)


class MassConcentrationAccumulator:
    def __init__(self):
        self.thresholds = [1e-4, 1e-3, 1e-2, 1e-1]
        self.n = 0.0
        self.mass = 0.0
        self.lt_n = {t: 0.0 for t in self.thresholds}
        self.lt_mass = {t: 0.0 for t in self.thresholds}
        self.ge_n = {t: 0.0 for t in self.thresholds}
        self.ge_mass = {t: 0.0 for t in self.thresholds}

    @torch.no_grad()
    def add(self, p: torch.Tensor, valid: torch.Tensor):
        v = p[valid].double()
        self.n += float(v.numel())
        self.mass += float(v.sum())
        for t in self.thresholds:
            lt = v < t; ge = ~lt
            self.lt_n[t] += float(lt.sum())
            self.lt_mass[t] += float(v[lt].sum())
            self.ge_n[t] += float(ge.sum())
            self.ge_mass[t] += float(v[ge].sum())

    def rows(self):
        out=[]
        for t in self.thresholds:
            out.append(dict(rule=f"p<{t:g}", entry_fraction=self.lt_n[t]/max(self.n,1),
                            mass_fraction=self.lt_mass[t]/max(self.mass,1e-300)))
            out.append(dict(rule=f"p>={t:g}", entry_fraction=self.ge_n[t]/max(self.n,1),
                            mass_fraction=self.ge_mass[t]/max(self.mass,1e-300)))
        return out


class RowmaxTailAccumulator:
    """Baseline-trajectory tail coverage statistics for practical Rowmax-PoT."""

    def __init__(self, Ks: Iterable[int]):
        self.Ks=sorted(set(int(k) for k in Ks))
        self.n_entries=0.0
        self.mass_total=0.0
        self.sum_octave=0.0
        self.max_octave=0.0
        self.tail_n={k:0.0 for k in self.Ks}
        self.tail_mass={k:0.0 for k in self.Ks}

    @torch.no_grad()
    def add(self, scores: torch.Tensor, valid: torch.Tensor):
        rmx=scores.masked_fill(~valid,float("-inf")).amax(-1,keepdim=True)
        D=(rmx-scores).clamp(min=0.0)
        x=D/LN2
        # Rowmax-PoT formal path uses round_mode='round', implemented in attention.py
        # as floor(x+0.5), not torch.round's ties-to-even convention.
        d=torch.floor(x+0.5)
        p=F.softmax(scores,dim=-1)
        xv=x[valid].double(); pv=p[valid].double()
        self.n_entries += float(xv.numel())
        self.mass_total += float(pv.sum())
        self.sum_octave += float(xv.sum())
        if xv.numel(): self.max_octave=max(self.max_octave,float(xv.max()))
        for K in self.Ks:
            ov=(d>float(K)) & valid
            self.tail_n[K] += float(ov.sum())
            self.tail_mass[K] += float(p[ov].double().sum())

    def rows(self) -> list[dict[str,Any]]:
        out=[]
        for K in self.Ks:
            frac=self.tail_n[K]/max(self.n_entries,1.0)
            out.append(dict(
                K=K,nominal_bits=math.log2(K+1),n_entries=int(self.n_entries),
                mean_octave=self.sum_octave/max(self.n_entries,1.0),max_octave=self.max_octave,
                tail_entry_fraction=frac,
                tail_softmax_mass_fraction=self.tail_mass[K]/max(self.mass_total,1e-300),
                implied_drop_density=1.0-frac,
            ))
        return out


class P4Accumulator:
    """GPU-friendly scalar aggregation for signed leakage and pV fidelity."""

    def __init__(self, qs: Iterable[float], primary_q: float, min_n: int):
        self.qs = sorted(set(float(q) for q in qs))
        self.primary_q = float(primary_q)
        self.min_n = int(min_n)
        self.mass: dict[tuple[int,float], dict[str,float]] = defaultdict(lambda: defaultdict(float))
        self.out: dict[int, dict[str,float]] = defaultdict(lambda: defaultdict(float))
        self.rescue: dict[int, dict[str,float]] = defaultdict(lambda: defaultdict(float))

    def add_mass(self, layer, q, keep, pm, hm, frac):
        if not bool(keep.any()): return
        a=self.mass[(int(layer),float(q))]
        leak=pm-hm; n=int(keep.sum())
        a["n"] += n
        a["ref"] += float(pm[keep].double().sum())
        a["alt"] += float(hm[keep].double().sum())
        a["leak"] += float(leak[keep].double().sum())
        a["abs_leak"] += float(leak[keep].abs().double().sum())
        a["frac"] += float(frac[keep].double().sum())
        a["positive"] += float((leak[keep]>0).double().sum())

    def add_output(self, layer, keep, y_ref, y_pre, y_used):
        if not bool(keep.any()): return
        yr=y_ref.float(); yp=y_pre.float(); yu=y_used.float()
        ref2=(yr*yr).sum(-1)
        dp=((yp-yr)**2).sum(-1)
        du=((yu-yr)**2).sum(-1)
        cospre=F.cosine_similarity(yp,yr,dim=-1,eps=1e-8)
        cosuse=F.cosine_similarity(yu,yr,dim=-1,eps=1e-8)
        a=self.out[int(layer)]; n=int(keep.sum())
        a["n"] += n
        a["ref2"] += float(ref2[keep].double().sum())
        a["pre_diff2"] += float(dp[keep].double().sum())
        a["used_diff2"] += float(du[keep].double().sum())
        a["cos_pre"] += float(cospre[keep].double().sum())
        a["cos_used"] += float(cosuse[keep].double().sum())

    def add_rescue(self, layer, keep, p, ph, pr, mask):
        if not bool(keep.any()): return
        tiny=1e-30
        pm=(p*mask).sum(-1); rm=(pr*mask).sum(-1)
        mm0=0.5*(p+ph); mm1=0.5*(p+pr)
        lp=torch.log(p.clamp_min(tiny)); lh=torch.log(ph.clamp_min(tiny)); lr=torch.log(pr.clamp_min(tiny))
        lm0=torch.log(mm0.clamp_min(tiny)); lm1=torch.log(mm1.clamp_min(tiny))
        js0=0.5*(p*(lp-lm0)).sum(-1)+0.5*(ph*(lh-lm0)).sum(-1)
        js1=0.5*(p*(lp-lm1)).sum(-1)+0.5*(pr*(lr-lm1)).sum(-1)
        tv0=0.5*(p-ph).abs().sum(-1); tv1=0.5*(p-pr).abs().sum(-1)
        err=(rm-pm).abs()
        a=self.rescue[int(layer)]; n=int(keep.sum())
        a["n"] += n
        a["mass_err"] += float(err[keep].double().sum())
        a["mass_err_max"] = max(a.get("mass_err_max",0.0), float(err[keep].max()))
        a["js_pre"] += float(js0[keep].double().sum())
        a["js_rescue"] += float(js1[keep].double().sum())
        a["tv_pre"] += float(tv0[keep].double().sum())
        a["tv_rescue"] += float(tv1[keep].double().sum())

    def mass_rows(self, condition: str):
        rows=[]
        for (li,q),a in sorted(self.mass.items()):
            n=max(a["n"],1.0)
            rows.append(dict(condition=condition,layer=li,q=q,n_rows=int(a["n"]),
                             ref_mass=a["ref"]/n,alt_mass=a["alt"]/n,
                             leakage=a["leak"]/n,mean_abs_leakage=a["abs_leak"]/n,
                             set_fraction=a["frac"]/n,positive_leak_fraction=a["positive"]/n))
        return rows

    def output_rows(self, condition: str):
        rows=[]
        for li,a in sorted(self.out.items()):
            n=max(a["n"],1.0); ref=max(a["ref2"],1e-300)
            rows.append(dict(condition=condition,layer=li,n_rows=int(a["n"]),
                             output_rms_pre=math.sqrt(max(a["pre_diff2"]/ref,0.0)),
                             output_rms_used=math.sqrt(max(a["used_diff2"]/ref,0.0)),
                             cosine_pre=a["cos_pre"]/n,cosine_used=a["cos_used"]/n))
        return rows

    def rescue_rows(self, condition: str):
        rows=[]
        for li,a in sorted(self.rescue.items()):
            n=max(a["n"],1.0)
            rows.append(dict(condition=condition,layer=li,n_rows=int(a["n"]),
                             mass_err_mean=a["mass_err"]/n,mass_err_max=a.get("mass_err_max",0.0),
                             js_pre=a["js_pre"]/n,js_rescue=a["js_rescue"]/n,
                             tv_pre=a["tv_pre"]/n,tv_rescue=a["tv_rescue"]/n))
        return rows


# =============================================================================
# 5. P4 top-mass utilities
# =============================================================================


@torch.no_grad()
def top_mass_masks(p: torch.Tensor, valid: torch.Tensor, qs: Iterable[float]) -> dict[float, torch.Tensor]:
    vals, idx = torch.sort(p, dim=-1, descending=True)
    vs = torch.gather(valid, -1, idx)
    csum = vals.cumsum(-1)
    prev = csum-vals
    out={}
    for q in qs:
        take=(prev < float(q)) & vs
        mask=torch.zeros_like(valid)
        mask.scatter_(-1, idx, take)
        out[float(q)] = mask
    return out


@torch.no_grad()
def mass_partition_rescue(p: torch.Tensor, ph: torch.Tensor, valid: torch.Tensor,
                          mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.equal(p, ph):
        return ph, torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
    p64=p.double(); h64=ph.double(); outmask=valid & (~mask)
    pm=(p64*mask).sum(-1,keepdim=True); po=(p64*outmask).sum(-1,keepdim=True)
    hm=(h64*mask).sum(-1,keepdim=True); ho=(h64*outmask).sum(-1,keepdim=True)
    total=(pm+po).clamp_min(1e-300)
    tin=pm/total; tout=po/total
    impossible=((tin>1e-12)&(hm<=0)) | ((tout>1e-12)&(ho<=0))
    if bool(impossible.any()):
        raise RuntimeError("P4 rescue encountered numerical support collapse")
    sin=torch.where(tin<=1e-12, torch.zeros_like(tin), tin/hm.clamp_min(1e-300))
    sout=torch.where(tout<=1e-12, torch.zeros_like(tout), tout/ho.clamp_min(1e-300))
    pr=torch.where(mask,h64*sin,h64*sout)
    pr=torch.where(valid,pr,torch.zeros_like(pr))
    pr=pr/pr.sum(-1,keepdim=True).clamp_min(1e-300)
    pr=pr.to(ph.dtype)
    pr=torch.where(valid,pr,torch.zeros_like(pr))
    pr=pr/pr.sum(-1,keepdim=True).clamp_min(1e-30)
    err=((pr*mask).sum(-1,keepdim=True)-tin.to(pr.dtype)).abs().squeeze(-1)
    return pr,err


# =============================================================================
# 6. Generic BTA instrumentation patch
# =============================================================================


class ProbeController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.condition: ConditionSpec | None = None
        self.div: DivAccumulator | None = None
        self.support: SupportAccumulator | None = None
        self.error: ErrorGeometryAccumulator | ErrorGeometryGroup | None = None
        self.mass_conc: MassConcentrationAccumulator | None = None
        self.rowmax_tail: RowmaxTailAccumulator | None = None
        self.p4: P4Accumulator | None = None
        self.p4_qs: list[float] = []
        self.p4_primary_q: float = 0.9
        self.p4_min_n: int = 64
        self.p4_probe: bool = False
        self.rescue: bool = False
        self.rescue_q: float | None = None
        self.head_chunk: int = 4
        self.last_capture: dict[str, Any] | None = None


CTRL = ProbeController()
_PATCHED = False
_PATCH_ORIG: dict[str, Any] = {}


def install_probe_patch(bta):
    global _PATCHED
    if _PATCHED:
        return
    _PATCH_ORIG["alt"] = bta._alt_probs
    _PATCH_ORIG["attend"] = bta._attend
    _PATCH_ORIG["log_sparsity"] = bta._log_sparsity
    orig_alt = bta._alt_probs
    orig_attend = bta._attend

    @torch.no_grad()
    def patched_log_sparsity(layer_idx, M, valid):
        if CTRL.support is not None:
            CTRL.support.add(int(layer_idx), M, valid)
        # Do not call the historical logger: it synchronizes once per attention
        # call. CrossModel owns the canonical support statistics.

    @torch.no_grad()
    def patched_alt(scores, valid, mode, q_len, layer_idx=-1, q_start=0, offset_full=0):
        ph, M = orig_alt(scores, valid, mode, q_len, layer_idx, q_start, offset_full)

        if CTRL.div is not None:
            CTRL.div.add(scores, valid, ph)
        if CTRL.error is not None:
            CTRL.error.add(scores, valid)
        if CTRL.rowmax_tail is not None:
            CTRL.rowmax_tail.add(scores, valid)

        need_p = CTRL.mass_conc is not None or CTRL.p4_probe or CTRL.rescue
        if not need_p:
            return ph, M

        p = F.softmax(scores, dim=-1)
        if CTRL.mass_conc is not None:
            CTRL.mass_conc.add(p, valid)

        if CTRL.rescue and (CTRL.condition is None or CTRL.condition.operator != "perturb_temp"):
            raise RuntimeError("P4 rescue is frozen to temperature perturbations")

        ph_pre = ph
        ph_used = ph
        # Process heads in small groups to cap sort/rescue temporary memory.
        if CTRL.p4_probe or CTRL.rescue:
            H = p.shape[1]
            pr = torch.empty_like(ph) if CTRL.rescue else None
            for hs in range(0, H, max(1,CTRL.head_chunk)):
                he=min(hs+max(1,CTRL.head_chunk),H)
                pp=p[:,hs:he]; hh=ph[:,hs:he]; vv=valid[:,hs:he]
                nv=vv.sum(-1); keep=nv>=CTRL.p4_min_n
                qs = CTRL.p4_qs if CTRL.p4_probe else [float(CTRL.rescue_q)]
                masks=top_mass_masks(pp,vv,qs)
                if CTRL.p4_probe and CTRL.p4 is not None:
                    for q in CTRL.p4_qs:
                        mm=masks[float(q)]
                        pm=(pp*mm).sum(-1); hm=(hh*mm).sum(-1)
                        frac=mm.sum(-1).float()/nv.clamp(min=1).float()
                        CTRL.p4.add_mass(layer_idx,q,keep,pm,hm,frac)
                if CTRL.rescue:
                    q0=float(CTRL.rescue_q)
                    rr,err=mass_partition_rescue(pp,hh,vv,masks[q0])
                    if bool(keep.any()) and bool((err[keep]>2e-5).any()):
                        raise RuntimeError(
                            f"P4 rescue partition audit failed condition={CTRL.condition.id} "
                            f"layer={layer_idx} max={float(err[keep].max()):.3e}"
                        )
                    pr[:,hs:he]=rr
                    if CTRL.p4_probe and CTRL.p4 is not None:
                        CTRL.p4.add_rescue(layer_idx,keep,pp,hh,rr,masks[q0])
            if CTRL.rescue:
                ph_used=pr

        if CTRL.p4_probe:
            CTRL.last_capture=dict(
                p=p, ph_pre=ph_pre, ph_used=ph_used,
                valid_n=valid.sum(-1), layer=int(layer_idx),
            )
        return ph_used, M

    @torch.no_grad()
    def patched_attend(q,k,v,attention_mask,scaling,mode,layer_idx,
                       q_start,q_len_full,kv_len,**kwargs):
        CTRL.last_capture=None
        out=orig_attend(q,k,v,attention_mask,scaling,mode,layer_idx,
                        q_start,q_len_full,kv_len,**kwargs)
        if not CTRL.p4_probe:
            return out
        cap=CTRL.last_capture
        if cap is None:
            raise RuntimeError(f"P4 probe active without probability capture at layer {layer_idx}")
        pp=cap["p"]; hp=cap["ph_pre"]
        if pp.ndim!=4 or v.ndim!=4 or out.ndim!=4:
            raise RuntimeError(
                f"P4 pV shape gate failed at layer {layer_idx}: "
                f"p={tuple(pp.shape)} v={tuple(v.shape)} out={tuple(out.shape)}"
            )
        if pp.shape[0]!=v.shape[0] or pp.shape[1]!=v.shape[1] or pp.shape[-1]!=v.shape[-2]:
            raise RuntimeError(
                f"P4 pV GQA/repeat_kv assumption failed at layer {layer_idx}: "
                f"p={tuple(pp.shape)} v={tuple(v.shape)}. "
                "Expected v to be expanded to query-head count before _attend."
            )
        if pp.shape[-2]!=out.shape[-2] or pp.shape[0]!=out.shape[0] or pp.shape[1]!=out.shape[1]:
            raise RuntimeError(
                f"P4 pV q_chunk assumption failed at layer {layer_idx}: "
                f"p={tuple(pp.shape)} out={tuple(out.shape)}"
            )
        keep=cap["valid_n"]>=CTRL.p4_min_n
        yref=torch.matmul(pp.to(v.dtype),v)
        ypre=torch.matmul(hp.to(v.dtype),v)
        yused=out if CTRL.rescue else ypre
        if CTRL.p4 is not None:
            CTRL.p4.add_output(layer_idx,keep,yref,ypre,yused)
        CTRL.last_capture=None
        return out

    bta._log_sparsity = patched_log_sparsity
    bta._alt_probs = patched_alt
    bta._attend = patched_attend
    _PATCHED=True


def restore_probe_patch(bta):
    global _PATCHED
    if not _PATCHED:
        return
    bta._alt_probs=_PATCH_ORIG["alt"]
    bta._attend=_PATCH_ORIG["attend"]
    bta._log_sparsity=_PATCH_ORIG["log_sparsity"]
    _PATCHED=False


# =============================================================================
# 7. Result store and provenance
# =============================================================================


class ResultStore:
    def __init__(self, root: Path, model_slug: str, *, overwrite: bool, resume: bool):
        self.root = Path(root)
        self.model_dir = self.root / model_slug
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            if overwrite:
                shutil.rmtree(self.model_dir)
            elif not resume:
                raise RuntimeError(
                    f"{self.model_dir} is non-empty. Use --resume or --overwrite."
                )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cond_dir = self.model_dir / "condition_blocks"
        self.cond_meta_dir = self.model_dir / "condition_meta"
        self.div_dir = self.model_dir / "divergence"
        self.cond_dir.mkdir(exist_ok=True)
        self.cond_meta_dir.mkdir(exist_ok=True)
        self.div_dir.mkdir(exist_ok=True)

    def task_dir(self, task: str) -> Path:
        p = self.model_dir / task
        p.mkdir(parents=True, exist_ok=True)
        return p

    def condition_block_path(self, cid: str, context: int) -> Path:
        return self.cond_dir / f"{slugify(cid)}__L{int(context)}.csv"

    def condition_meta_path(self, cid: str, context: int) -> Path:
        return self.cond_meta_dir / f"{slugify(cid)}__L{int(context)}.json"

    def divergence_path(self, cid: str, context: int, windows: int) -> Path:
        return self.div_dir / f"{slugify(cid)}__L{int(context)}__W{int(windows)}.json"

    def write_condition_meta(self, spec: ConditionSpec, context: int, protocol: dict[str,Any],
                             support: dict[str,Any] | None = None,
                             cache_state: str = "complete"):
        obj=dict(
            condition=spec.serializable(),
            forward_condition=spec.forward_serializable(),
            context=int(context),
            protocol=protocol,
            signature=short_hash(dict(
                condition=spec.forward_serializable(),
                context=context,
                protocol=protocol,
            )),
            cache_state=str(cache_state),
        )
        if support is not None:
            obj["support"] = support
        write_json(self.condition_meta_path(spec.id,context),obj)

    def all_condition_summaries(self) -> pd.DataFrame:
        rows=[]
        for p in sorted(self.cond_dir.glob("*.csv")):
            try:
                df=pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            cid=str(df["condition"].iloc[0]); L=int(df["seq_len"].iloc[0])
            n_pred=float(df.get("n_pred",pd.Series([1]*len(df))).sum())
            nll=float(np.average(df.nll, weights=df.get("n_pred",pd.Series(np.ones(len(df))))))
            base=float(np.average(df.baseline_nll, weights=df.get("n_pred",pd.Series(np.ones(len(df))))))
            row=dict(condition=cid,seq_len=L,n_blocks=len(df),nll=nll,
                     baseline_nll=base,delta_nll=nll-base,ppl=exp_safe(nll),
                     ppl_ratio=exp_safe(nll-base))
            for c in ["output_kl_forward","output_kl_reverse","output_jsd","top1_agreement"]:
                if c in df and df[c].notna().any():
                    w=df.get("n_pred",pd.Series(np.ones(len(df)))).to_numpy(float)
                    x=df[c].to_numpy(float)
                    ok=np.isfinite(x)
                    row[c]=float(np.average(x[ok],weights=w[ok])) if ok.any() else np.nan
            mp=self.condition_meta_path(cid,L)
            meta=read_json(mp,{}) or {}
            sup=(meta.get("support") or {})
            if "density" in sup:
                row["support_density"]=sup["density"]
            rows.append(row)
        return pd.DataFrame(rows)


# =============================================================================
# 8. Runtime helpers: model, tokens, safe logits/NLL/output fidelity
# =============================================================================


def get_backbone(model):
    if hasattr(model,"model"):
        return model.model
    if hasattr(model,"get_decoder"):
        dec=model.get_decoder()
        if dec is not None:
            return dec
    raise RuntimeError(f"cannot locate decoder backbone for {type(model).__name__}")


def get_output_head(model):
    head=model.get_output_embeddings()
    if head is None:
        head=getattr(model,"lm_head",None)
    if head is None:
        raise RuntimeError(f"cannot locate output embeddings/lm_head for {type(model).__name__}")
    return head


def postprocess_logits(model, logits: torch.Tensor) -> torch.Tensor:
    """Known model-family logit transformations; T00 safe-NLL gate validates this."""
    cfg=model.config
    cap=getattr(cfg,"final_logit_softcapping",None)
    if cap is not None:
        cap=float(cap)
        if cap>0:
            logits=torch.tanh(logits/cap)*cap
    # Do not guess additional family-specific transforms from generic config
    # attributes. T00 compares this chunked path against the model's own
    # forward loss and fails closed if another transform must be implemented.
    return logits


@torch.no_grad()
def backbone_hidden(model, input_ids: torch.Tensor):
    out=get_backbone(model)(input_ids=input_ids,use_cache=False,return_dict=True)
    h=getattr(out,"last_hidden_state",None)
    if h is None:
        h=out[0]
    return h


@torch.no_grad()
def hidden_nll_and_output_metrics(model, h_cond: torch.Tensor, input_ids: torch.Tensor,
                                  *, h_base: torch.Tensor | None = None,
                                  ce_chunk: int = 64,
                                  collect_output: bool = True) -> dict[str,Any]:
    """Chunked LM-head evaluation. h_base enables vocab KL/JSD and Top-1 agreement."""
    head=get_output_head(model)
    npos=int(input_ids.shape[1]-1)
    if npos<=0:
        raise ValueError("sequence too short for LM loss")
    loss_sum=0.0
    base_loss_sum=0.0
    fwd=rev=js=0.0
    top1=0
    for st in range(0,npos,int(ce_chunk)):
        en=min(st+int(ce_chunk),npos)
        hc=h_cond[:,st:en]
        lc=postprocess_logits(model,head(hc)).float()
        tgt=input_ids[:,st+1:en+1]
        loss_sum += float(F.cross_entropy(lc.reshape(-1,lc.shape[-1]),tgt.reshape(-1),reduction="sum"))
        if h_base is not None:
            hb=h_base[:,st:en]
            lb=postprocess_logits(model,head(hb)).float()
            base_loss_sum += float(F.cross_entropy(lb.reshape(-1,lb.shape[-1]),tgt.reshape(-1),reduction="sum"))
            if collect_output:
                lpb=F.log_softmax(lb,dim=-1)
                lpc=F.log_softmax(lc,dim=-1)
                pb=torch.exp(lpb)
                pc=torch.exp(lpc)
                fwd += float((pb*(lpb-lpc)).sum(dtype=torch.float64))
                rev += float((pc*(lpc-lpb)).sum(dtype=torch.float64))
                lm=torch.logaddexp(lpb,lpc)-LN2
                js += 0.5*float((pb*(lpb-lm)).sum(dtype=torch.float64)) \
                    + 0.5*float((pc*(lpc-lm)).sum(dtype=torch.float64))
                top1 += int((lb.argmax(-1)==lc.argmax(-1)).sum().item())
                del lpb,lpc,pb,pc,lm
            del lb
        del lc
    out=dict(n_pred=npos,nll=loss_sum/npos,loss_sum=loss_sum)
    if h_base is not None:
        out["baseline_nll"]=base_loss_sum/npos
        if collect_output:
            out.update(
                output_kl_forward=fwd/npos,
                output_kl_reverse=rev/npos,
                output_jsd=js/npos,
                top1_agreement=top1/npos,
                top1_matches=top1,
            )
    return out


@torch.no_grad()
def standard_nll(model, input_ids: torch.Tensor) -> float:
    return float(model(input_ids=input_ids,labels=input_ids,use_cache=False,return_dict=True).loss)


def token_signature(ids: torch.Tensor) -> str:
    x=ids.detach().cpu().contiguous().numpy()
    return hashlib.md5(x.tobytes()).hexdigest()[:12]


# =============================================================================
# 9. CrossModel runner
# =============================================================================


class CrossModelRunner:
    def __init__(self,args):
        self.args=args
        self.hf_model,self.alias,self.registry=resolve_model(args.model)
        self.profile=args.profile or self.registry["profile"]
        if self.profile not in PROFILE_TASKS:
            raise ValueError(f"unknown profile {self.profile}")
        base_tasks=parse_csv_str(args.tasks) if args.tasks else PROFILE_TASKS[self.profile]
        if not args.tasks:
            extras=set(self.registry.get("extras",[]))
            if "tv_robustness" in extras:
                base_tasks=list(base_tasks)+["T09"]
            if "longctx_8k" in extras:
                base_tasks=list(base_tasks)+["T11"]
        self.tasks=resolve_task_order(base_tasks)
        self.features=set(self.registry.get("extras",[]))
        if self.profile=="deep_reference":
            self.features.add("deep_reference")
        self.model_slug=slugify(self.alias or self.hf_model.split("/")[-1])
        self.store=ResultStore(Path(args.out_root),self.model_slug,
                               overwrite=args.overwrite,resume=args.resume or args.dry_run)
        self.model=None; self.tok=None; self.ids=None; self.bta=None
        self.device=None; self.n_layers=0; self.q_chunk=None
        self.qchunk_gate_rows=[]
        self.implementation_signature=None
        self.runtime_ready=False
        self._div_cache: dict[tuple,dict[str,Any]]={}
        self._baseline_cache: dict[int,pd.DataFrame]={}
        self.protocol_base=dict(
            model=self.hf_model,model_alias=self.alias,profile=self.profile,
            dataset=args.dataset,seq_len=args.seq,num_eval_tokens=args.tokens,
            dtype="bfloat16",batch_size=1,training="none",kv_cache=False,
            fp32_scores=True,bootstrap_reps=args.boot,bootstrap_seed=args.boot_seed,
            intervention_layers="all",intervention_heads="all",
        )
        # Model-wide execution identity contains only settings that can affect
        # shared numerical/model outputs.  T12-only lm-eval knobs are deliberately
        # excluded and get their own task-local signature below.
        self.execution_spec=dict(
            crossmodel_release=CROSSMODEL_RELEASE,
            forward_cache_schema=FORWARD_CACHE_SCHEMA,
            model=self.hf_model,profile=self.profile,dataset=args.dataset,seq_len=args.seq,tokens=args.tokens,
            q_chunk_request=str(args.q_chunk),ce_chunk=args.ce_chunk,verify_windows=args.verify_windows,
            div_tokens=args.div_tokens,div_min_n=args.div_min_n,error_probe_windows=args.error_probe_windows,
            error_min_n=args.error_min_n,error_min_span=args.error_min_span,
            p4_probe_windows=args.p4_probe_windows,p4_min_n=args.p4_min_n,p4_head_chunk=args.p4_head_chunk,
            contexts=args.contexts,tol=args.tol,tol_pref=args.tol_pref,max_root_iter=args.max_root_iter,
            bootstrap_reps=args.boot,bootstrap_seed=args.boot_seed,bound_ratio_tol=args.bound_ratio_tol,
            chunk_tol=args.chunk_tol,chunk_hidden_rel_tol=args.chunk_hidden_rel_tol,
            probe_identity_rel_tol=args.probe_identity_rel_tol,
            impl_rel_tol=args.impl_rel_tol,padding_ratio_tol=args.padding_ratio_tol,
            output_fidelity=not args.no_output_fidelity,fast=bool(args.fast),max_blocks=int(args.max_blocks),
        )
        self.execution_signature=short_hash(self.execution_spec,16)
        self.compatible_execution_signatures={self.execution_signature}
        exec_path=self.store.model_dir/"execution_manifest.json"
        if exec_path.exists() and not args.dry_run:
            old=read_json(exec_path,{}) or {}
            old_sig=old.get("execution_signature")
            old_spec=dict(old.get("execution_spec") or {})
            # v2.2 incorrectly put T12-only settings into the model-wide
            # signature.  Accept and migrate an existing manifest when the only
            # differences are those task-local fields.
            for k in ("zero_shot_tasks","zero_shot_bs","zero_shot_limit"):
                old_spec.pop(k,None)
            if old_sig not in {None,self.execution_signature}:
                if old_spec==self.execution_spec:
                    self.compatible_execution_signatures.add(old_sig)
                    print("[manifest migrate] v2.2 T12-only fields removed from model-wide execution identity")
                else:
                    raise RuntimeError(
                        "Existing output directory was created with different formal/debug settings. "
                        "Do not mix --fast/--max-blocks or changed shared statistical settings with a formal run. "
                        "Use a separate --out-root. WARNING: --overwrite deletes the whole model result directory."
                    )
        if not args.dry_run:
            write_json(exec_path,dict(execution_signature=self.execution_signature,
                                      execution_spec=self.execution_spec))
        self.plan=self.make_plan()
        if not args.dry_run:
            write_json(self.store.model_dir/"protocol.json",self.protocol_base)
            write_json(self.store.model_dir/"analysis_plan.json",self.plan)
            write_json(self.store.model_dir/"run_status.json",dict(status="planned",tasks=self.tasks,
                                                                   execution_signature=self.execution_signature))

    # ------------------------------------------------------------------
    # Plan / runtime
    # ------------------------------------------------------------------
    def task_execution_spec(self, task: str) -> dict[str,Any]:
        """Execution identity for task-local knobs that must not invalidate shared caches."""
        spec=dict(model_execution_signature=self.execution_signature,task=str(task))
        if task=="T01B":
            spec.update(
                strict_support_protocol="single_layer_mean_threshold_v1",
                strict_support_layers="all",
                strict_support_weighting_arms=["exact_relative_softmax", "uniform"],
            )
        if task=="T12":
            spec.update(
                zero_shot_tasks=self.args.zero_shot_tasks,
                zero_shot_bs=int(self.args.zero_shot_bs),
                zero_shot_limit=int(self.args.zero_shot_limit),
                allow_missing_lm_eval=bool(self.args.allow_missing_lm_eval),
            )
        return spec

    def task_execution_signature(self, task: str) -> str:
        return short_hash(self.task_execution_spec(task),16)

    def make_plan(self):
        return dict(
            code="crossmodel",
            execution_signature=self.execution_signature,
            model=self.hf_model,alias=self.alias,profile=self.profile,
            features=sorted(self.features),tasks=self.tasks,
            final_data_policy="paper figures/tables consume only unified pipeline outputs",
            primary_endpoints=dict(
                strict_weighting="D_W = mean_layer[dNLL(uniform)-dNLL(exact-weight)] under strict single-layer same support",
                allocation="D_R = dNLL(R1,K21)-dNLL(R4,K21)",
                direction="D_T = dNLL(A-)-dNLL(A+)",
                rescue="G_gap = D_orig-D_rescue",
            ),
            guardrails=[
                "No formal fixed-span sweep.",
                "No downstream-outcome-guided JSD/TV matching.",
                "Every model derives its own K0 from local PoT m=1 JSD.",
                "P4 high-mass sets use local exact softmax on the current trajectory.",
                "R<1 points are pre-specified deep-reference anti-allocation controls.",
                "Vocabulary KL/JSD and Top-1 agreement are secondary diagnostics.",
            ],
        )

    def load_runtime(self):
        if self.runtime_ready:
            return
        from softmax_approx import attention as bta
        from softmax_approx import data as _eval_ppl
        from softmax_approx import model_loader as _load_model
        build_tokens=_eval_ppl.build_tokens
        load=_load_model.load
        MASK_REGISTERED=_load_model.MASK_REGISTERED
        self._load_fn=load
        self._attn_impl=_load_model.ATTN_IMPLEMENTATION
        self.bta=bta
        impl_files={
            "crossmodel": str(Path(__file__).resolve()),
            "bta": str(Path(bta.__file__).resolve()),
            "load_model": str(Path(_load_model.__file__).resolve()),
            "eval_ppl": str(Path(_eval_ppl.__file__).resolve()),
        }
        impl_byte_hashes={k:file_sha256(v) for k,v in impl_files.items()}
        impl_ast_hashes={k:python_ast_sha256(v) for k,v in impl_files.items()}
        # Expensive forward caches are tied to the attention/model-loading
        # semantics, not to comments or analysis-only changes in this driver.
        # If crossmodel's FORWARD semantics change, bump FORWARD_CACHE_SCHEMA.
        forward_sig_payload=dict(
            schema=FORWARD_CACHE_SCHEMA,
            bta_ast=impl_ast_hashes.get("bta"),
            load_model_ast=impl_ast_hashes.get("load_model"),
        )
        self.implementation_signature=short_hash(forward_sig_payload,16)
        self.analysis_implementation_signature=short_hash(
            dict(release=CROSSMODEL_RELEASE,ast=impl_ast_hashes),16
        )
        self.protocol_base["implementation_signature"]=self.implementation_signature
        self.protocol_base["analysis_implementation_signature"]=self.analysis_implementation_signature
        self.protocol_base["implementation_byte_hashes"]=impl_byte_hashes
        self.protocol_base["implementation_ast_hashes"]=impl_ast_hashes
        self.protocol_base["forward_cache_schema"]=FORWARD_CACHE_SCHEMA
        install_probe_patch(bta)
        print(f"[load] {self.hf_model}")
        self.model,self.tok=load(self.hf_model,trust_remote_code=self.args.trust_remote_code)
        self.model.eval(); self.device=model_device(self.model)
        self.n_layers=int(self.model.config.num_hidden_layers)
        self.ids=build_tokens(self.tok,n_tokens=self.args.tokens,tag=self.hf_model,
                              dataset=self.args.dataset)
        if self.ids.ndim==2 and self.ids.shape[0]==1:
            self.ids=self.ids.squeeze(0)
        if self.ids.ndim!=1:
            raise RuntimeError(f"unexpected token shape {tuple(self.ids.shape)}")
        self.q_chunk=self.select_q_chunk()
        self.protocol_base["q_chunk"]=self.q_chunk
        self.protocol_base["token_signature"]=token_signature(self.ids)
        self.protocol_base["mask_registered"]=bool(MASK_REGISTERED)
        self.protocol_base["model_revision"]=getattr(self.model.config,"_commit_hash",None)
        self.protocol_base["tokenizer_revision"]=(getattr(self.tok,"init_kwargs",{}) or {}).get("_commit_hash")
        write_json(self.store.model_dir/"protocol.json",self.protocol_base)
        hq=int(self.model.config.num_attention_heads)
        hkv=int(getattr(self.model.config,"num_key_value_heads",hq))
        info=dict(
            model=self.hf_model,alias=self.alias,family=self.registry.get("family","unknown"),
            approx_params=self.registry.get("approx_params","unknown"),
            exact_parameters=model_param_count(self.model),num_layers=self.n_layers,
            hidden_size=int(self.model.config.hidden_size),num_attention_heads=hq,
            num_key_value_heads=hkv,
            attention_type="MHA" if hq==hkv else f"GQA x{hq//hkv}",
            max_position_embeddings=getattr(self.model.config,"max_position_embeddings",None),
            vocab_size=int(self.model.config.vocab_size),dtype=str(next(self.model.parameters()).dtype),
            device=str(self.device),chosen_q_chunk=self.q_chunk,
            tokenizer=getattr(self.tok,"name_or_path",None),
            model_revision=getattr(self.model.config,"_commit_hash",None),
            tokenizer_revision=(getattr(self.tok,"init_kwargs",{}) or {}).get("_commit_hash"),
            attn_logit_softcapping=getattr(self.model.config,"attn_logit_softcapping",None),
            final_logit_softcapping=getattr(self.model.config,"final_logit_softcapping",None),
            sliding_window=getattr(self.model.config,"sliding_window",None),
            transformers_version=self._pkg_version("transformers"),
            torch_version=torch.__version__,
        )
        write_json(self.store.model_dir/"model_info.json",info)
        self.runtime_ready=True
        print(f"[runtime] layers={self.n_layers} device={self.device} q_chunk={self.q_chunk} "
              f"tokens={len(self.ids)} sig={self.protocol_base['token_signature']}")

    @staticmethod
    def _pkg_version(name):
        try:
            import importlib.metadata as im
            return im.version(name)
        except Exception:
            return None

    def _memory_gate_max_context(self) -> int:
        """Largest context promised by this model/profile, independent of task subset.

        q_chunk is a model-level frozen protocol choice.  Therefore a smoke run
        such as ``--tasks T00`` must still validate the maximum context that the
        model's registered profile is committed to run later.  Do not key this
        decision on ``self.tasks``: otherwise Qwen2.5-1.5B T00-only would select
        q_chunk using only 2K even though the deep-reference profile later runs
        16K, violating the frozen-q_chunk contract.
        """
        if "deep_reference" in self.features:
            vals=parse_csv_int(self.args.contexts) or [2048,8192,16384]
            return max(vals)
        if "longctx_8k" in self.features:
            return 8192
        return int(self.args.seq)

    def select_q_chunk(self) -> int:
        self.qchunk_gate_rows=[]
        if str(self.args.q_chunk).lower()!="auto":
            q=int(self.args.q_chunk)
            if q<=0: raise ValueError("--q-chunk must be positive or auto")
            self.qchunk_gate_rows.append(dict(check="q_chunk_memory_gate",q_chunk=q,
                                              status="user_fixed",selected=True))
            return q

        # Outcome-independent memory gate with two stressors:
        #   (1) local 2K-ish P4 rescue, the heaviest probe path;
        #   (2) the largest context promised by T11, using R4 K21.
        # The selected q_chunk is then frozen for all conditions of this model.
        cand=[512,256,128]
        p4_len=min(int(self.args.seq),2048)
        long_len=int(self._memory_gate_max_context())
        maxpos=getattr(self.model.config,"max_position_embeddings",None)
        if maxpos is not None and long_len>int(maxpos):
            raise RuntimeError(
                f"q_chunk memory gate requested L={long_len}, but model "
                f"max_position_embeddings={maxpos}"
            )

        def run_stress(q:int,kind:str,L:int)->float:
            ch=self.ids[:L].unsqueeze(0).to(self.device)
            if ch.shape[-1]!=L:
                raise RuntimeError(f"not enough cached tokens for q_chunk stress L={L}")
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if kind=="p4_rescue":
                sp=c_temp(0.9,cid="qchunk_stress_p4")
                configure_condition(self.bta,sp,self.n_layers,q)
                CTRL.reset(); CTRL.condition=sp; CTRL.rescue=True; CTRL.rescue_q=0.9
            elif kind=="longctx_r4":
                sp=c_multibit(21,4,"nearest","exp",cid="qchunk_stress_longctx")
                configure_condition(self.bta,sp,self.n_layers,q)
                CTRL.reset(); CTRL.condition=sp
            else:
                raise ValueError(kind)
            with torch.no_grad():
                oo=get_backbone(self.model)(input_ids=ch,use_cache=False,return_dict=True)
            del oo,ch
            peak=(torch.cuda.max_memory_allocated()/2**30) if torch.cuda.is_available() else float("nan")
            CTRL.reset()
            gc.collect()
            torch.cuda.empty_cache()
            return peak

        for q in cand:
            local=[]
            try:
                pk=run_stress(q,"p4_rescue",p4_len)
                local.append(dict(stressor="p4_rescue",seq_len=p4_len,peak_allocated_gb=pk))
                if long_len>p4_len:
                    pk=run_stress(q,"longctx_r4",long_len)
                    local.append(dict(stressor="longctx_r4",seq_len=long_len,peak_allocated_gb=pk))
                for r in local:
                    self.qchunk_gate_rows.append(dict(
                        check="q_chunk_memory_gate",q_chunk=q,status="ok",
                        selected=True,**r
                    ))
                peak=max(r["peak_allocated_gb"] for r in local)
                print(
                    f"[q_chunk auto] selected {q}; stress peak={peak:.2f} GB "
                    f"(P4 L={p4_len}, max-context L={long_len})"
                )
                return q
            except torch.cuda.OutOfMemoryError:
                peak=(torch.cuda.max_memory_allocated()/2**30) if torch.cuda.is_available() else float("nan")
                self.qchunk_gate_rows.append(dict(
                    check="q_chunk_memory_gate",q_chunk=q,status="oom",
                    stressor="combined",seq_len=max(p4_len,long_len),
                    peak_allocated_gb=peak,selected=False
                ))
                CTRL.reset()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[q_chunk auto] {q} OOM under formal stress gate -> try smaller")
        raise RuntimeError("q_chunk auto gate failed even at 128")

    # ------------------------------------------------------------------
    # Block/token helpers
    # ------------------------------------------------------------------
    def n_blocks(self,L:int) -> int:
        n=len(self.ids)//int(L)
        if self.args.max_blocks>0:
            n=min(n,int(self.args.max_blocks))
        if self.args.fast:
            n=min(n,4)
        return n

    def get_block(self,bi:int,L:int) -> torch.Tensor:
        st=bi*L
        ch=self.ids[st:st+L]
        if ch.numel()!=L:
            raise RuntimeError(f"incomplete block {bi} L={L}")
        return ch.unsqueeze(0).to(self.device)

    def divergence_windows(self,L:int) -> int:
        if self.args.fast:
            return 1
        if L==self.args.seq:
            return int(self.args.verify_windows)
        return max(1,int(self.args.div_tokens)//int(L))

    # ------------------------------------------------------------------
    # Condition evaluation with output-fidelity metrics
    # ------------------------------------------------------------------
    def _condition_protocol(self,spec:ConditionSpec,L:int) -> dict[str,Any]:
        return dict(
            model=self.hf_model,dataset=self.args.dataset,token_signature=token_signature(self.ids),
            model_revision=self.protocol_base.get("model_revision"),
            tokenizer_revision=self.protocol_base.get("tokenizer_revision"),
            implementation_signature=self.implementation_signature,
            forward_cache_schema=FORWARD_CACHE_SCHEMA,
            seq_len=int(L),q_chunk=int(self.q_chunk),dtype="bf16",ce_chunk=int(self.args.ce_chunk),
            condition=spec.forward_serializable(),collect_output_fidelity=not self.args.no_output_fidelity,
        )

    def _load_compatible_condition(self,spec:ConditionSpec,L:int) -> pd.DataFrame|None:
        """Load compatible complete or partial condition rows.

        Stale protocol caches are invalidated and recomputed instead of raising.
        """
        p=self.store.condition_block_path(spec.id,L)
        mp=self.store.condition_meta_path(spec.id,L)
        if not p.exists() or not mp.exists():
            return None
        meta=read_json(mp,{}) or {}
        expect=self._condition_protocol(spec,L)
        sig=short_hash(dict(condition=spec.forward_serializable(),context=L,protocol=expect))
        if meta.get("signature")!=sig:
            print(f"[invalidate] stale condition cache -> recompute: {p.name}")
            return None
        try:
            df=pd.read_csv(p)
        except Exception:
            return None
        required={"block","condition","seq_len","nll","baseline_nll","delta_nll","n_pred"}
        if not required.issubset(df.columns):
            return None
        if not self.args.no_output_fidelity:
            needed={"output_kl_forward","output_kl_reverse","output_jsd","top1_agreement"}
            if not needed.issubset(df.columns):
                return None
        df=df[(df.seq_len.astype(int)==int(L)) & (df.condition.astype(str)==str(spec.id))].copy()
        if df.empty:
            return None
        return (df.sort_values("block").drop_duplicates("block",keep="last")
                .reset_index(drop=True))

    def _load_complete_condition(self,spec:ConditionSpec,L:int,nblocks:int) -> pd.DataFrame|None:
        df=self._load_compatible_condition(spec,L)
        if df is None:
            return None
        have=set(int(x) for x in df.block)
        need=set(range(nblocks))
        if need.issubset(have):
            return df[df.block.isin(range(nblocks))].sort_values("block").reset_index(drop=True)
        return None

    @torch.no_grad()
    def evaluate_group(self,specs:list[ConditionSpec],L:int|None=None,
                       *, task:str, collect_support_ids:Iterable[str]=(),
                       baseline_mean_support_path:Path|None=None) -> dict[str,pd.DataFrame]:
        self.load_runtime(); L=int(L or self.args.seq); nblocks=self.n_blocks(L)
        if nblocks<=0: raise RuntimeError(f"no full blocks for L={L}")
        out:dict[str,pd.DataFrame]={}; missing=[]
        collect_support=set(collect_support_ids)

        rows:dict[str,list[dict[str,Any]]]={}
        done:dict[str,set[int]]={}
        for sp in specs:
            d=self._load_compatible_condition(sp,L)
            if d is not None:
                have={int(x) for x in d.block if 0<=int(x)<nblocks}
                d=d[d.block.isin(sorted(have))].sort_values("block").reset_index(drop=True)
            else:
                have=set()
            if set(range(nblocks)).issubset(have):
                out[sp.id]=d[d.block.isin(range(nblocks))].reset_index(drop=True)
                print(f"[reuse] {sp.id} L={L} blocks={nblocks}")
            else:
                missing.append(sp)
                rows[sp.id]=([] if d is None else d.to_dict("records"))
                done[sp.id]=have
                if have:
                    print(f"[resume-partial] {sp.id} L={L} have={len(have)}/{nblocks}")

        support_acc={sp.id:SupportAccumulator() for sp in missing if sp.id in collect_support}
        # Support stats are aggregate-only, so if downstream rows are partial
        # rebuild support over all blocks while reusing completed NLL rows.
        support_rebuild={cid for cid in support_acc if done.get(cid)}

        baseline_support_acc=None
        baseline_support_sig=None
        baseline_support_needed=False
        if baseline_mean_support_path is not None:
            baseline_mean_support_path=Path(baseline_mean_support_path)
            baseline_support_sig=short_hash(dict(
                model=self.hf_model,seq_len=L,nblocks=nblocks,q_chunk=self.q_chunk,
                token_signature=token_signature(self.ids),
                model_revision=self.protocol_base.get("model_revision"),
                implementation_signature=self.implementation_signature,
                definition="mean-threshold mask on exact-softmax baseline trajectory",
            ),16)
            oldsup=read_json(baseline_mean_support_path,{}) or {}
            baseline_support_needed=(oldsup.get("signature")!=baseline_support_sig)
            if baseline_support_needed:
                baseline_support_acc=SupportAccumulator()

        if not missing and not baseline_support_needed:
            return out

        for sp in missing:
            self.store.write_condition_meta(
                sp,L,self._condition_protocol(sp,L),support=None,cache_state="partial"
            )

        def checkpoint():
            for sp in missing:
                rr=rows[sp.id]
                if not rr:
                    continue
                df=(pd.DataFrame(rr).sort_values("block")
                    .drop_duplicates("block",keep="last"))
                df.to_csv(self.store.condition_block_path(sp.id,L),index=False)

        t0=time.time()
        ck=max(1,int(self.args.checkpoint_every))
        for bi in range(nblocks):
            need_block=baseline_support_needed
            if not need_block:
                for sp in missing:
                    if bi not in done.get(sp.id,set()) or sp.id in support_rebuild:
                        need_block=True
                        break
            if not need_block:
                continue

            ch=self.get_block(bi,L)
            if baseline_support_needed:
                bsp=c_binary(True)
                configure_condition(self.bta,bsp,self.n_layers,self.q_chunk)
                self.bta.CFG.probe_sparsity=True
                CTRL.reset(); CTRL.condition=bsp; CTRL.support=baseline_support_acc
            else:
                configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk)
                CTRL.reset()
            hb=backbone_hidden(self.model,ch)
            CTRL.reset()
            bmet=hidden_nll_and_output_metrics(
                self.model,hb,ch,ce_chunk=self.args.ce_chunk,collect_output=False
            )
            bnll=float(bmet["nll"])

            for sp in missing:
                already=(bi in done.get(sp.id,set()))
                rebuild_support=(sp.id in support_rebuild)
                if already and not rebuild_support:
                    continue

                if sp.operator=="baseline":
                    if not already:
                        row=dict(
                            model=self.hf_model,task=task,condition=sp.id,block=bi,seq_len=L,
                            n_pred=int(bmet["n_pred"]),baseline_nll=bnll,nll=bnll,delta_nll=0.0,
                        )
                        if not self.args.no_output_fidelity:
                            row.update(output_kl_forward=0.0,output_kl_reverse=0.0,
                                       output_jsd=0.0,top1_agreement=1.0,
                                       top1_matches=int(bmet["n_pred"]))
                        rows[sp.id].append(row); done[sp.id].add(bi)
                    continue

                configure_condition(self.bta,sp,self.n_layers,self.q_chunk)
                CTRL.reset(); CTRL.condition=sp
                if sp.id in support_acc:
                    CTRL.support=support_acc[sp.id]
                if sp.rescue_q is not None:
                    CTRL.rescue=True; CTRL.rescue_q=float(sp.rescue_q)
                hc=backbone_hidden(self.model,ch)

                if already:
                    CTRL.reset(); del hc
                    continue

                met=hidden_nll_and_output_metrics(
                    self.model,hc,ch,h_base=hb,ce_chunk=self.args.ce_chunk,
                    collect_output=not self.args.no_output_fidelity,
                )
                row=dict(
                    model=self.hf_model,task=task,condition=sp.id,block=bi,seq_len=L,
                    n_pred=int(met["n_pred"]),baseline_nll=bnll,nll=float(met["nll"]),
                    delta_nll=float(met["nll"])-bnll,
                )
                for k in ("output_kl_forward","output_kl_reverse","output_jsd","top1_agreement","top1_matches"):
                    if k in met: row[k]=met[k]
                rows[sp.id].append(row); done[sp.id].add(bi)
                CTRL.reset(); del hc

            del hb,ch
            if (bi+1)%ck==0 or bi+1==nblocks:
                checkpoint()
            if (bi+1)%max(1,min(10,nblocks))==0 or bi+1==nblocks:
                print(f"  [{task}] block {bi+1:3d}/{nblocks:3d} conditions={len(missing)} "
                      f"elapsed={time.time()-t0:.0f}s")

        checkpoint()
        for sp in missing:
            df=(pd.DataFrame(rows[sp.id]).sort_values("block")
                .drop_duplicates("block",keep="last"))
            have=set(int(x) for x in df.block)
            if not set(range(nblocks)).issubset(have):
                raise RuntimeError(
                    f"incomplete condition after evaluation: {sp.id} have={len(have)}/{nblocks}"
                )
            df=df[df.block.isin(range(nblocks))].reset_index(drop=True)
            df.to_csv(self.store.condition_block_path(sp.id,L),index=False)
            sup=support_acc[sp.id].report() if sp.id in support_acc else None
            self.store.write_condition_meta(
                sp,L,self._condition_protocol(sp,L),sup,cache_state="complete"
            )
            out[sp.id]=df

        if baseline_support_needed:
            rep=baseline_support_acc.report()
            write_json(baseline_mean_support_path,dict(
                signature=baseline_support_sig,
                definition="mean-threshold mask on exact-softmax baseline trajectory",
                seq_len=L,n_blocks=nblocks,support=rep,
            ))
        CTRL.reset()
        return out

    def condition_summary(self,spec:ConditionSpec,L:int|None=None) -> dict[str,Any]:
        L=int(L or self.args.seq); n=self.n_blocks(L)
        df=self._load_complete_condition(spec,L,n)
        if df is None:
            raise RuntimeError(f"condition not evaluated: {spec.id} L={L}")
        w=df.n_pred.to_numpy(float)
        nll=float(np.average(df.nll,weights=w)); bnll=float(np.average(df.baseline_nll,weights=w))
        out=dict(condition=spec.id,operator=spec.operator,role=spec.role,family=spec.family,
                 seq_len=L,n_blocks=len(df),n_pred=int(w.sum()),nll=nll,baseline_nll=bnll,
                 delta_nll=nll-bnll,ppl=exp_safe(nll),ppl_ratio=exp_safe(nll-bnll),
                 rescue_q=spec.rescue_q)
        for pk in ("K","R","reconstruction","level_weight","alpha","m","tail_mode","fine_m","topk_ratio","target_layer"):
            if pk in spec.params:
                out[pk]=spec.params[pk]
        if "K" in spec.params:
            out["nominal_bits"]=math.log2(int(spec.params["K"])+1)
        for k in ("output_kl_forward","output_kl_reverse","output_jsd","top1_agreement"):
            if k in df:
                x=df[k].to_numpy(float); ok=np.isfinite(x)
                if ok.any(): out[k]=float(np.average(x[ok],weights=w[ok]))
        meta=read_json(self.store.condition_meta_path(spec.id,L),{}) or {}
        if (meta.get("support") or {}).get("density") is not None:
            out["support_density"]=meta["support"]["density"]
        return out

    def delta_series(self,spec:ConditionSpec,L:int|None=None) -> np.ndarray:
        L=int(L or self.args.seq); n=self.n_blocks(L)
        df=self._load_complete_condition(spec,L,n)
        if df is None: raise RuntimeError(f"missing blocks for {spec.id}")
        return df.sort_values("block").delta_nll.to_numpy(np.float64)

    # ------------------------------------------------------------------
    # Divergence measurement: same current-trajectory local-reference rule
    # ------------------------------------------------------------------
    @torch.no_grad()
    def measure_divergence(self,spec:ConditionSpec,L:int|None=None,windows:int|None=None,
                           *, force=False) -> dict[str,Any]:
        self.load_runtime(); L=int(L or self.args.seq)
        nw=int(windows or self.divergence_windows(L))
        key=(spec.id,short_hash(spec.forward_serializable(),16),L,nw)
        if key in self._div_cache and not force:
            return self._div_cache[key]
        path=self.store.divergence_path(spec.id,L,nw)
        if path.exists() and not force:
            obj=read_json(path)
            sig=short_hash(dict(condition=spec.forward_serializable(),L=L,windows=nw,
                                q_chunk=self.q_chunk,dataset=self.args.dataset,
                                token_signature=token_signature(self.ids),
                                model_revision=self.protocol_base.get("model_revision"),
                                implementation_signature=self.implementation_signature))
            if obj.get("signature")==sig:
                self._div_cache[key]=obj["metrics"]
                return obj["metrics"]
        configure_condition(self.bta,spec,self.n_layers,self.q_chunk)
        CTRL.reset(); CTRL.condition=spec; CTRL.div=DivAccumulator(min_n=self.args.div_min_n)
        for w in range(nw):
            ch=self.get_block(w,L)
            _=get_backbone(self.model)(input_ids=ch,use_cache=False,return_dict=True)
            del ch
        met=CTRL.div.report(); CTRL.reset()
        sig=short_hash(dict(condition=spec.forward_serializable(),L=L,windows=nw,
                            q_chunk=self.q_chunk,dataset=self.args.dataset,
                            token_signature=token_signature(self.ids),
                            model_revision=self.protocol_base.get("model_revision"),
                            implementation_signature=self.implementation_signature))
        write_json(path,dict(
            signature=sig,
            condition=spec.serializable(),
            forward_condition=spec.forward_serializable(),
            seq_len=L,
            windows=nw,
            metrics=met,
        ))
        self._div_cache[key]=met
        return met

    # ------------------------------------------------------------------
    # Root matching
    # ------------------------------------------------------------------
    def match_temperature(self,target:float,side:str,axis:str,L:int,
                          x_start:float,label:str,trace_rows:list[dict[str,Any]],
                          windows:int | None = None) -> dict[str,Any]:
        if side not in {"A-","A+"}: raise ValueError(side)
        metric_key="js_mean" if axis=="jsd" else "tv_mean"
        xmin,xmax=(1e-5,0.95) if side=="A-" else (1e-5,50.0)
        hist=[]

        def alpha_of(x): return 1.0-x if side=="A-" else 1.0+x
        def ev(x,it,phase):
            x=float(np.clip(x,xmin,xmax)); a=alpha_of(x)
            sp=c_temp(a,cid=f"match_{label}_{side}_a{a:.10g}")
            rep=self.measure_divergence(sp,L,windows)
            val=float(rep[metric_key]); rel=abs(val-target)/target
            row=dict(target_label=label,axis=axis,cell=side,phase=phase,iteration=it,
                     x=x,alpha=a,metric=val,target=target,rel_err=rel)
            trace_rows.append(row); hist.append((x,val,rel,a,sp,rep))
            print(f"    [{label} {side} {axis}] {phase:<7s} it={it:<2d} alpha={a:.9g} "
                  f"metric={val:.6e} rel={100*rel:.3f}%")
            return hist[-1]

        x0,v0,r0,a0,s0,rep0=ev(x_start,0,"verify")
        if r0<=self.args.tol_pref:
            return dict(x=x0,alpha=a0,metric=v0,rel_err=r0,spec=s0,rep=rep0,status="converged")
        low=high=None
        if v0<target:
            low=(x0,v0); x=x0
            for j in range(1,self.args.max_root_iter+1):
                xn=min(xmax,x*1.6)
                if xn<=x*(1+1e-12): break
                x,v,r,a,s,rep=ev(xn,j,"bracket")
                if r<=self.args.tol_pref:
                    return dict(x=x,alpha=a,metric=v,rel_err=r,spec=s,rep=rep,status="converged")
                if v>=target: high=(x,v); break
                low=(x,v)
        else:
            high=(x0,v0); x=x0
            for j in range(1,self.args.max_root_iter+1):
                xn=max(xmin,x/1.6)
                if xn>=x*(1-1e-12): break
                x,v,r,a,s,rep=ev(xn,j,"bracket")
                if r<=self.args.tol_pref:
                    return dict(x=x,alpha=a,metric=v,rel_err=r,spec=s,rep=rep,status="converged")
                if v<=target: low=(x,v); break
                high=(x,v)
        if low is not None and high is not None:
            lo,hi=low[0],high[0]
            base_it=len(hist)
            for j in range(self.args.max_root_iter):
                mid=0.5*(lo+hi)
                x,v,r,a,s,rep=ev(mid,base_it+j,"bisect")
                if r<=self.args.tol_pref:
                    return dict(x=x,alpha=a,metric=v,rel_err=r,spec=s,rep=rep,status="converged")
                if v<target: lo=x
                else: hi=x
        best=min(hist,key=lambda t:t[2])
        status="converged" if best[2]<=self.args.tol else "not_converged"
        return dict(x=best[0],alpha=best[3],metric=best[1],rel_err=best[2],
                    spec=best[4],rep=best[5],status=status)

    def match_pot_jsd(self,target:float,L:int,x_start:float,label:str,
                      trace_rows:list[dict[str,Any]], windows:int | None = None) -> dict[str,Any]:
        xmin,xmax=1e-4,50.0 # x=1/m
        hist=[]
        def ev(x,it,phase):
            x=float(np.clip(x,xmin,xmax)); m=1.0/x
            sp=c_pot(m,cid=f"match_{label}_pot_m{m:.10g}")
            rep=self.measure_divergence(sp,L,windows); val=float(rep["js_mean"]); rel=abs(val-target)/target
            trace_rows.append(dict(target_label=label,axis="jsd",cell="C",phase=phase,
                                   iteration=it,x=x,m=m,metric=val,target=target,rel_err=rel))
            hist.append((x,val,rel,m,sp,rep))
            print(f"    [{label} C jsd] {phase:<7s} it={it:<2d} m={m:.9g} "
                  f"metric={val:.6e} rel={100*rel:.3f}%")
            return hist[-1]
        x0,v0,r0,m0,s0,rep0=ev(x_start,0,"verify")
        if r0<=self.args.tol_pref:
            return dict(x=x0,m=m0,metric=v0,rel_err=r0,spec=s0,rep=rep0,status="converged")
        low=high=None
        if v0<target:
            low=(x0,v0); x=x0
            for j in range(1,self.args.max_root_iter+1):
                xn=min(xmax,x*1.6); x,v,r,m,s,rep=ev(xn,j,"bracket")
                if r<=self.args.tol_pref: return dict(x=x,m=m,metric=v,rel_err=r,spec=s,rep=rep,status="converged")
                if v>=target: high=(x,v); break
                low=(x,v)
        else:
            high=(x0,v0); x=x0
            for j in range(1,self.args.max_root_iter+1):
                xn=max(xmin,x/1.6); x,v,r,m,s,rep=ev(xn,j,"bracket")
                if r<=self.args.tol_pref: return dict(x=x,m=m,metric=v,rel_err=r,spec=s,rep=rep,status="converged")
                if v<=target: low=(x,v); break
                high=(x,v)
        if low is not None and high is not None:
            lo,hi=low[0],high[0]; base=len(hist)
            for j in range(self.args.max_root_iter):
                mid=0.5*(lo+hi); x,v,r,m,s,rep=ev(mid,base+j,"bisect")
                if r<=self.args.tol_pref: return dict(x=x,m=m,metric=v,rel_err=r,spec=s,rep=rep,status="converged")
                if v<target: lo=x
                else: hi=x
        best=min(hist,key=lambda t:t[2]); status="converged" if best[2]<=self.args.tol else "not_converged"
        return dict(x=best[0],m=best[3],metric=best[1],rel_err=best[2],spec=best[4],rep=best[5],status=status)


    # =============================================================================
    # T00 — Engineering / reproducibility gates
    # =============================================================================
    @torch.no_grad()
    def quantizer_adapter_gate(self) -> list[dict[str, Any]]:
        """Verify that the independent error-analysis mirror matches attention.py.

        T04/T05 use :class:`Grid` to characterize log-weight error while the
        actual forward path is implemented in bta._alt_probs.  A future edit to
        either implementation must not silently invalidate the analysis, so we
        compare normalized probabilities on a small causal synthetic tensor.
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(1701)
        qn = 16
        h = min(2, int(self.model.config.num_attention_heads))
        scores = torch.randn((1, h, qn, qn), generator=g, device=self.device,
                             dtype=torch.float32)
        qi = torch.arange(qn, device=self.device).view(1, 1, qn, 1)
        ki = torch.arange(qn, device=self.device).view(1, 1, 1, qn)
        valid = (ki <= qi).expand_as(scores)
        scores = scores.masked_fill(~valid, float("-inf"))
        rows: list[dict[str, Any]] = []
        tol = float(self.args.adapter_tol)

        for R in [0.25, 0.5, 1.0, 4.0, 8.0]:
            for recon in ["up", "nearest", "lerp"]:
                sp = c_multibit(21, R, recon, "exp", cid=f"adapter_R{R:g}_{recon}")
                configure_condition(self.bta, sp, self.n_layers, self.q_chunk)
                CTRL.reset()
                ref, _ = self.bta._alt_probs(scores, valid, "multibit", qn, layer_idx=0)

                mx = scores.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
                mn = scores.masked_fill(~valid, float("inf")).amin(-1, keepdim=True)
                cov = (mx - mn).clamp(min=1e-6)
                u = (scores - mn).clamp(min=0.0)
                eps, _, _ = Grid(cov, 21, R).recon(u, recon)
                lw = torch.where(
                    valid,
                    u.double() + eps.double(),
                    torch.full_like(u.double(), float("-inf")),
                )
                mine = F.softmax(lw, dim=-1).to(ref.dtype)
                err = float((mine - ref).abs()[valid].max())
                ok = err <= tol
                rows.append(dict(check="quantizer_adapter", R=R, reconstruction=recon,
                                 max_abs_p_diff=err, tolerance=tol, pass_=bool(ok)))
                if not ok:
                    raise RuntimeError(
                        f"quantizer adapter drift R={R:g} recon={recon}: {err:.3e} > {tol:.3e}"
                    )

        for m in [0.5, 1.0, 2.0, 4.0]:
            sp = c_pot(m, cid=f"adapter_pot_m{m:g}")
            configure_condition(self.bta, sp, self.n_layers, self.q_chunk)
            CTRL.reset()
            ref, _ = self.bta._alt_probs(scores, valid, "perturb", qn, layer_idx=0)
            mx = scores.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
            D = (mx - scores).clamp(min=0.0)
            eps = pot_eps(D, m)
            lw = torch.where(valid, -D + eps, torch.full_like(D, float("-inf")))
            mine = F.softmax(lw, dim=-1)
            err = float((mine - ref).abs()[valid].max())
            ok = err <= tol
            rows.append(dict(check="pot_adapter", m=m, max_abs_p_diff=err,
                             tolerance=tol, pass_=bool(ok)))
            if not ok:
                raise RuntimeError(f"PoT adapter drift m={m:g}: {err:.3e} > {tol:.3e}")

        CTRL.reset()
        return rows

    @torch.no_grad()
    def rescue_numerics_gate(self) -> dict[str, Any]:
        """Synthetic audit of P4 partition equality and conditional preservation."""
        g = torch.Generator(device=self.device)
        g.manual_seed(2718)
        qn = 16
        logits = torch.randn((1, 2, qn, qn), generator=g, device=self.device,
                             dtype=torch.float32)
        qi = torch.arange(qn, device=self.device).view(1, 1, qn, 1)
        ki = torch.arange(qn, device=self.device).view(1, 1, 1, qn)
        valid = (ki <= qi).expand_as(logits)
        p = F.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        ph = F.softmax((0.78 * logits).masked_fill(~valid, float("-inf")), dim=-1)
        mask = top_mass_masks(p, valid, [0.9])[0.9]
        pr, err = mass_partition_rescue(p, ph, valid, mask)

        pm = (p * mask).sum(-1, keepdim=True)
        rm = (pr * mask).sum(-1, keepdim=True)
        mass_err = float((pm - rm).abs().max())

        hm = (ph * mask).sum(-1, keepdim=True)
        rmi = (pr * mask).sum(-1, keepdim=True)
        hc = torch.where(mask, ph / hm.clamp_min(1e-30), torch.zeros_like(ph))
        rc = torch.where(mask, pr / rmi.clamp_min(1e-30), torch.zeros_like(pr))
        cond_in = float((hc - rc).abs().max())

        outmask = valid & (~mask)
        ho = (ph * outmask).sum(-1, keepdim=True)
        ro = (pr * outmask).sum(-1, keepdim=True)
        hco = torch.where(outmask, ph / ho.clamp_min(1e-30), torch.zeros_like(ph))
        rco = torch.where(outmask, pr / ro.clamp_min(1e-30), torch.zeros_like(pr))
        # Rows whose complement is numerically empty carry no conditional
        # distribution there and are excluded from this synthetic comparison.
        have_out = (ho.squeeze(-1) > 1e-12)
        if bool(have_out.any()):
            cond_out = float((hco - rco)[have_out].abs().max())
        else:
            cond_out = 0.0

        tol = float(self.args.rescue_gate_tol)
        ok = mass_err <= tol and float(err.max()) <= tol and cond_in <= tol and cond_out <= tol
        row = dict(check="p4_rescue_numerics", mass_error=mass_err,
                   returned_mass_error=float(err.max()), conditional_in_error=cond_in,
                   conditional_out_error=cond_out, tolerance=tol, pass_=bool(ok))
        if not ok:
            raise RuntimeError(f"P4 rescue numerical self-test failed: {row}")
        return row

    @torch.no_grad()
    def _batch_vs_solo_error(self, model, tok, *, impl_label: str) -> dict[str, dict[str, float]]:
        """Measure left/right padding batch-vs-solo error for one loaded model.

        Absolute batch-vs-solo equality is not a valid bf16 criterion because
        matrix shapes change reduction order. The discriminative test is whether
        BTA is materially worse than official eager under the same experiment.
        """
        texts=[
            "The capital of France is Paris.",
            "In a distant galaxy long ago there lived a civilization that recorded every voyage.",
            "Hello.",
            "Machine learning models trained on large corpora often exhibit emergent behaviours.",
        ]
        if tok.pad_token is None:
            tok.pad_token=tok.eos_token
        out={}; old_side=getattr(tok,"padding_side","right")
        try:
            for side in ("right","left"):
                tok.padding_side=side
                enc=tok(texts,return_tensors="pt",padding=True)
                enc={k:v.to(self.device) for k,v in enc.items()}
                bat=model(**enc,use_cache=False,return_dict=True).logits.float()
                worst=0.0; scale=0.0
                for i,t in enumerate(texts):
                    one=tok(t,return_tensors="pt")
                    one={k:v.to(self.device) for k,v in one.items()}
                    solo=model(**one,use_cache=False,return_dict=True).logits.float()[0]
                    n=int(solo.shape[0]); seg=bat[i,-n:] if side=="left" else bat[i,:n]
                    worst=max(worst,float((seg-solo).abs().max()))
                    scale=max(scale,float(solo.abs().max()))
                    del solo
                out[side]=dict(relative_max_error=worst/max(scale,1e-12),
                               absolute_max_error=worst,scale=scale,implementation=impl_label)
                del bat,enc
        finally:
            tok.padding_side=old_side
        return out

    def _release_model(self):
        m=self.model; self.model=None
        if m is not None: del m
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    @torch.no_grad()
    def official_eager_reference_gate(self, ch: torch.Tensor) -> list[dict[str,Any]]:
        """Sequential BTA-vs-official-eager implementation gate.

        BTA with fp32_scores=False must agree closely with official eager.
        BTA with fp32_scores=True is the final protocol and its gap relative to
        eager is reported as a precision choice, not treated as an implementation bug.
        Models are loaded sequentially so this remains feasible on a single 24 GB GPU.
        """
        rows=[]
        if not hasattr(self,"_load_fn"):
            raise RuntimeError("model loader unavailable for official reference gate")

        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk)
        self.bta.CFG.fp32_scores=False; CTRL.reset()
        hb16=backbone_hidden(self.model,ch)
        nll_b16=float(hidden_nll_and_output_metrics(self.model,hb16,ch,ce_chunk=self.args.ce_chunk,
                                                     collect_output=False)["nll"])
        del hb16

        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk)
        self.bta.CFG.fp32_scores=True; CTRL.reset()
        h32=backbone_hidden(self.model,ch)
        nll_f32=float(hidden_nll_and_output_metrics(self.model,h32,ch,ce_chunk=self.args.ce_chunk,
                                                     collect_output=False)["nll"])
        del h32

        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk)
        self.bta.CFG.fp32_scores=False; CTRL.reset()
        pad_bta=self._batch_vs_solo_error(self.model,self.tok,impl_label="bta_bf16_scores")

        self._release_model(); eager=None
        try:
            eager,tok_e=self._load_fn(self.hf_model,impl="eager",trust_remote_code=self.args.trust_remote_code)
            eager.eval(); self.device=model_device(eager); ch_e=ch.to(self.device)
            nll_eager=float(standard_nll(eager,ch_e))
            pad_eager=self._batch_vs_solo_error(eager,tok_e,impl_label="official_eager")
        finally:
            if eager is not None: del eager
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        self.model,self.tok=self._load_fn(self.hf_model,impl=self._attn_impl,trust_remote_code=self.args.trust_remote_code)
        self.model.eval(); self.device=model_device(self.model)
        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk)
        self.bta.CFG.fp32_scores=True; CTRL.reset()

        ppl_b16=math.exp(nll_b16); ppl_e=math.exp(nll_eager)
        rel_impl=abs(ppl_b16-ppl_e)/max(ppl_e,1e-30)
        impl_ok=rel_impl<=float(self.args.impl_rel_tol)
        rows.append(dict(check="bta_bf16_vs_official_eager",bta_bf16_nll=nll_b16,
                         eager_nll=nll_eager,bta_bf16_ppl=ppl_b16,eager_ppl=ppl_e,
                         relative_ppl_diff=rel_impl,tolerance=float(self.args.impl_rel_tol),
                         pass_=bool(impl_ok)))
        if not impl_ok:
            raise RuntimeError(f"BTA implementation gate failed: bf16-score BTA vs eager relative PPL diff={rel_impl:.3e} > {self.args.impl_rel_tol:.3e}. Check architecture-specific mask/GQA/attention handling.")

        fp32_gap=(math.exp(nll_f32)-ppl_e)/max(ppl_e,1e-30)
        rows.append(dict(check="fp32_score_precision_gap_vs_eager",bta_fp32_nll=nll_f32,
                         eager_nll=nll_eager,relative_ppl_gap=fp32_gap,pass_=True,
                         interpretation="descriptive precision difference, not implementation failure"))

        for side in ("right","left"):
            rb=float(pad_bta[side]["relative_max_error"]); re_=float(pad_eager[side]["relative_max_error"])
            ratio=rb/max(re_,1e-12); ok=ratio<=float(self.args.padding_ratio_tol)
            rows.append(dict(check=f"{side}_padding_bta_vs_eager_error_ratio",
                             bta_relative_error=rb,eager_relative_error=re_,ratio=ratio,
                             tolerance_ratio=float(self.args.padding_ratio_tol),pass_=bool(ok)))
            if not ok and not self.args.allow_padding_gate_failure:
                raise RuntimeError(f"{side}-padding mask gate failed: BTA/eager error ratio {ratio:.2f}x > {self.args.padding_ratio_tol:.2f}x")
        return rows

    @torch.no_grad()
    def probe_sparsity_identity_gate(self,ch:torch.Tensor) -> dict[str,Any]:
        """Verify probe_sparsity measures support but preserves exact-softmax output."""
        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk)
        CTRL.reset()
        h0=backbone_hidden(self.model,ch).float()

        sp=c_binary(True)
        configure_condition(self.bta,sp,self.n_layers,self.q_chunk)
        self.bta.CFG.probe_sparsity=True
        acc=SupportAccumulator()
        CTRL.reset(); CTRL.condition=sp; CTRL.support=acc
        h1=backbone_hidden(self.model,ch).float()
        rep=acc.report()

        absd=float((h1-h0).abs().max())
        scale=float(h0.abs().max())
        reld=absd/max(scale,1e-12)
        ok=reld<=float(self.args.probe_identity_rel_tol)
        CTRL.reset()
        del h0,h1
        return dict(
            check="probe_sparsity_output_identity",
            max_hidden_abs_diff=absd,max_hidden_rel_diff=reld,
            tolerance=float(self.args.probe_identity_rel_tol),
            measured_support_density=rep.get("density"),
            n_layers_with_support=len(rep.get("per_layer",{})),
            pass_=bool(ok),
        )

    @torch.no_grad()
    def p4_runtime_shape_gate(self,ch:torch.Tensor) -> dict[str,Any]:
        """Exercise actual GQA/q_chunk pV instrumentation across all layers."""
        sp=c_temp(0.9,cid="gate_p4_shape")
        configure_condition(self.bta,sp,self.n_layers,self.q_chunk)
        min_n=max(2,min(int(self.args.p4_min_n),max(2,int(ch.shape[-1])//4)))
        acc=P4Accumulator([0.9],0.9,min_n=min_n)
        CTRL.reset()
        CTRL.condition=sp
        CTRL.p4_probe=True
        CTRL.p4=acc
        CTRL.p4_qs=[0.9]
        CTRL.p4_primary_q=0.9
        CTRL.p4_min_n=min_n
        CTRL.head_chunk=max(1,int(self.args.p4_head_chunk))
        hh=backbone_hidden(self.model,ch)
        del hh
        layers=sorted({int(r["layer"]) for r in acc.output_rows(sp.id)})
        ok=(len(layers)==self.n_layers)
        CTRL.reset()
        return dict(
            check="p4_pv_runtime_shape_gate",
            layers_seen=len(layers),expected_layers=self.n_layers,
            layer_min=(min(layers) if layers else None),
            layer_max=(max(layers) if layers else None),
            pass_=bool(ok),
        )

    def run_T00(self):
        self.load_runtime(); td=self.store.task_dir("T00")
        gate_len=min(int(self.args.gate_len),int(self.args.seq),len(self.ids))
        ch=self.ids[:gate_len].unsqueeze(0).to(self.device)
        rows=[]
        rows.extend(self.qchunk_gate_rows)

        # Pure numerical adapter checks protect the independent T04/T05 error
        # analysis and P4 rescue code from implementation drift in attention.py.
        rows.extend(self.quantizer_adapter_gate())
        rows.append(self.rescue_numerics_gate())

        # Explicitly validate assumptions that the early scripts relied on.
        pr=self.probe_sparsity_identity_gate(ch)
        rows.append(pr)
        if not pr["pass_"]:
            raise RuntimeError(
                f"probe_sparsity identity gate failed: hidden_rel={pr['max_hidden_rel_diff']:.3e}"
            )
        pg=self.p4_runtime_shape_gate(ch)
        rows.append(pg)
        if not pg["pass_"]:
            raise RuntimeError(
                f"P4 pV runtime shape gate failed: layers={pg['layers_seen']}/{pg['expected_layers']}"
            )
        sc=getattr(self.model.config,"attn_logit_softcapping",None)
        sc_ok=bool(sc is None or getattr(self.bta,"SUPPORTS_ATTN_LOGIT_SOFTCAPPING",False))
        rows.append(dict(
            check="attention_logit_softcapping_support",
            configured_value=(None if sc is None else float(sc)),
            bta_declares_support=bool(getattr(self.bta,"SUPPORTS_ATTN_LOGIT_SOFTCAPPING",False)),
            pass_=sc_ok,
        ))
        if not sc_ok:
            raise RuntimeError(
                "Model config uses attn_logit_softcapping but the attention backend does not declare support."
            )

        rows.extend(self.official_eager_reference_gate(ch))

        hq=int(self.model.config.num_attention_heads)
        hkv=int(getattr(self.model.config,"num_key_value_heads",hq))
        gqa_ok=(hkv>0 and hq % hkv == 0)
        rows.append(dict(check="gqa_head_mapping",num_attention_heads=hq,
                         num_key_value_heads=hkv,groups=(hq//hkv if gqa_ok else None),
                         pass_=bool(gqa_ok)))
        if not gqa_ok:
            raise RuntimeError(f"invalid Q/KV head mapping hq={hq}, hkv={hkv}")

        # Safe chunked LM-head path vs the model's own forward loss.
        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk); CTRL.reset()
        with torch.no_grad():
            h=backbone_hidden(self.model,ch)
            safe=hidden_nll_and_output_metrics(self.model,h,ch,ce_chunk=self.args.ce_chunk,
                                               collect_output=False)["nll"]
            std=standard_nll(self.model,ch)
        diff=abs(float(safe)-float(std))
        rows.append(dict(check="safe_nll_vs_model_forward",safe_nll=safe,standard_nll=std,
                         abs_diff=diff,pass_=bool(diff<=self.args.safe_nll_tol)))
        if diff>self.args.safe_nll_tol:
            raise RuntimeError(
                f"safe NLL projection mismatch {diff:.3e} > {self.args.safe_nll_tol:.3e}. "
                "Add the model family's final-logit transform before formal collection."
            )

        # q_chunk equivalence on three representative paths.
        for spec in [
            c_baseline(),
            c_multibit(21,4,"nearest","exp",cid="gate_R4K21"),
            c_temp(0.9,cid="gate_temp_a0p9"),
        ]:
            vals=[]; href=None; h_abs=0.0; h_rel=0.0
            for qi,qc in enumerate([0,self.q_chunk]):
                configure_condition(self.bta,spec,self.n_layers,qc); CTRL.reset()
                hh=backbone_hidden(self.model,ch)
                vals.append(hidden_nll_and_output_metrics(self.model,hh,ch,ce_chunk=self.args.ce_chunk,
                                                          collect_output=False)["nll"])
                hf=hh.float()
                if qi==0:
                    href=hf.detach().cpu()
                else:
                    cur=hf.detach().cpu(); h_abs=float((cur-href).abs().max())
                    h_rel=h_abs/max(float(href.abs().max()),1e-12)
                del hh,hf
            d=abs(vals[1]-vals[0])
            ok=(d<=self.args.chunk_tol and h_rel<=self.args.chunk_hidden_rel_tol)
            rows.append(dict(check="q_chunk_equivalence",condition=spec.id,q0_nll=vals[0],
                             qchosen_nll=vals[1],abs_nll_diff=d,max_hidden_abs_diff=h_abs,
                             max_hidden_rel_diff=h_rel,nll_tolerance=self.args.chunk_tol,
                             hidden_rel_tolerance=self.args.chunk_hidden_rel_tol,pass_=bool(ok)))
            if not ok:
                raise RuntimeError(f"q_chunk gate failed for {spec.id}: dNLL={d:.3e}, hidden_rel={h_rel:.3e}")

        # Temperature alpha=1 identity, including model-output distribution metrics.
        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk); CTRL.reset()
        hb=backbone_hidden(self.model,ch)
        ident=c_temp(1.0,cid="temp_identity")
        configure_condition(self.bta,ident,self.n_layers,self.q_chunk); CTRL.reset()
        CTRL.condition=ident; CTRL.div=DivAccumulator(min_n=1)
        hi=backbone_hidden(self.model,ch)
        attn_id=CTRL.div.report()
        expected_rows=self.n_layers*hq*gate_len
        row_ok=(int(attn_id["n_rows"])==int(expected_rows) and
                float(attn_id["js_mean"])<=self.args.identity_div_tol and
                float(attn_id["tv_mean"])<=10*self.args.identity_div_tol)
        rows.append(dict(check="attention_identity_valid_rows",
                         observed_rows=int(attn_id["n_rows"]),expected_rows=int(expected_rows),
                         attention_jsd=float(attn_id["js_mean"]),attention_tv=float(attn_id["tv_mean"]),
                         pass_=bool(row_ok)))
        if not row_ok:
            raise RuntimeError(f"attention identity/valid-row audit failed: {rows[-1]}")
        CTRL.reset()
        im=hidden_nll_and_output_metrics(self.model,hi,ch,h_base=hb,ce_chunk=self.args.ce_chunk,
                                         collect_output=True)
        bmet=hidden_nll_and_output_metrics(self.model,hb,ch,ce_chunk=self.args.ce_chunk,
                                           collect_output=False)
        id_dnll=float(im["nll"])-float(bmet["nll"])
        ok=abs(id_dnll)<=self.args.identity_tol and im["output_jsd"]<=self.args.identity_div_tol
        rows.append(dict(check="temperature_identity",delta_nll=id_dnll,
                         output_kl_forward=im["output_kl_forward"],output_jsd=im["output_jsd"],
                         top1_agreement=im["top1_agreement"],pass_=bool(ok)))
        if not ok:
            raise RuntimeError(f"temperature identity failed: dNLL={id_dnll:.3e}, JSD={im['output_jsd']:.3e}")
        del hb,hi,h

        # Formal baseline: all complete paired blocks, exported as a first-class
        # condition so the open-source release contains the requested baseline
        # NLL/PPL and block-level values rather than only internal references.
        self.evaluate_group([c_baseline()],task="T00")
        bsum=self.condition_summary(c_baseline())
        write_json(td/"baseline_summary.json",bsum)

        write_csv(td/"engineering_checks.csv",rows)
        write_json(td/"status.json",dict(status="complete",gate_len=gate_len,checks=rows,baseline=bsum))
        write_json(self.store.model_dir/"engineering_checks.json",dict(checks=rows))
        print(f"[T00] PASS baseline_nll={bsum['nll']:.7f} baseline_ppl={bsum['ppl']:.6f}")

    @torch.no_grad()
    def padding_gate(self) -> dict[str,Any]:
        texts=[
            "The capital of France is Paris.",
            "In a distant galaxy long ago there lived a civilization that recorded every voyage.",
            "Hello.",
            "Machine learning models trained on large corpora often exhibit emergent behaviours.",
        ]
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token
        self.tok.padding_side="left"
        enc=self.tok(texts,return_tensors="pt",padding=True)
        enc={k:v.to(self.device) for k,v in enc.items()}
        configure_condition(self.bta,c_baseline(),self.n_layers,self.q_chunk); CTRL.reset()
        bat=self.model(**enc,use_cache=False,return_dict=True).logits.float()
        worst=0.0; scale=0.0
        for i,t in enumerate(texts):
            one=self.tok(t,return_tensors="pt")
            one={k:v.to(self.device) for k,v in one.items()}
            solo=self.model(**one,use_cache=False,return_dict=True).logits.float()[0]
            n=solo.shape[0]; seg=bat[i,-n:]
            worst=max(worst,float((seg-solo).abs().max()))
            scale=max(scale,float(solo.abs().max()))
        rel=worst/max(scale,1e-12)
        # Historical eager-vs-batch bf16 differences can be a few percent; a
        # broken left-padding mask is dramatically worse. Keep this deliberately
        # loose and record the exact value.
        ok=rel<=self.args.padding_rel_tol
        return dict(check="left_padding_batch_vs_solo",relative_max_error=rel,
                    absolute_max_error=worst,tolerance=self.args.padding_rel_tol,pass_=bool(ok))

    # =============================================================================
    # T01 — Support characterization
    # =============================================================================
    def run_T01(self):
        self.load_runtime(); td=self.store.task_dir("T01")
        specs=[c_binary(True),c_binary(False)]
        if "deep_reference" in self.features:
            for r in [0.50,0.25,0.15,0.10,0.065,0.026,0.01,0.0065]:
                specs += [c_topk(r,True),c_topk(r,False)]
        bs_path=td/"baseline_mean_support.json"
        self.evaluate_group(specs,task="T01",collect_support_ids=[s.id for s in specs],
                            baseline_mean_support_path=bs_path)
        bs_probe=read_json(bs_path)
        baseline_density=float(bs_probe["support"]["density"])
        rows=[self.condition_summary(s) for s in specs]
        for r in rows:
            r["baseline_mean_support_density"]=baseline_density
        write_csv(td/"conditions.csv",rows)

        effects=[]
        a=self.delta_series(c_binary(True)); b=self.delta_series(c_binary(False))
        est,lo,hi=Bootstrapper.mean_ci(b-a,self.args.boot,self.args.boot_seed)
        effects.append(dict(effect="same_support_uniform_minus_softmax",estimate=est,ci_low=lo,ci_high=hi))
        if "deep_reference" in self.features:
            for r in [0.50,0.25,0.15,0.10,0.065,0.026,0.01,0.0065]:
                ss=c_topk(r,True); uu=c_topk(r,False)
                est,lo,hi=Bootstrapper.mean_ci(self.delta_series(uu)-self.delta_series(ss),
                                               self.args.boot,self.args.boot_seed)
                effects.append(dict(effect=f"topk_{r:g}_uniform_minus_softmax",ratio=r,
                                    estimate=est,ci_low=lo,ci_high=hi))
        write_csv(td/"effects.csv",effects)
        print(f"[T01] {len(specs)} conditions complete")

    # =============================================================================
    # T01B — Strict single-layer same-support weighting control
    # =============================================================================
    def run_T01B(self):
        """Strictly separate support selection from weighting.

        For every target layer l, both arms are baseline in layers < l. At l,
        both therefore see the same hidden state, Q/K scores, valid mask, and
        deterministic mean-threshold support M. The soft arm keeps learned
        relative softmax weights inside M; the uniform arm replaces only those
        within-M weights by a uniform distribution. Layers > l return to exact
        baseline attention.

        Layers are a fixed intervention set, not independent statistical units.
        The aggregate endpoint averages the paired uniform-minus-softmax effect
        across layers within each aligned block, then bootstraps the blocks.
        """
        self.load_runtime(); td=self.store.task_dir("T01B")
        layers=list(range(self.n_layers))
        specs=[]
        for li in layers:
            specs += [c_strict_same_support(li,True), c_strict_same_support(li,False)]

        self.evaluate_group(
            specs, task="T01B", collect_support_ids=[sp.id for sp in specs]
        )
        write_csv(td/"conditions.csv",[self.condition_summary(sp) for sp in specs])

        effects=[]; audits=[]; block_effects=[]
        for li in layers:
            ss=c_strict_same_support(li,True)
            uu=c_strict_same_support(li,False)
            ds=self.delta_series(ss); du=self.delta_series(uu)
            if ds.shape != du.shape:
                raise RuntimeError(
                    f"T01B paired block shape mismatch layer={li}: {ds.shape} vs {du.shape}"
                )
            diff=du-ds
            est,lo,hi=Bootstrapper.mean_ci(diff,self.args.boot,self.args.boot_seed)
            effects.append(dict(
                target_layer=li,
                effect="strict_same_support_uniform_minus_softmax",
                estimate=est,ci_low=lo,ci_high=hi,n_blocks=int(diff.size),
            ))
            for bi,val in enumerate(diff):
                block_effects.append(dict(block=int(bi),target_layer=li,effect=float(val)))

            ms=read_json(self.store.condition_meta_path(ss.id,self.args.seq),{}) or {}
            mu=read_json(self.store.condition_meta_path(uu.id,self.args.seq),{}) or {}
            rs=dict(ms.get("support") or {}); ru=dict(mu.get("support") or {})
            if not rs or not ru:
                raise RuntimeError(f"T01B support metadata missing at layer={li}")
            ps=set((rs.get("per_layer") or {}).keys())
            pu=set((ru.get("per_layer") or {}).keys())
            expected={str(li)}
            layer_only_ok=(ps==expected and pu==expected)
            kept_s=float(rs.get("kept",float("nan"))); kept_u=float(ru.get("kept",float("nan")))
            valid_s=float(rs.get("valid",float("nan"))); valid_u=float(ru.get("valid",float("nan")))
            dens_s=float(rs.get("density",float("nan"))); dens_u=float(ru.get("density",float("nan")))
            counts_equal=(
                np.isfinite(kept_s) and np.isfinite(kept_u) and
                np.isfinite(valid_s) and np.isfinite(valid_u) and
                kept_s==kept_u and valid_s==valid_u
            )
            pass_=bool(layer_only_ok and counts_equal)
            audits.append(dict(
                target_layer=li,soft_condition=ss.id,uniform_condition=uu.id,
                soft_support_density=dens_s,uniform_support_density=dens_u,
                density_abs_diff=abs(dens_s-dens_u),
                soft_kept=kept_s,uniform_kept=kept_u,kept_diff=kept_u-kept_s,
                soft_valid=valid_s,uniform_valid=valid_u,valid_diff=valid_u-valid_s,
                soft_support_layers=",".join(sorted(ps)),
                uniform_support_layers=",".join(sorted(pu)),
                layer_only_pass=bool(layer_only_ok),equal_count_pass=bool(counts_equal),
                pass_=pass_,
            ))
            if not pass_:
                raise RuntimeError(
                    "T01B strict support audit failed at layer="
                    f"{li}: soft kept/valid=({kept_s},{valid_s}) "
                    f"uniform=({kept_u},{valid_u}) support_layers={ps}/{pu}"
                )

        write_csv(td/"effects_by_layer.csv",effects)
        write_csv(td/"support_audit.csv",audits)
        write_csv(td/"block_effects_by_layer.csv",block_effects)

        bd=pd.DataFrame(block_effects)
        piv=bd.pivot(index="block",columns="target_layer",values="effect").sort_index()
        if list(piv.columns)!=layers:
            raise RuntimeError(
                f"T01B aggregate missing layer columns: have={list(piv.columns)} expected={layers}"
            )
        agg_by_block=piv.mean(axis=1).to_numpy(np.float64)
        est,lo,hi=Bootstrapper.mean_ci(agg_by_block,self.args.boot,self.args.boot_seed)
        aggregate=[dict(
            effect="D_W_strict_same_support_uniform_minus_softmax",
            definition="mean over fixed target-layer set within each block; paired bootstrap over blocks",
            estimate=est,ci_low=lo,ci_high=hi,n_layers=len(layers),
            n_blocks=int(agg_by_block.size),
            support_identity=(
                "same pre-target baseline trajectory + same deterministic mean-threshold rule; "
                "exact selected/valid counts audited equal in both arms"
            ),
        )]
        write_csv(td/"aggregate_effect.csv",aggregate)
        write_csv(td/"aggregate_block_effects.csv",[
            dict(block=int(b),effect=float(v)) for b,v in zip(piv.index,agg_by_block)
        ])
        print(
            f"[T01B] layers={len(layers)} strict same-support control complete; "
            f"D_W={est:+.6g} [{lo:+.6g},{hi:+.6g}]"
        )

    # =============================================================================
    # T02 — Weighting geometry
    # =============================================================================
    def run_T02(self):
        self.load_runtime(); td=self.store.task_dir("T02")
        Ks=[16]
        if "deep_reference" in self.features:
            Ks=[4,8,12,16,24,32,48,64,128,256,512,1024]
        specs=[]
        # Canonical geometry control: full support, full row range, uniform
        # partition (R=1), up reconstruction. The ONLY changed property is the
        # level-to-weight mapping (exp vs linear).
        for K in Ks:
            specs.append(c_multibit(K,1,"up","exp",role="geometry",cid=f"geometry_exp_K{K}"))
            specs.append(c_multibit(K,1,"up","linear",role="geometry",cid=f"geometry_linear_K{K}"))
        self.evaluate_group(specs,task="T02")
        rows=[self.condition_summary(s) for s in specs]
        write_csv(td/"conditions.csv",rows)
        effects=[]
        for K in Ks:
            e=c_multibit(K,1,"up","exp",cid=f"geometry_exp_K{K}")
            l=c_multibit(K,1,"up","linear",cid=f"geometry_linear_K{K}")
            est,lo,hi=Bootstrapper.mean_ci(self.delta_series(l)-self.delta_series(e),
                                           self.args.boot,self.args.boot_seed)
            effects.append(dict(K=K,effect="linear_minus_exp",estimate=est,ci_low=lo,ci_high=hi))
        write_csv(td/"effects.csv",effects)
        print(f"[T02] geometry K grid={Ks}")

    # =============================================================================
    # T03 — Full-range precision
    # =============================================================================
    def run_T03(self):
        self.load_runtime(); td=self.store.task_dir("T03")
        Ks=[16,32,64,128,256] if "deep_reference" in self.features else [32]
        specs=[c_multibit(K,4,"nearest","exp",role="precision",cid=f"precision_K{K}_R4") for K in Ks]
        self.evaluate_group(specs,task="T03")
        write_csv(td/"conditions.csv",[self.condition_summary(s) for s in specs])
        print(f"[T03] full-range precision K={Ks}")

    # =============================================================================
    # Probe helpers for T04/T05/T06
    # =============================================================================
    def _resolved_error_probe_windows(self) -> int:
        if self.args.fast:
            return 1
        if "deep_reference" in self.features:
            return (int(self.args.error_probe_windows) if int(self.args.error_probe_windows)>0
                    else self.n_blocks(self.args.seq))
        return (int(self.args.error_probe_windows) if int(self.args.error_probe_windows)>0
                else min(int(self.args.verify_windows),self.n_blocks(self.args.seq)))

    @torch.no_grad()
    def ensure_deep_numerical_probe_bundle(self) -> dict[str,Any]:
        """One baseline-trajectory pass for all deep entrywise numerical probes.

        Sweep21-style error geometry, reconstruction diagnostics, PoT error
        distributions, attention-mass concentration, and Rowmax tail coverage
        all depend on the SAME exact local-softmax score tensors. Running them
        together avoids three separate 200k-token model passes and guarantees
        that every diagnostic sees an identical trajectory.
        """
        if "deep_reference" not in self.features:
            raise RuntimeError("deep numerical probe bundle is deep_reference only")
        self.load_runtime()
        outdir=self.store.model_dir/"shared_numerical_probe"
        outdir.mkdir(parents=True,exist_ok=True)
        alloc=[c_multibit(21,R,"nearest","exp",role="allocation",cid=f"alloc_K21_R{R:g}")
               for R in [0.25,0.5,1,2,4,8]]
        recon=[c_multibit(21,4,m,"exp",role="reconstruction",cid=f"recon_K21_R4_{m}")
               for m in ["up","nearest","lerp"]]
        pots=[c_pot(m,role="pot_diagnostic",cid=f"pot_diag_m{m:g}") for m in [0.5,1,2,4]]
        specs=[]; seen=set()
        for sp in alloc+recon+pots:
            if sp.id not in seen:
                specs.append(sp); seen.add(sp.id)
        rowmax_Ks=[12,14,16,20,24,32,48]
        nw=self._resolved_error_probe_windows()
        sig_obj=dict(
            condition_specs=[sp.serializable() for sp in specs],rowmax_Ks=rowmax_Ks,
            windows=nw,seq_len=self.args.seq,q_chunk=self.q_chunk,
            token_signature=token_signature(self.ids),model_revision=self.protocol_base.get("model_revision"),
            implementation_signature=self.implementation_signature,
            min_n=self.args.error_min_n,min_span=self.args.error_min_span,
        )
        sig=short_hash(sig_obj,16)
        mp=outdir/"manifest.json"
        files=dict(error=outdir/"error_geometry.csv",bins=outdir/"error_bins_delta.csv",
                   mass=outdir/"mass_concentration.csv",tail=outdir/"rowmax_tail_probe.csv")
        if mp.exists() and all(x.exists() for x in files.values()):
            old=read_json(mp,{}) or {}
            if old.get("signature")==sig:
                return dict(
                    signature=sig,windows=nw,
                    error=pd.read_csv(files["error"]).to_dict("records"),
                    bins=pd.read_csv(files["bins"]).to_dict("records"),
                    mass=pd.read_csv(files["mass"]).to_dict("records"),
                    tail=pd.read_csv(files["tail"]).to_dict("records"),
                )
            raise RuntimeError(
                "shared_numerical_probe cache has a different protocol signature; use --overwrite"
            )

        dummy=specs[0]
        configure_condition(self.bta,dummy,self.n_layers,self.q_chunk)
        self.bta.CFG.probe_sparsity=True
        CTRL.reset(); CTRL.condition=dummy
        accs=[ErrorGeometryAccumulator(sp,min_n=self.args.error_min_n,
                                       min_span=self.args.error_min_span) for sp in specs]
        CTRL.error=ErrorGeometryGroup(accs)
        CTRL.mass_conc=MassConcentrationAccumulator()
        CTRL.rowmax_tail=RowmaxTailAccumulator(rowmax_Ks)
        t0=time.time()
        for w in range(nw):
            ch=self.get_block(w,self.args.seq)
            _=get_backbone(self.model)(input_ids=ch,use_cache=False,return_dict=True)
            del ch
            if (w+1)%max(1,min(10,nw))==0 or w+1==nw:
                print(f"  [shared numerical probe] window {w+1:3d}/{nw:3d} "
                      f"error_conditions={len(specs)} elapsed={time.time()-t0:.0f}s")
        er=[a.report() for a in accs]
        eb=[]
        for a in accs: eb.extend(a.bin_rows())
        mc=CTRL.mass_conc.rows()
        tr=CTRL.rowmax_tail.rows()
        CTRL.reset()
        write_csv(files["error"],er); write_csv(files["bins"],eb)
        write_csv(files["mass"],mc); write_csv(files["tail"],tr)
        write_json(mp,dict(signature=sig,protocol=sig_obj,created=time.time()))
        return dict(signature=sig,windows=nw,error=er,bins=eb,mass=mc,tail=tr)

    @torch.no_grad()
    def error_geometry_probe(self,specs:list[ConditionSpec],task:str,
                             *, include_mass_concentration=False) -> tuple[list[dict],list[dict],list[dict]]:
        self.load_runtime(); summaries=[]; bins=[]; massrows=[]
        if not specs:
            return summaries,bins,massrows

        if "deep_reference" in self.features:
            bundle=self.ensure_deep_numerical_probe_bundle()
            er_by={str(r["condition"]):r for r in bundle["error"]}
            ids={sp.id for sp in specs}
            missing=[sp.id for sp in specs if sp.id not in er_by]
            if missing:
                raise RuntimeError(f"deep numerical probe bundle missing conditions: {missing}")
            summaries=[er_by[sp.id] for sp in specs]
            bins=[r for r in bundle["bins"] if str(r["condition"]) in ids]
            if include_mass_concentration:
                massrows=bundle["mass"]
            return summaries,bins,massrows

        # Replication profiles: only T06's m=1 numerical diagnostic uses this
        # helper. Keep it on the verify-window budget instead of duplicating a
        # 200k-token deep characterization on every model.
        nw=self._resolved_error_probe_windows()
        dummy=specs[0]
        configure_condition(self.bta,dummy,self.n_layers,self.q_chunk)
        self.bta.CFG.probe_sparsity=True
        CTRL.reset(); CTRL.condition=dummy
        accs=[ErrorGeometryAccumulator(sp,min_n=self.args.error_min_n,
                                       min_span=self.args.error_min_span) for sp in specs]
        CTRL.error=ErrorGeometryGroup(accs)
        if include_mass_concentration:
            CTRL.mass_conc=MassConcentrationAccumulator()
        t0=time.time()
        for w in range(nw):
            ch=self.get_block(w,self.args.seq)
            _=get_backbone(self.model)(input_ids=ch,use_cache=False,return_dict=True)
            del ch
            if (w+1)%max(1,min(10,nw))==0 or w+1==nw:
                print(f"  [{task} error-probe] window {w+1:3d}/{nw:3d} conditions={len(specs)} "
                      f"elapsed={time.time()-t0:.0f}s")
        summaries=[a.report() for a in accs]
        for a in accs: bins.extend(a.bin_rows())
        if CTRL.mass_conc is not None: massrows=CTRL.mass_conc.rows()
        CTRL.reset()
        return summaries,bins,massrows

    # =============================================================================
    # T04 — Precision allocation
    # =============================================================================
    def run_T04(self):
        self.load_runtime(); td=self.store.task_dir("T04")
        Rs=[0.25,0.5,1,2,4,8] if "deep_reference" in self.features else [1,4]
        specs=[c_multibit(21,R,"nearest","exp",role="allocation",cid=f"alloc_K21_R{R:g}") for R in Rs]
        self.evaluate_group(specs,task="T04")
        rows=[self.condition_summary(s) for s in specs]
        write_csv(td/"conditions.csv",rows)

        if "deep_reference" in self.features:
            er,eb,mc=self.error_geometry_probe(specs,"T04",include_mass_concentration=True)
            write_csv(td/"error_geometry.csv",er); write_csv(td/"error_bins_delta.csv",eb)
            write_csv(td/"mass_concentration.csv",mc)

        r1=c_multibit(21,1,"nearest","exp",cid="alloc_K21_R1")
        r4=c_multibit(21,4,"nearest","exp",cid="alloc_K21_R4")
        d=self.delta_series(r1)-self.delta_series(r4)
        est,lo,hi=Bootstrapper.mean_ci(d,self.args.boot,self.args.boot_seed)
        effects=[dict(effect="D_R_K21_R1_minus_R4",K=21,estimate=est,ci_low=lo,ci_high=hi)]

        if "deep_reference" in self.features or "allocation_extended" in self.features:
            extra=[]
            for K in [16,21,32]:
                for R in [1,4]:
                    sp=c_multibit(K,R,"nearest","exp",role="allocation_budget",cid=f"alloc_K{K}_R{R}")
                    if sp.id not in {x.id for x in specs}: extra.append(sp)
            self.evaluate_group(extra,task="T04") if extra else None
            D={}
            for K in [16,21,32]:
                a=c_multibit(K,1,"nearest","exp",cid=f"alloc_K{K}_R1")
                b=c_multibit(K,4,"nearest","exp",cid=f"alloc_K{K}_R4")
                x=self.delta_series(a)-self.delta_series(b)
                e,l,h=Bootstrapper.mean_ci(x,self.args.boot,self.args.boot_seed)
                D[K]=x; effects.append(dict(effect=f"D_R_K{K}",K=K,estimate=e,ci_low=l,ci_high=h))
            inter=D[16]-D[32]
            e,l,h=Bootstrapper.mean_ci(inter,self.args.boot,self.args.boot_seed)
            effects.append(dict(effect="allocation_interaction_D16_minus_D32",estimate=e,ci_low=l,ci_high=h))
        write_csv(td/"effects.csv",effects)
        print(f"[T04] allocation R={Rs}; D_R={est:+.6g} [{lo:+.6g},{hi:+.6g}]")

    # =============================================================================
    # T05 — Reconstruction geometry
    # =============================================================================
    def run_T05(self):
        if "deep_reference" not in self.features:
            print("[T05] skipped: deep_reference only"); return
        self.load_runtime(); td=self.store.task_dir("T05")
        specs=[c_multibit(21,4,m,"exp",role="reconstruction",cid=f"recon_K21_R4_{m}")
               for m in ["up","nearest","lerp"]]
        self.evaluate_group(specs,task="T05")
        write_csv(td/"conditions.csv",[self.condition_summary(s) for s in specs])
        er,eb,_=self.error_geometry_probe(specs,"T05")
        write_csv(td/"error_geometry.csv",er); write_csv(td/"error_bins_delta.csv",eb)
        audit=[]
        for s,r in zip(specs,er):
            ratio=float(r["max_over_bound"]); ok=ratio<=1.0+self.args.bound_ratio_tol
            audit.append(dict(condition=s.id,reconstruction=s.params["reconstruction"],
                              empirical_max_abs_eps=r["max_abs_eps"],
                              max_over_bound=ratio,tolerance_ratio=1.0+self.args.bound_ratio_tol,pass_=bool(ok)))
            if not ok:
                raise RuntimeError(f"T05 analytical bound audit failed for {s.id}: ratio={ratio:.6g}")
        write_csv(td/"analytic_audit.csv",audit)
        print("[T05] reconstruction up/nearest/lerp complete")

    # =============================================================================
    # T06 — PoT diagnostic + model-specific K0 + practical Rowmax-PoT
    # =============================================================================
    @torch.no_grad()
    def rowmax_tail_probe(self, Ks:list[int]) -> list[dict[str,Any]]:
        self.load_runtime()
        if "deep_reference" in self.features:
            bundle=self.ensure_deep_numerical_probe_bundle()
            by={int(r["K"]):r for r in bundle["tail"]}
            miss=[k for k in Ks if int(k) not in by]
            if miss: raise RuntimeError(f"deep Rowmax tail bundle missing K={miss}")
            return [by[int(k)] for k in Ks]

        # One common exact-softmax trajectory suffices for every K because K
        # changes only the tail cutoff. Use a Rowmax-PoT dummy operator so
        # _alt_probs exposes scores, then force the actual output back to softmax.
        dummy=c_rowmax_pot(max(Ks),cid="rowmax_tail_probe")
        configure_condition(self.bta,dummy,self.n_layers,self.q_chunk)
        self.bta.CFG.probe_sparsity=True
        CTRL.reset(); CTRL.condition=dummy; CTRL.rowmax_tail=RowmaxTailAccumulator(Ks)
        nw=self._resolved_error_probe_windows()
        for w in range(nw):
            ch=self.get_block(w,self.args.seq)
            _=get_backbone(self.model)(input_ids=ch,use_cache=False,return_dict=True)
            del ch
        rows=CTRL.rowmax_tail.rows()
        CTRL.reset()
        return rows

    def run_T06(self):
        self.load_runtime(); td=self.store.task_dir("T06")
        pot_ms=[0.5,1,2,4] if "deep_reference" in self.features else [1]
        # Diagnostic error geometry on baseline trajectory.
        pot_specs=[c_pot(m,role="pot_diagnostic",cid=f"pot_diag_m{m:g}") for m in pot_ms]
        er,eb,_=self.error_geometry_probe(pot_specs,"T06")
        write_csv(td/"pot_error_geometry.csv",er); write_csv(td/"pot_error_bins_delta.csv",eb)
        pot_audit=[]
        for r in er:
            empirical=float(r["max_abs_eps"])
            theoretical=float(r["theoretical_bound"])
            ratio=float(r["max_over_bound"])
            abs_excess=max(empirical-theoretical,0.0)
            # Two complementary tolerances:
            #   (1) the existing relative allowance, useful for the larger-magnitude bounds;
            #   (2) a tiny absolute fp32 allowance in nats, needed at large m where the
            #       theoretical bound itself is small and max-statistic roundoff dominates.
            rel_abs_allow=theoretical*float(self.args.bound_ratio_tol)
            allowed_abs_excess=max(rel_abs_allow,POT_FP32_BOUND_ABS_TOL)
            ok=abs_excess<=allowed_abs_excess
            pot_audit.append(dict(
                condition=r["condition"],
                empirical_max_abs_eps=empirical,
                theoretical_bound=theoretical,
                absolute_excess=abs_excess,
                relative_excess=max(ratio-1.0,0.0),
                max_over_bound=ratio,
                relative_tolerance_ratio=1.0+self.args.bound_ratio_tol,
                fp32_absolute_tolerance=POT_FP32_BOUND_ABS_TOL,
                allowed_absolute_excess=allowed_abs_excess,
                pass_=bool(ok),
            ))
            if ratio>1.0+self.args.bound_ratio_tol and ok:
                print(
                    f"[T06] PoT ideal-bound excess accepted as fp32 roundoff: "
                    f"{r['condition']} ratio={ratio:.6g} "
                    f"abs_excess={abs_excess:.3g} <= {allowed_abs_excess:.3g}"
                )
            if not ok:
                raise RuntimeError(
                    f"T06 PoT bound audit failed for {r['condition']}: "
                    f"ratio={ratio:.6g}, abs_excess={abs_excess:.6g}, "
                    f"allowed_abs_excess={allowed_abs_excess:.6g}"
                )
        write_csv(td/"pot_bound_audit.csv",pot_audit)

        # m=1 defines model-specific K0 and gets a full downstream pass.
        p1=c_pot(1.0,role="K0_anchor",cid="pot_m1")
        k0rep=self.measure_divergence(p1)
        self.evaluate_group([p1],task="T06")
        k0=float(k0rep["js_mean"])
        write_json(td/"K0.json",dict(model=self.hf_model,K0_jsd=k0,pot_m=1.0,
                                     divergence=k0rep,downstream=self.condition_summary(p1)))

        Ks=[12,14,16,20,24,32,48] if "deep_reference" in self.features else [20]
        tail_rows=self.rowmax_tail_probe(Ks)
        write_csv(td/"rowmax_tail_probe.csv",tail_rows)
        tail_by_k={int(r["K"]):r for r in tail_rows}
        rowspec=[c_rowmax_pot(K,cid=f"rowmax_pot_K{K}_clamp",role="rowmax_pot") for K in Ks]
        self.evaluate_group(rowspec,task="T06",collect_support_ids=[s.id for s in rowspec])
        rowrows=[]
        for sp in rowspec:
            sm=self.condition_summary(sp); K=int(sp.params["K"]); sm.update(tail_by_k[K]); sm["tail_mode"]="clamp"
            rowrows.append(sm)

        if "deep_reference" in self.features:
            drops=[c_rowmax_pot(K,tail_mode="drop",cid=f"rowmax_pot_K{K}_drop",role="rowmax_tail")
                   for K in [12,16,20,24]]
            self.evaluate_group(drops,task="T06",collect_support_ids=[s.id for s in drops])
            for sp in drops:
                sm=self.condition_summary(sp); K=int(sp.params["K"]); sm.update(tail_by_k[K]); sm["tail_mode"]="drop"
                rowrows.append(sm)
        write_csv(td/"rowmax_pot_conditions.csv",rowrows)
        print(f"[T06] K0={k0:.6e}; Rowmax-PoT K={Ks}")

    def read_K0(self) -> float:
        p=self.store.task_dir("T06")/"K0.json"
        if not p.exists(): self.run_T06()
        return float(read_json(p)["K0_jsd"])


    # =============================================================================
    # T07 — Unified distortion-response map
    # =============================================================================
    @staticmethod
    def _ols(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> dict[str,Any]:
        X=np.asarray(X,float); y=np.asarray(y,float)
        if weights is not None:
            w=np.sqrt(np.asarray(weights,float)).reshape(-1,1)
            Xw=X*w; yw=y*w[:,0]
        else:
            Xw=X; yw=y
        beta=np.linalg.lstsq(Xw,yw,rcond=None)[0]
        pred=X@beta
        sse=float(((y-pred)**2).sum()); sst=float(((y-y.mean())**2).sum())
        r2=1-sse/sst if sst>0 else float("nan")
        return dict(beta=beta.tolist(),r2=r2,rmse=float(np.sqrt(np.mean((y-pred)**2))),pred=pred)

    def run_T07(self):
        if "deep_reference" not in self.features:
            print("[T07] skipped: deep_reference only"); return
        self.load_runtime(); td=self.store.task_dir("T07")
        exp_specs=[
            c_multibit(21,1,"nearest","exp",cid="alloc_K21_R1"),
            c_multibit(21,2,"nearest","exp",cid="alloc_K21_R2"),
            c_multibit(21,4,"nearest","exp",cid="alloc_K21_R4"),
            c_multibit(21,8,"nearest","exp",cid="alloc_K21_R8"),
            c_multibit(16,1,"nearest","exp",cid="alloc_K16_R1"),
            c_multibit(16,4,"nearest","exp",cid="alloc_K16_R4"),
            c_multibit(32,1,"nearest","exp",cid="alloc_K32_R1"),
            c_multibit(32,4,"nearest","exp",cid="alloc_K32_R4"),
            c_multibit(21,4,"up","exp",cid="recon_K21_R4_up"),
            c_multibit(21,4,"lerp","exp",cid="recon_K21_R4_lerp"),
        ]
        pot_specs=[]
        for m in T07_POT_M:
            if abs(m-1.0)<1e-12:
                pot_specs.append(c_pot(1.0,cid="pot_m1",role="response_map"))
            else:
                pot_specs.append(c_pot(m,cid=f"response_pot_m{m:.10g}",role="response_map"))
        flat_specs=[c_temp(a,cid=f"response_temp_a{a:.10g}",role="response_map") for a in T07_TEMP_FLAT]
        sharp_specs=[c_temp(a,cid=f"response_temp_a{a:.10g}",role="response_map") for a in T07_TEMP_SHARP]
        specs=exp_specs+pot_specs+flat_specs+sharp_specs
        # Deduplicate by condition id while preserving fixed family membership.
        uniq=[]; seen=set()
        for s in specs:
            if s.id not in seen: uniq.append(s); seen.add(s.id)
        self.evaluate_group(uniq,task="T07")
        rows=[]
        for s in specs:
            sm=self.condition_summary(s)
            dv=self.measure_divergence(s)
            rows.append(dict(
                point_id=s.id,condition=s.id,family=s.family,operator=s.operator,
                alpha=s.params.get("alpha",np.nan),m=s.params.get("m",np.nan),
                K=s.params.get("K",np.nan),R=s.params.get("R",np.nan),
                reconstruction=s.params.get("reconstruction",""),
                js_mean=dv["js_mean"],tv_mean=dv["tv_mean"],
                kl_fwd_mean=dv["kl_fwd_mean"],kl_rev_mean=dv["kl_rev_mean"],
                delta_nll=sm["delta_nll"],ppl_ratio=sm["ppl_ratio"],
                output_jsd=sm.get("output_jsd",np.nan),
                top1_agreement=sm.get("top1_agreement",np.nan),
            ))
        df=pd.DataFrame(rows)
        df["usable_loglog"]=(df.js_mean>0)&(df.delta_nll>0)&np.isfinite(df.js_mean)&np.isfinite(df.delta_nll)
        df.to_csv(td/"distortion_points.csv",index=False)

        k0=self.read_K0(); fits=[]
        for regime,sub0 in [
            ("all",df[df.usable_loglog]),
            ("local_le_10K0",df[df.usable_loglog & (df.js_mean<=10*k0)]),
        ]:
            sub=sub0.copy()
            if len(sub)>=4:
                x=np.log(sub.js_mean.to_numpy(float)); y=np.log(sub.delta_nll.to_numpy(float))
                X=np.c_[np.ones(len(sub)),x]
                f=self._ols(X,y)
                fits.append(dict(regime=regime,model="global",n=len(sub),intercept=f["beta"][0],
                                 slope=f["beta"][1],r2=f["r2"],rmse_log=f["rmse"]))
                fams=sorted(sub.family.unique())
                # Family-specific intercepts + common slope. Drop one global intercept;
                # use one column per family plus a common log-JSD slope.
                F=np.column_stack([(sub.family.to_numpy()==fa).astype(float) for fa in fams])
                XF=np.c_[F,x]
                ff=self._ols(XF,y)
                fits.append(dict(regime=regime,model="family_intercept_common_slope",n=len(sub),
                                 slope=ff["beta"][-1],r2=ff["r2"],rmse_log=ff["rmse"],
                                 families=";".join(fams)))
                counts=sub.family.value_counts().to_dict()
                w=np.asarray([1.0/counts[z] for z in sub.family],float)
                fb=self._ols(X,y,w)
                fits.append(dict(regime=regime,model="family_balanced_global",n=len(sub),
                                 intercept=fb["beta"][0],slope=fb["beta"][1],
                                 r2=fb["r2"],rmse_log=fb["rmse"]))
            for fam,g in sub0.groupby("family"):
                if len(g)>=3:
                    x=np.log(g.js_mean.to_numpy(float)); y=np.log(g.delta_nll.to_numpy(float))
                    f=self._ols(np.c_[np.ones(len(g)),x],y)
                    fits.append(dict(regime=regime,model=f"family:{fam}",n=len(g),
                                     intercept=f["beta"][0],slope=f["beta"][1],
                                     r2=f["r2"],rmse_log=f["rmse"]))
        write_csv(td/"fit_summary.csv",fits)
        print(f"[T07] response map points={len(df)}, usable log-log={int(df.usable_loglog.sum())}")

    # =============================================================================
    # T08 — Matched-JSD directional sensitivity
    # =============================================================================
    def run_T08(self):
        self.load_runtime(); td=self.store.task_dir("T08")
        k0=self.read_K0(); mult=[1,3,10] if "deep_reference" in self.features else [1,10]
        trace=[]; man=dict(model=self.hf_model,K0=k0,axis="mean row-wise JSD",targets={})
        effects=[]; eval_specs=[]
        for mlt in mult:
            label="K0" if mlt==1 else f"{mlt:g}K0"
            target=k0*mlt
            am=self.match_temperature(target,"A-","jsd",self.args.seq,0.07*math.sqrt(mlt),label,trace)
            ap=self.match_temperature(target,"A+","jsd",self.args.seq,0.075*math.sqrt(mlt),label,trace)
            cp=self.match_pot_jsd(target,self.args.seq,math.sqrt(mlt),label,trace)
            for x,nm in [(am,"A-"),(ap,"A+"),(cp,"C")]:
                if x["status"]!="converged" or x["rel_err"]>self.args.tol:
                    write_csv(td/"match_trace.csv",trace)
                    raise RuntimeError(f"T08 {label} {nm} match failed rel={x['rel_err']:.3%}")
            # Rename matched temperature conditions to stable paper-facing ids.
            # The frozen alpha and achieved divergence from the matching pass are
            # the provenance; renaming does not trigger redundant remeasurement.
            sm=c_temp(am["alpha"],cid=f"jsd_{label}_Aminus",role="matched_jsd")
            sp=c_temp(ap["alpha"],cid=f"jsd_{label}_Aplus",role="matched_jsd")
            eval_specs += [sm,sp]
            man["targets"][label]=dict(
                multiplier=mlt,target_jsd=target,
                Aminus=dict(alpha=am["alpha"],achieved=am["metric"],rel_err=am["rel_err"],spec=sm.serializable()),
                Aplus=dict(alpha=ap["alpha"],achieved=ap["metric"],rel_err=ap["rel_err"],spec=sp.serializable()),
                C=dict(m=cp["m"],achieved=cp["metric"],rel_err=cp["rel_err"],spec=cp["spec"].serializable()),
            )
        write_csv(td/"match_trace.csv",trace); write_json(td/"match_manifest.json",man)
        self.evaluate_group(eval_specs,task="T08")
        for mlt in mult:
            label="K0" if mlt==1 else f"{mlt:g}K0"
            sm=c_temp(man["targets"][label]["Aminus"]["alpha"],cid=f"jsd_{label}_Aminus")
            sp=c_temp(man["targets"][label]["Aplus"]["alpha"],cid=f"jsd_{label}_Aplus")
            x=self.delta_series(sm)-self.delta_series(sp)
            e,l,h=Bootstrapper.mean_ci(x,self.args.boot,self.args.boot_seed)
            effects.append(dict(target_label=label,target_jsd=man["targets"][label]["target_jsd"],
                                effect="D_T_Aminus_minus_Aplus",estimate=e,ci_low=l,ci_high=h,
                                alpha_minus=sm.params["alpha"],alpha_plus=sp.params["alpha"]))
        write_csv(td/"directional_effects.csv",effects)
        write_csv(td/"conditions.csv",[self.condition_summary(s) for s in eval_specs])
        print("[T08] matched-JSD directional effects complete")

    def read_T08_manifest(self) -> dict[str,Any]:
        p=self.store.task_dir("T08")/"match_manifest.json"
        if not p.exists(): self.run_T08()
        return read_json(p)

    # =============================================================================
    # T09 — Matched-TV robustness
    # =============================================================================
    def run_T09(self):
        if "deep_reference" not in self.features and "tv_robustness" not in self.features:
            print("[T09] skipped: selected-model robustness only"); return
        self.load_runtime(); td=self.store.task_dir("T09"); jsman=self.read_T08_manifest()
        trace=[]; man=dict(model=self.hf_model,axis="mean row-wise TV",targets={}); specs=[]; effects=[]
        for label in ["K0","10K0"]:
            if label not in jsman["targets"]:
                raise RuntimeError(f"T09 requires T08 target {label}")
            m=float(jsman["targets"][label]["C"]["m"])
            pot=c_pot(m,cid=f"tv_anchor_{label}_pot_m{m:.10g}")
            rep=self.measure_divergence(pot)
            target=float(rep["tv_mean"])
            # Initialize from the already matched JSD temperature distances only;
            # TV itself decides the final match.
            xa=1.0-float(jsman["targets"][label]["Aminus"]["alpha"])
            xp=float(jsman["targets"][label]["Aplus"]["alpha"])-1.0
            am=self.match_temperature(target,"A-","tv",self.args.seq,max(xa,1e-4),f"TV_{label}",trace)
            ap=self.match_temperature(target,"A+","tv",self.args.seq,max(xp,1e-4),f"TV_{label}",trace)
            for x,nm in [(am,"A-"),(ap,"A+")]:
                if x["status"]!="converged" or x["rel_err"]>self.args.tol:
                    write_csv(td/"match_trace.csv",trace)
                    raise RuntimeError(f"T09 {label} {nm} match failed rel={x['rel_err']:.3%}")
            sm=c_temp(am["alpha"],cid=f"tv_{label}_Aminus",role="matched_tv")
            sp=c_temp(ap["alpha"],cid=f"tv_{label}_Aplus",role="matched_tv")
            specs += [sm,sp]
            man["targets"][label]=dict(target_tv=target,pot_m=m,
                Aminus=dict(alpha=am["alpha"],achieved=am["metric"],rel_err=am["rel_err"],
                            secondary_jsd=float(am["rep"]["js_mean"]),spec=sm.serializable()),
                Aplus=dict(alpha=ap["alpha"],achieved=ap["metric"],rel_err=ap["rel_err"],
                           secondary_jsd=float(ap["rep"]["js_mean"]),spec=sp.serializable()),
                secondary_anchor_jsd=rep["js_mean"])
        write_csv(td/"match_trace.csv",trace); write_json(td/"match_manifest.json",man)
        self.evaluate_group(specs,task="T09")
        for label in ["K0","10K0"]:
            sm=c_temp(man["targets"][label]["Aminus"]["alpha"],cid=f"tv_{label}_Aminus")
            sp=c_temp(man["targets"][label]["Aplus"]["alpha"],cid=f"tv_{label}_Aplus")
            x=self.delta_series(sm)-self.delta_series(sp)
            e,l,h=Bootstrapper.mean_ci(x,self.args.boot,self.args.boot_seed)
            effects.append(dict(target_label=label,effect="D_TV_Aminus_minus_Aplus",
                                estimate=e,ci_low=l,ci_high=h,target_tv=man["targets"][label]["target_tv"],
                                jsd_Aminus=man["targets"][label]["Aminus"]["secondary_jsd"],
                                jsd_Aplus=man["targets"][label]["Aplus"]["secondary_jsd"],
                                jsd_anchor=man["targets"][label]["secondary_anchor_jsd"]))
        write_csv(td/"directional_effects.csv",effects)
        write_csv(td/"conditions.csv",[self.condition_summary(s) for s in specs])
        print("[T09] matched-TV robustness complete")

    # =============================================================================
    # T10 — High-mass redistribution and local mass-partition rescue
    # =============================================================================
    @torch.no_grad()
    def p4_probe_condition(self,spec:ConditionSpec,qs:list[float],*,rescue=False) -> tuple[list[dict],list[dict],list[dict],dict]:
        configure_condition(self.bta,spec,self.n_layers,self.q_chunk)
        CTRL.reset(); CTRL.condition=spec
        CTRL.p4=P4Accumulator(qs,0.9,self.args.p4_min_n)
        CTRL.p4_qs=list(qs); CTRL.p4_primary_q=0.9; CTRL.p4_min_n=self.args.p4_min_n
        CTRL.p4_probe=True; CTRL.head_chunk=self.args.p4_head_chunk
        CTRL.rescue=bool(rescue or spec.rescue_q is not None)
        CTRL.rescue_q=float(spec.rescue_q if spec.rescue_q is not None else 0.9) if CTRL.rescue else None
        nw=1 if self.args.fast else int(self.args.p4_probe_windows)
        for w in range(nw):
            ch=self.get_block(w,self.args.seq)
            _=get_backbone(self.model)(input_ids=ch,use_cache=False,return_dict=True)
            del ch
        acc=CTRL.p4
        mass=acc.mass_rows(spec.id); out=acc.output_rows(spec.id); rescue_rows=acc.rescue_rows(spec.id)

        # Aggregate from raw accumulators, preserving global RMS definition.
        agg=dict(condition=spec.id)
        for q in qs:
            cells=[a for (li,qq),a in acc.mass.items() if abs(qq-q)<1e-12]
            n=sum(a["n"] for a in cells)
            if n>0:
                agg[f"q{q:g}_n_rows"]=int(n)
                agg[f"q{q:g}_ref_mass"]=sum(a["ref"] for a in cells)/n
                agg[f"q{q:g}_alt_mass"]=sum(a["alt"] for a in cells)/n
                agg[f"q{q:g}_leakage"]=sum(a["leak"] for a in cells)/n
                agg[f"q{q:g}_set_fraction"]=sum(a["frac"] for a in cells)/n
        oc=list(acc.out.values()); ref=sum(a["ref2"] for a in oc)
        if ref>0:
            agg["output_n_rows"]=int(sum(a["n"] for a in oc))
            agg["output_rms_pre"]=math.sqrt(sum(a["pre_diff2"] for a in oc)/ref)
            agg["output_rms_used"]=math.sqrt(sum(a["used_diff2"] for a in oc)/ref)
            n=sum(a["n"] for a in oc)
            agg["output_cosine_pre"]=sum(a["cos_pre"] for a in oc)/max(n,1)
            agg["output_cosine_used"]=sum(a["cos_used"] for a in oc)/max(n,1)
        rc=list(acc.rescue.values())
        if rc:
            n=sum(a["n"] for a in rc)
            agg["rescue_mass_err_mean"]=sum(a["mass_err"] for a in rc)/max(n,1)
            agg["rescue_mass_err_max"]=max(a.get("mass_err_max",0.0) for a in rc)
            agg["rescue_js_pre"]=sum(a["js_pre"] for a in rc)/max(n,1)
            agg["rescue_js_after"]=sum(a["js_rescue"] for a in rc)/max(n,1)
            agg["rescue_tv_pre"]=sum(a["tv_pre"] for a in rc)/max(n,1)
            agg["rescue_tv_after"]=sum(a["tv_rescue"] for a in rc)/max(n,1)
        CTRL.reset()
        return mass,out,rescue_rows,agg

    def run_T10(self):
        self.load_runtime(); td=self.store.task_dir("T10"); man=self.read_T08_manifest()
        for lbl in ["K0","10K0"]:
            if lbl not in man["targets"]: raise RuntimeError(f"T10 requires T08 {lbl}")
        qs=[0.5,0.8,0.9,0.95] if "deep_reference" in self.features else [0.9]
        probes=[]
        for lbl in ["K0","10K0"]:
            probes += [
                c_temp(float(man["targets"][lbl]["Aminus"]["alpha"]),cid=f"jsd_{lbl}_Aminus"),
                c_temp(float(man["targets"][lbl]["Aplus"]["alpha"]),cid=f"jsd_{lbl}_Aplus"),
            ]
        probes += [
            c_multibit(21,1,"nearest","exp",cid="alloc_K21_R1"),
            c_multibit(21,4,"nearest","exp",cid="alloc_K21_R4"),
            c_pot(1.0,cid="pot_m1"),
        ]
        massrows=[]; outrows=[]; rescdiag=[]; aggregates=[]
        for s in probes:
            m,o,r,a=self.p4_probe_condition(s,qs,rescue=False)
            massrows+=m; outrows+=o; rescdiag+=r; aggregates.append(a)
            print(f"  [P4 probe] {s.id}")
        write_csv(td/"mass_probe.csv",massrows); write_csv(td/"output_probe.csv",outrows)
        write_csv(td/"probe_summary.csv",aggregates)

        rescue_specs=[]
        original={}
        for lbl in ["K0","10K0"]:
            am=c_temp(float(man["targets"][lbl]["Aminus"]["alpha"]),cid=f"jsd_{lbl}_Aminus")
            ap=c_temp(float(man["targets"][lbl]["Aplus"]["alpha"]),cid=f"jsd_{lbl}_Aplus")
            rm=c_temp(am.params["alpha"],cid=f"rescue_{lbl}_Aminus_q0p9",role="p4_rescue",rescue_q=0.9)
            rp=c_temp(ap.params["alpha"],cid=f"rescue_{lbl}_Aplus_q0p9",role="p4_rescue",rescue_q=0.9)
            original[lbl]=(am,ap); rescue_specs += [rm,rp]
        self.evaluate_group(rescue_specs,task="T10")

        effects=[]
        for lbl in ["K0","10K0"]:
            am,ap=original[lbl]
            rm=next(s for s in rescue_specs if s.id==f"rescue_{lbl}_Aminus_q0p9")
            rp=next(s for s in rescue_specs if s.id==f"rescue_{lbl}_Aplus_q0p9")
            dorig=self.delta_series(am)-self.delta_series(ap)
            dres=self.delta_series(rm)-self.delta_series(rp)
            gap=dorig-dres
            eo,lo,ho=Bootstrapper.mean_ci(dorig,self.args.boot,self.args.boot_seed)
            er,lr,hr=Bootstrapper.mean_ci(dres,self.args.boot,self.args.boot_seed)
            eg,lg,hg=Bootstrapper.mean_ci(gap,self.args.boot,self.args.boot_seed)
            gm=self.delta_series(am)-self.delta_series(rm)
            gp=self.delta_series(ap)-self.delta_series(rp)
            em,lm,hm=Bootstrapper.mean_ci(gm,self.args.boot,self.args.boot_seed)
            ep,lp,hp=Bootstrapper.mean_ci(gp,self.args.boot,self.args.boot_seed)
            effects.append(dict(target_label=lbl,original_gap=eo,original_ci_low=lo,original_ci_high=ho,
                                rescued_gap=er,rescued_ci_low=lr,rescued_ci_high=hr,
                                gap_reduction=eg,gap_reduction_ci_low=lg,gap_reduction_ci_high=hg,
                                rescue_gain_Aminus=em,rescue_gain_Aminus_ci_low=lm,rescue_gain_Aminus_ci_high=hm,
                                rescue_gain_Aplus=ep,rescue_gain_Aplus_ci_low=lp,rescue_gain_Aplus_ci_high=hp,
                                descriptive_fraction_removed=(eg/eo if abs(eo)>1e-15 else np.nan)))
        write_csv(td/"rescue_effects.csv",effects)
        write_csv(td/"rescue_conditions.csv",[self.condition_summary(s) for s in rescue_specs])

        if "deep_reference" in self.features:
            for s in rescue_specs:
                m,o,r,a=self.p4_probe_condition(s,[0.9],rescue=True)
                rescdiag+=r; aggregates.append(a)
            write_csv(td/"rescue_probe.csv",rescdiag)
            write_csv(td/"probe_summary_with_rescue.csv",aggregates)
        print("[T10] P4 leakage + rescue complete")


    # =============================================================================
    # T11 — Long-context transfer
    # =============================================================================
    def run_T11(self):
        if "deep_reference" in self.features:
            contexts=parse_csv_int(self.args.contexts) or [2048,8192,16384]
        elif "longctx_8k" in self.features:
            contexts=[2048,8192]
        else:
            print("[T11] skipped: selected-model long-context validation only"); return
        self.load_runtime(); td=self.store.task_dir("T11")
        maxpos=getattr(self.model.config,"max_position_embeddings",None)
        if maxpos is not None:
            bad=[L for L in contexts if L>int(maxpos)]
            if bad:
                raise RuntimeError(f"requested contexts {bad} exceed model max_position_embeddings={maxpos}")
        trace=[]; manifest=dict(model=self.hf_model,contexts={}); effects=[]; condrows=[]
        for L in contexts:
            nw=1 if self.args.fast else max(1,int(self.args.div_tokens)//int(L))
            pot=c_pot(1.0,cid="longctx_pot_m1",role="longctx")
            krep=self.measure_divergence(pot,L,nw)
            k0=float(krep["js_mean"])
            am=self.match_temperature(k0,"A-","jsd",L,0.07,"longctx_K0",trace,windows=nw)
            ap=self.match_temperature(k0,"A+","jsd",L,0.075,"longctx_K0",trace,windows=nw)
            if am["status"]!="converged" or am["rel_err"]>self.args.tol or \
               ap["status"]!="converged" or ap["rel_err"]>self.args.tol:
                write_csv(td/"match_trace.csv",trace)
                raise RuntimeError(f"T11 matching failed at L={L}")
            r1=c_multibit(21,1,"nearest","exp",cid="longctx_alloc_R1",role="longctx")
            r4=c_multibit(21,4,"nearest","exp",cid="longctx_alloc_R4",role="longctx")
            tm=c_temp(am["alpha"],cid="longctx_Aminus",role="longctx")
            tp=c_temp(ap["alpha"],cid="longctx_Aplus",role="longctx")
            # Baseline is first-class per context; evaluate_group reuses hb so
            # this adds no extra backbone pass.
            specs=[c_baseline(),r1,r4,pot,tm,tp]
            self.evaluate_group(specs,L,task="T11")
            condrows += [self.condition_summary(s,L) for s in specs]
            dr=self.delta_series(r1,L)-self.delta_series(r4,L)
            dt=self.delta_series(tm,L)-self.delta_series(tp,L)
            er,lr,hr=Bootstrapper.mean_ci(dr,self.args.boot,self.args.boot_seed)
            et,lt,ht=Bootstrapper.mean_ci(dt,self.args.boot,self.args.boot_seed)
            effects += [
                dict(seq_len=L,effect="D_R_R1_minus_R4",estimate=er,ci_low=lr,ci_high=hr),
                dict(seq_len=L,effect="D_T_Aminus_minus_Aplus",estimate=et,ci_low=lt,ci_high=ht),
            ]
            manifest["contexts"][str(L)]=dict(
                divergence_windows=nw,K0_jsd=k0,
                Aminus=dict(alpha=am["alpha"],achieved=am["metric"],rel_err=am["rel_err"]),
                Aplus=dict(alpha=ap["alpha"],achieved=ap["metric"],rel_err=ap["rel_err"]),
            )
            print(f"[T11] L={L} K0={k0:.6e} D_R={er:+.5g} D_T={et:+.5g}")
        write_csv(td/"match_trace.csv",trace); write_json(td/"match_manifest.json",manifest)
        write_csv(td/"conditions.csv",condrows); write_csv(td/"within_context_effects.csv",effects)

    # =============================================================================
    # T12 — Zero-shot downstream breadth
    # =============================================================================
    def run_T12(self):
        self.load_runtime(); td=self.store.task_dir("T12")
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except Exception as ex:
            if self.args.allow_missing_lm_eval:
                write_json(td/"status.json",dict(
                    status="skipped",reason=f"lm_eval unavailable: {ex}",
                    execution_signature=self.execution_signature,
                    task_execution_signature=self.task_execution_signature("T12"),
                    task_execution_spec=self.task_execution_spec("T12"),
                ))
                print(f"[T12] skipped: lm_eval unavailable ({ex})")
                return
            raise RuntimeError("T12 requires lm-evaluation-harness (lm_eval)") from ex

        tasks=parse_csv_str(self.args.zero_shot_tasks)
        eng=read_json(self.store.model_dir/"engineering_checks.json",{}) or {}
        pads=[r for r in eng.get("checks",[])
              if r.get("check")=="left_padding_bta_vs_eager_error_ratio"]
        if self.args.zero_shot_bs>1:
            if not pads:
                raise RuntimeError(
                    "T12 refuses batch_size>1 because the formal left-padding "
                    "BTA-vs-eager gate is missing. Run T00 successfully first."
                )
            if not bool(pads[-1].get("pass_",False)):
                raise RuntimeError(
                    "T12 refuses batch_size>1 because the left-padding mask gate failed. "
                    "Use --zero-shot-bs 1 for a diagnostic run or fix the attention mask path."
                )
        specs=[
            c_baseline(),
            c_binary(True),
            c_multibit(32,4,"nearest","exp",cid="zero_exp_K32_R4"),
            c_multibit(64,4,"nearest","exp",cid="zero_exp_K64_R4"),
            c_rowmax_pot(20,cid="zero_rowmax_pot_K20"),
        ]
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
        self.tok.padding_side="left"
        lm=HFLM(pretrained=self.model,tokenizer=self.tok,batch_size=self.args.zero_shot_bs)
        rows=[]
        for spec in specs:
            configure_condition(self.bta,spec,self.n_layers,self.q_chunk); CTRL.reset(); CTRL.condition=spec
            print(f"[T12] {spec.id}: {tasks}")
            res=simple_evaluate(model=lm,tasks=tasks,num_fewshot=0,
                                limit=(self.args.zero_shot_limit if self.args.zero_shot_limit>0 else None))
            for task,d in res.get("results",{}).items():
                for key,val in d.items():
                    if key=="alias" or key.endswith("_stderr,none") or not isinstance(val,(int,float)):
                        continue
                    metric=key.replace(",none","")
                    if metric=="sample_len": continue
                    se=d.get(f"{metric}_stderr,none",np.nan)
                    try: se=float(se)
                    except Exception: se=np.nan
                    rows.append(dict(condition=spec.id,task=task,metric=metric,value=float(val),stderr=se))
            write_csv(td/"zero_shot_results.csv",rows)
        CTRL.reset()
        print(f"[T12] zero-shot rows={len(rows)}")

    # =============================================================================
    # T13 — Model-output fidelity aggregation
    # =============================================================================
    def run_T13(self):
        self.load_runtime(); td=self.store.task_dir("T13")
        df=self.store.all_condition_summaries()
        if df.empty:
            write_json(td/"status.json",dict(status="complete",note="no full downstream conditions yet"))
            print("[T13] no condition outputs to aggregate")
            return
        cols=[c for c in ["condition","seq_len","n_blocks","delta_nll","ppl_ratio",
                          "output_kl_forward","output_kl_reverse","output_jsd","top1_agreement"] if c in df.columns]
        df[cols].to_csv(td/"output_fidelity_conditions.csv",index=False)
        diag=[]
        d0=df[df.seq_len==self.args.seq].copy()
        for m in ["output_kl_forward","output_kl_reverse","output_jsd","top1_agreement"]:
            if m not in d0: continue
            z=d0[[m,"delta_nll"]].replace([np.inf,-np.inf],np.nan).dropna()
            if len(z)>=4:
                pear=float(z[m].corr(z.delta_nll,method="pearson"))
                # pandas rank correlation gives Spearman without scipy.
                spear=float(z[m].rank().corr(z.delta_nll.rank(),method="pearson"))
                diag.append(dict(metric=m,n=len(z),pearson_with_delta_nll=pear,
                                 spearman_with_delta_nll=spear))
        write_csv(td/"diagnostic_correlations.csv",diag)
        print(f"[T13] aggregated output fidelity for {len(df)} condition/context rows")

    # =============================================================================
    # Final summaries / root-level cross-model tables
    # =============================================================================
    def finalize(self):
        self.load_runtime()
        cs=self.store.all_condition_summaries()
        cs.to_csv(self.store.model_dir/"condition_summary.csv",index=False)
        # Concatenate all block files into the model-level blocks.csv. This is
        # intentionally redundant with condition-specific files for convenient
        # paper analysis.
        allb=[]
        for p in sorted(self.store.cond_dir.glob("*.csv")):
            try:
                d=pd.read_csv(p)
                if not d.empty: allb.append(d)
            except Exception: pass
        if allb:
            pd.concat(allb,ignore_index=True).to_csv(self.store.model_dir/"blocks.csv",index=False)

        info=read_json(self.store.model_dir/"model_info.json",{}) or {}
        summary=dict(
            model=self.hf_model,alias=self.alias,family=info.get("family"),
            parameter_count=info.get("exact_parameters"),num_layers=info.get("num_layers"),
            attention_type=info.get("attention_type"),num_attention_heads=info.get("num_attention_heads"),
            num_kv_heads=info.get("num_key_value_heads"),dtype=info.get("dtype"),
            chosen_q_chunk=self.q_chunk,token_signature=token_signature(self.ids),
            model_revision=info.get("model_revision"),
            tokenizer_revision=info.get("tokenizer_revision"),
            transformers_version=info.get("transformers_version"),
            torch_version=info.get("torch_version"),
            attn_logit_softcapping=info.get("attn_logit_softcapping"),
            final_logit_softcapping=info.get("final_logit_softcapping"),
        )
        # Baseline from any condition row.
        if not cs.empty:
            z=cs[cs.seq_len==self.args.seq]
            if not z.empty:
                bn=float(z.baseline_nll.iloc[0]); summary["baseline_nll"]=bn; summary["baseline_ppl"]=exp_safe(bn)

        def cond(cid,key,outkey=None):
            if cs.empty:return
            z=cs[(cs.condition==cid)&(cs.seq_len==self.args.seq)]
            if not z.empty and key in z:
                summary[outkey or f"{cid}_{key}"]=float(z.iloc[0][key])
        bsp=self.store.task_dir("T01")/"baseline_mean_support.json"
        if bsp.exists():
            summary["support_density"]=float(read_json(bsp)["support"]["density"])
        else:
            cond("mean_sparse_softmax","support_density","support_density")
        cond("mean_sparse_softmax","delta_nll","sparse_dnll")
        cond("mean_sparse_uniform","delta_nll","uniform_dnll")
        p=self.store.task_dir("T01B")/"aggregate_effect.csv"
        if p.exists():
            z=pd.read_csv(p)
            if not z.empty:
                r=z.iloc[0]
                summary["strict_same_support_D_W"]=float(r.estimate)
                summary["strict_same_support_D_W_ci_low"]=float(r.ci_low)
                summary["strict_same_support_D_W_ci_high"]=float(r.ci_high)
        cond("geometry_exp_K16","delta_nll","expK16_dnll")
        cond("geometry_linear_K16","delta_nll","linearK16_dnll")
        cond("precision_K32_R4","delta_nll","expK32_dnll")
        cond("alloc_K21_R1","delta_nll","R1K21_dnll")
        cond("alloc_K21_R4","delta_nll","R4K21_dnll")
        cond("pot_m1","delta_nll","pot_m1_dnll")
        cond("rowmax_pot_K20_clamp","delta_nll","rowmax_pot_K20_dnll")
        cond("rowmax_pot_K20_clamp","ppl_ratio","rowmax_pot_K20_ppl_ratio")
        k0p=self.store.task_dir("T06")/"K0.json"
        if k0p.exists(): summary["K0_jsd"]=float(read_json(k0p)["K0_jsd"])

        # Pull primary effects.
        p=self.store.task_dir("T04")/"effects.csv"
        if p.exists():
            d=pd.read_csv(p); z=d[d.effect=="D_R_K21_R1_minus_R4"]
            if not z.empty:
                r=z.iloc[0]; summary.update(R1_minus_R4=float(r.estimate),R_effect_ci_low=float(r.ci_low),R_effect_ci_high=float(r.ci_high))
        p=self.store.task_dir("T08")/"directional_effects.csv"
        if p.exists():
            d=pd.read_csv(p)
            for lbl in ["K0","10K0"]:
                z=d[d.target_label==lbl]
                if not z.empty:
                    r=z.iloc[0]; summary[f"dir_{lbl}"]=float(r.estimate)
                    summary[f"dir_{lbl}_ci_low"]=float(r.ci_low); summary[f"dir_{lbl}_ci_high"]=float(r.ci_high)
                    summary[f"alpha_minus_{lbl}"]=float(r.alpha_minus); summary[f"alpha_plus_{lbl}"]=float(r.alpha_plus)
        p=self.store.task_dir("T10")/"rescue_effects.csv"
        if p.exists():
            d=pd.read_csv(p)
            for lbl in ["K0","10K0"]:
                z=d[d.target_label==lbl]
                if not z.empty:
                    r=z.iloc[0]; summary[f"rescue_gap_{lbl}"]=float(r.rescued_gap)
                    summary[f"rescue_gap_reduction_{lbl}"]=float(r.gap_reduction)
                    summary[f"rescue_gap_reduction_{lbl}_ci_low"]=float(r.gap_reduction_ci_low)
                    summary[f"rescue_gap_reduction_{lbl}_ci_high"]=float(r.gap_reduction_ci_high)
        p=self.store.task_dir("T10")/"probe_summary.csv"
        if p.exists():
            d=pd.read_csv(p)
            for lbl,cid in [("minus_K0","jsd_K0_Aminus"),("plus_K0","jsd_K0_Aplus"),
                            ("minus_10K0","jsd_10K0_Aminus"),("plus_10K0","jsd_10K0_Aplus")]:
                z=d[d.condition==cid]
                if not z.empty:
                    r=z.iloc[0]
                    if "q0.9_leakage" in z.columns: summary[f"leak_{lbl}_q90"]=float(r["q0.9_leakage"])
                    if "output_rms_pre" in z.columns: summary[f"output_rms_{lbl}"]=float(r.output_rms_pre)
                    if "output_cosine_pre" in z.columns: summary[f"output_cos_{lbl}"]=float(r.output_cosine_pre)

        fidelity={}
        if not cs.empty:
            z=cs[cs.seq_len==self.args.seq]
            for _,r in z.iterrows():
                vals={}
                for m in ("output_kl_forward","output_kl_reverse","output_jsd","top1_agreement"):
                    if m in z.columns and pd.notna(r.get(m,np.nan)):
                        vals[m]=float(r[m])
                if vals:
                    fidelity[str(r.condition)]=vals
        summary["output_fidelity_by_condition"]=fidelity
        write_json(self.store.model_dir/"model_summary.json",summary)

        # Unified per-layer export requested by the open-source task spec.
        layer_frames=[]
        for mp in sorted(self.store.cond_meta_dir.glob("*.json")):
            meta=read_json(mp,{}) or {}
            sup=meta.get("support") or {}
            pl=sup.get("per_layer") or {}
            if pl:
                cid=(meta.get("condition") or {}).get("id",mp.stem)
                layer_frames.append(pd.DataFrame([
                    dict(source="support",condition=cid,layer=int(li),
                         support_density=float(val))
                    for li,val in pl.items()
                ]))
        for fn,src in [
            (self.store.task_dir("T10")/"mass_probe.csv","p4_mass"),
            (self.store.task_dir("T10")/"output_probe.csv","p4_output"),
            (self.store.task_dir("T10")/"rescue_probe.csv","p4_rescue"),
        ]:
            if fn.exists():
                try:
                    d=pd.read_csv(fn)
                    if not d.empty and "layer" in d.columns:
                        d.insert(0,"source",src)
                        layer_frames.append(d)
                except Exception:
                    pass
        if layer_frames:
            pd.concat(layer_frames,ignore_index=True,sort=False).to_csv(
                self.store.model_dir/"layer_metrics.csv",index=False
            )

        # Root-level tables: replace this model's row, preserving other models.
        self.store.root.mkdir(parents=True,exist_ok=True)
        root_sum=self.store.root/"crossmodel_summary.csv"
        if root_sum.exists():
            d=pd.read_csv(root_sum); key=self.alias or self.hf_model
            if "alias" in d: d=d[d.alias.fillna("")!=str(self.alias or "")]
            elif "model" in d: d=d[d.model!=self.hf_model]
            d=pd.concat([d,pd.DataFrame([summary])],ignore_index=True,sort=False)
        else:
            d=pd.DataFrame([summary])
        d.to_csv(root_sum,index=False)

        # Root effects table from T04/T08/T10/T09 if present.
        eff=[]
        for task,fn in [("T01B","aggregate_effect.csv"),("T04","effects.csv"),("T08","directional_effects.csv"),
                        ("T09","directional_effects.csv"),("T10","rescue_effects.csv")]:
            p=self.store.task_dir(task)/fn
            if p.exists():
                z=pd.read_csv(p); z.insert(0,"task",task); z.insert(0,"model",self.hf_model); eff.append(z)
        root_eff=self.store.root/"crossmodel_effects.csv"
        if eff:
            cur=pd.concat(eff,ignore_index=True,sort=False)
            if root_eff.exists():
                old=pd.read_csv(root_eff)
                if "model" in old: old=old[old.model!=self.hf_model]
                cur=pd.concat([old,cur],ignore_index=True,sort=False)
            cur.to_csv(root_eff,index=False)

        write_json(self.store.root/"crossmodel_manifest.json",dict(
            generated_by="softmax_approx/characterization.py",
            models=[str(x) for x in sorted(p.name for p in self.store.root.iterdir() if p.is_dir())],
            primary_endpoints=["D_W_strict_same_support","D_R","D_T_K0","D_T_10K0","G_gap_10K0"],
        ))
        write_json(self.store.model_dir/"run_status.json",dict(status="complete",tasks=self.tasks,time=time.time(),
                                                               execution_signature=self.execution_signature))
        print(f"[finalize] {self.store.model_dir}")

    def run(self):
        if self.args.dry_run:
            print(json.dumps(self.plan,indent=2,ensure_ascii=False))
            return
        write_json(self.store.model_dir/"run_status.json",dict(status="running",tasks=self.tasks,time=time.time(),
                                                               execution_signature=self.execution_signature))
        try:
            for t in self.tasks:
                td=self.store.task_dir(t); st=read_json(td/"status.json",{}) or {}
                task_sig=self.task_execution_signature(t)
                if self.args.resume and st.get("status")=="complete" and not self.args.rerun_complete:
                    stored_task_sig=st.get("task_execution_signature")
                    stored_exec_sig=st.get("execution_signature")
                    if stored_task_sig==task_sig:
                        print(f"[{t}] already complete -> skip (--resume)")
                        continue
                    # Backward-compatible migration for v2.2 statuses.  T12 is
                    # intentionally excluded because its limit/task/bs settings
                    # now form part of its own identity.
                    if t!="T12" and stored_task_sig is None and stored_exec_sig in (
                        {None} | set(self.compatible_execution_signatures)
                    ):
                        print(f"[{t}] already complete -> skip (--resume; migrated legacy status)")
                        write_json(td/"status.json",dict(
                            status="complete",task=t,time=st.get("time",time.time()),
                            execution_signature=self.execution_signature,
                            task_execution_signature=task_sig,
                            task_execution_spec=self.task_execution_spec(t),
                        ))
                        continue
                    raise RuntimeError(
                        f"{t} status belongs to different task settings. "
                        f"For T12, changing --zero-shot-limit/--zero-shot-bs/--zero-shot-tasks "
                        f"requires --rerun-complete (same fast/formal regime) or a separate --out-root."
                    )
                print("\n"+"="*110); print(f"{t}  model={self.hf_model} profile={self.profile}"); print("="*110)
                fn=getattr(self,f"run_{t}")
                fn()
                # A task may have intentionally written skipped status.
                st2=read_json(td/"status.json",{}) or {}
                if st2.get("status") not in {"skipped"}:
                    write_json(td/"status.json",dict(
                        status="complete",task=t,time=time.time(),
                        execution_signature=self.execution_signature,
                        task_execution_signature=task_sig,
                        task_execution_spec=self.task_execution_spec(t),
                    ))
            self.finalize()
        except Exception as ex:
            write_json(self.store.model_dir/"run_status.json",dict(status="failed",tasks=self.tasks,
                       error=f"{type(ex).__name__}: {ex}",time=time.time(),
                       execution_signature=self.execution_signature))
            raise
        finally:
            CTRL.reset()


# =============================================================================
# 10. CLI
# =============================================================================


def build_argparser():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model",default="qwen15",
                    help="registry alias (qwen15,llama1,...) or a Hugging Face repo id")
    ap.add_argument("--profile",choices=sorted(PROFILE_TASKS),default=None)
    ap.add_argument("--tasks",default=None,help="comma-separated task ids (T00,T01,T01B,...,T13); dependencies added automatically")
    ap.add_argument("--out-root",default="results")
    ap.add_argument("--dataset",default="wikitext")
    ap.add_argument("--seq",type=int,default=2048)
    ap.add_argument("--tokens",type=int,default=200_000)
    ap.add_argument("--q-chunk",default="auto",
                    help="positive integer or auto (512->256->128; 2K P4 + max-context stress gate)")
    ap.add_argument("--ce-chunk",type=int,default=64)
    ap.add_argument("--verify-windows",type=int,default=8)
    ap.add_argument("--div-tokens",type=int,default=32_768)
    ap.add_argument("--div-min-n",type=int,default=1)
    ap.add_argument("--error-probe-windows",type=int,default=0,
                    help="0 = all blocks for deep_reference, verify-window budget otherwise")
    ap.add_argument("--error-min-n",type=int,default=64)
    ap.add_argument("--error-min-span",type=float,default=1e-3)
    ap.add_argument("--p4-probe-windows",type=int,default=4)
    ap.add_argument("--p4-min-n",type=int,default=64)
    ap.add_argument("--p4-head-chunk",type=int,default=4)
    ap.add_argument("--contexts",default="2048,8192,16384")
    ap.add_argument("--tol",type=float,default=0.02,help="required relative divergence matching tolerance")
    ap.add_argument("--tol-pref",type=float,default=0.01,dest="tol_pref")
    ap.add_argument("--max-root-iter",type=int,default=18)
    ap.add_argument("--boot",type=int,default=5000)
    ap.add_argument("--boot-seed",type=int,default=0)
    ap.add_argument("--gate-len",type=int,default=1024)
    ap.add_argument("--safe-nll-tol",type=float,default=2e-5)
    ap.add_argument("--chunk-tol",type=float,default=2e-5)
    ap.add_argument("--chunk-hidden-rel-tol",type=float,default=2e-5,
                    help="q_chunk equivalence tolerance on max relative hidden-state difference")
    ap.add_argument("--probe-identity-rel-tol",type=float,default=2e-6,
                    help="T00 probe_sparsity exact-output hidden-state relative tolerance")
    ap.add_argument("--checkpoint-every",type=int,default=1,
                    help="incrementally flush condition block CSVs every N processed text blocks")
    ap.add_argument("--impl-rel-tol",type=float,default=5e-3,
                    help="BTA bf16-score vs official eager relative PPL tolerance")
    ap.add_argument("--padding-ratio-tol",type=float,default=5.0,
                    help="maximum BTA/eager batch-vs-solo error ratio for left/right padding")
    ap.add_argument("--identity-tol",type=float,default=2e-6)
    ap.add_argument("--identity-div-tol",type=float,default=2e-7)
    ap.add_argument("--adapter-tol",type=float,default=3e-5,
                    help="T00 max probability mismatch for analytical/BTA adapter audit")
    ap.add_argument("--rescue-gate-tol",type=float,default=3e-6,
                    help="T00 numerical tolerance for synthetic P4 rescue audit")
    ap.add_argument("--bound-ratio-tol",type=float,default=2e-4,
                    help="allowed relative floating-point excess above ideal analytical error bounds")
    ap.add_argument("--padding-rel-tol",type=float,default=0.15)
    ap.add_argument("--allow-padding-gate-failure",action="store_true")
    ap.add_argument("--zero-shot-tasks",default="lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande")
    ap.add_argument("--zero-shot-bs",type=int,default=8)
    ap.add_argument("--zero-shot-limit",type=int,default=0)
    ap.add_argument("--allow-missing-lm-eval",action="store_true")
    ap.add_argument("--trust-remote-code",action="store_true")
    ap.add_argument("--no-output-fidelity",action="store_true",
                    help="debug-only: skip vocab KL/JSD/Top1; formal runs should not use this")
    ap.add_argument("--max-blocks",type=int,default=0,help="debug only; 0=all complete blocks")
    ap.add_argument("--fast",action="store_true",help="smoke test: <=4 downstream blocks and minimal probes")
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--resume",action="store_true")
    mode.add_argument("--overwrite",action="store_true")
    ap.add_argument("--rerun-complete",action="store_true",
                    help="with --resume, rerun task-level analysis even if status is complete")
    ap.add_argument("--dry-run",action="store_true")
    return ap


def validate_args(args):
    if args.seq<128: raise ValueError("--seq must be >=128")
    if args.tokens<args.seq: raise ValueError("--tokens must be >= --seq")
    if args.ce_chunk<=0: raise ValueError("--ce-chunk must be positive")
    if args.checkpoint_every<=0: raise ValueError("--checkpoint-every must be positive")
    if args.boot<100: raise ValueError("--boot must be >=100")
    if not (0<args.tol_pref<=args.tol): raise ValueError("require 0 < tol-pref <= tol")
    if args.fast:
        args.boot=min(args.boot,500)
        args.verify_windows=1
        args.error_probe_windows=1
        args.p4_probe_windows=1
    # dry-run should be non-destructive and may reuse an existing plan directory.
    if args.dry_run and not args.overwrite:
        args.resume=True


def main():
    ap=build_argparser(); args=ap.parse_args(); validate_args(args)
    runner=CrossModelRunner(args)
    runner.run()


if __name__=="__main__":
    main()
