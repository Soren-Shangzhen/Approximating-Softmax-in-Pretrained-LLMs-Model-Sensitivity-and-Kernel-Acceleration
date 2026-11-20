#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rowmax-H15 PyTorch reference that reproduces the patched kernel's bit operations step by step (no exp2(n/2)
shortcut), for element-wise comparison with the kernel's probabilities (correctness gates, Appendix B.2).

FA4 applies the exponential after scale_subtract_rowmax, i.e. to values already relative to the running row
maximum: x = 2 (sigma - sigma_max) + 2 max_offset in doubled log2 units (the lattice has two points per octave).
The operator is therefore anchored at the row maximum: the maximal entry gets an integer x = 2 max_offset and the
exact weight 2^max_offset.  This reference anchors at the whole-row maximum, which equals the kernel's running
maximum only within a single key tile (seqlen_k <= 128); multi-tile rows are covered by the envelope gate.
"""

from __future__ import annotations

import torch

MAGIC = float(2 ** 23 + 2 ** 22)          # 0x4B400000
CLAMP_LO = -252.0                         # -> floor(n/2) = -126 -> 2^-126 (normalized); see PATCH_NOTES.md
LOG2E = 1.4426950408889634
_BITS_ONE = 0x3F800000                     # 1.0f


@torch.no_grad()
def h15_weights_from_x(x: torch.Tensor) -> torch.Tensor:
    """x: already the kernel's post-`scale_subtract_rowmax`, DOUBLED log2 value.
    Reproduces combine_h15: n = add.rn(x); bits = ((n & 0x3FF) << 22) + 0x3F800000; reinterpret.
    Returns (1.0 or 1.5) * 2^{floor(n/2)} for finite x; 0 where x is -inf (masked)."""
    finite = torch.isfinite(x)
    xc = torch.clamp(torch.where(finite, x, torch.zeros_like(x)), min=CLAMP_LO)
    n = torch.round(xc)                    # torch.round is ties-to-even == add.rn
    ni = n.to(torch.int32)
    bits = ((ni & 0x3FF) << 22) + _BITS_ONE
    w = bits.view(torch.float32)
    return torch.where(finite, w, torch.zeros_like(w))


@torch.no_grad()
def h15_probs_from_scores(scores: torch.Tensor, valid: torch.Tensor, max_offset: int = 0) -> torch.Tensor:
    """Full kernel path reference: rowmax-subtract -> double -> H15 weights -> normalize.
    Matches what the patched kernel computes (rowmax-anchored, phase-aligned)."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, float("-inf"))
    smax = sigma.amax(dim=-1, keepdim=True)
    x = 2.0 * (sigma - smax) + 2.0 * float(max_offset)     # scale_log2_exp / max_offset_exp doubling
    x = torch.where(valid, x, torch.full_like(x, float("-inf")))
    w = h15_weights_from_x(x)
    w = torch.where(valid, w, torch.zeros_like(w))
    return w / w.sum(dim=-1, keepdim=True)


@torch.no_grad()
def h15_lattice_index_from_scores(scores: torch.Tensor, valid: torch.Tensor, max_offset: int = 0) -> torch.Tensor:
    """The integer half-octave lattice index n the kernel rounds to (for L1 comparison).
    Masked positions -> a sentinel (int32 min)."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, float("-inf"))
    smax = sigma.amax(dim=-1, keepdim=True)
    x = 2.0 * (sigma - smax) + 2.0 * float(max_offset)
    xc = torch.clamp(torch.where(valid, x, torch.zeros_like(x)), min=CLAMP_LO)
    n = torch.round(xc).to(torch.int32)
    return torch.where(valid, n, torch.full_like(n, torch.iinfo(torch.int32).min))
