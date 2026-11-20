#!/usr/bin/env python3
"""CPU tests of the Rowmax-H15 kernel reference (no GPU).  Run: python kernel/reference/test_h15_reference.py

  1. lattice values: even n -> 2^(n/2), odd n -> 1.5 * 2^((n-1)/2); n = -252 -> 2^-126 (> 0); -254 is clamped;
  2. masked entries (-inf) -> 0 and ties to even (the kernel's add.rn);
  3. the reference equals softmax_approx.lattice_operators (whole-row Rowmax-H15) bit for bit;
  4. anchor: the maximal entry of every row has weight exactly 1 (row-maximum anchor), whereas the
     absolute-anchored {1, 1.5} x 2^k lattice (Table B2 control ilpot_h15_v1) gives it 0.75-1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from h15_reference import h15_probs_from_scores, h15_weights_from_x  # noqa: E402
from softmax_approx import lattice_operators as lops  # noqa: E402

LOG2E = 1.4426950408889634


def test_lattice_values():
    n = torch.arange(-252, 33, dtype=torch.float32)
    w = h15_weights_from_x(n)
    for i, nv in enumerate(n.tolist()):
        nv = int(nv)
        expected = (1.5 if nv & 1 else 1.0) * (2.0 ** (nv // 2))
        assert float(w[i]) == expected, f"n={nv}: got {float(w[i])}, expected {expected}"
    assert float(h15_weights_from_x(torch.tensor([-252.0]))) == 2.0 ** -126
    assert float(h15_weights_from_x(torch.tensor([-254.0]))) == 2.0 ** -126     # clamped
    print("PASS lattice values (even/odd, boundary -252 -> 2^-126)")


def test_masked_and_ties():
    w = h15_weights_from_x(torch.tensor([float("-inf"), 0.5, -0.5, 1.5, 2.5]))
    assert w.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0], w.tolist()
    print("PASS masked -> 0; ties to even (round(0.5)=0, round(1.5)=2, round(2.5)=2)")


def _rows(seed=0, q=256, k=256, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, 4, q, k, generator=g) * scale
    s[..., 0] += 20.0
    valid = (torch.arange(k).view(1, -1) <= torch.arange(q).view(-1, 1)).view(1, 1, q, k).expand(1, 4, q, k)
    return s, valid


def test_matches_release_operator():
    s, valid = _rows()
    p_ref = h15_probs_from_scores(s, valid, max_offset=0)
    p_rel, _ = lops.probs(s, valid, "h15rowmax")
    assert torch.equal(p_ref, p_rel)
    print("PASS reference == lattice_operators h15rowmax (bit for bit)")


def test_rowmax_anchor():
    s, valid = _rows(seed=1)
    sigma = (s * LOG2E).masked_fill(~valid, float("-inf"))
    am = sigma.argmax(-1, keepdim=True)
    x = 2.0 * (sigma - sigma.amax(-1, keepdim=True))
    w = h15_weights_from_x(torch.where(valid, x, torch.full_like(x, float("-inf"))))
    assert torch.equal(w.gather(-1, am), torch.ones_like(am, dtype=torch.float32))
    wa = lops.weights_absolute_h15(s, valid)
    wmax = wa.gather(-1, am)
    print(f"PASS row-maximum anchor: maximal entry weight == 1; absolute lattice gives "
          f"[{float(wmax.min()):.3f}, {float(wmax.max()):.3f}]")


if __name__ == "__main__":
    test_lattice_values()
    test_masked_and_ties()
    test_matches_release_operator()
    test_rowmax_anchor()
    print("ALL REFERENCE TESTS PASSED")
