#!/usr/bin/env python3
"""CPU tests of softmax_approx.lattice_operators (Section 5, Appendix B.1-B.2).  Run with pytest or directly."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from softmax_approx import attention  # noqa: E402
from softmax_approx import lattice_operators as lops  # noqa: E402

LOG2E = lops.LOG2E


def _rows(seed=0, b=1, h=3, q=64, k=300, scale=6.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, h, q, k, generator=g) * scale
    valid = (torch.arange(k).view(1, -1) <= torch.arange(q).view(-1, 1) + (k - q)).view(1, 1, q, k).expand(b, h, q, k)
    return s.masked_fill(~valid, float("-inf")), valid


def test_h15_bit_construction_equals_ideal_operator():
    """Eq. (28): the add-and-shift bit pattern of 2x equals H(x) = S(Q_{1/2}(x)) on [-126, 0] (ties included)."""
    x = torch.cat([-torch.arange(0, 126 * 2 ** 10 + 1, 3, dtype=torch.float64) / 2 ** 10,
                   -torch.rand(200_000, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 126]).float()
    assert torch.equal(lops.h15_bits(2.0 * x).view(torch.int32), lops.H(x).view(torch.int32))


def test_h15_octave_envelope():
    """log2 H(x) - x lies in [-1/4, log2(1.5) - 1/4] = [-0.25, 0.33496] (Appendix B.1)."""
    x = -torch.rand(500_000, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 120
    e = torch.log2(lops.H(x.float()).double()) - x.float().double()
    assert float(e.min()) >= -0.25 - 1e-6 and float(e.max()) <= math.log2(1.5) - 0.25 + 1e-6


def test_schraudolph_reference_values():
    x = torch.tensor([0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0])
    assert lops.S(x).tolist() == [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
    assert torch.equal(lops.S_k(x, 1), lops.H(x))


def test_absolute_octave_lattice_invariants():
    s, valid = _rows()
    w = lops.weights_absolute_octave(s, valid)
    assert float(w.max()) <= 1.0
    assert bool(((w == 1.0) & valid).any(-1).all())          # some entry sits at the lattice maximum
    assert bool((w.sum(-1) >= 1.0).all())


def test_absolute_half_octave_exact_max():
    s, valid = _rows(seed=2)
    w = lops.weights_absolute_half_octave_exact_max(s, valid)
    sigma = (s * LOG2E).masked_fill(~valid, float("-inf"))
    f = w.gather(-1, lops._first_argmax(sigma))
    assert float(f.min()) >= 2 ** -0.75 * (1 - 1e-6) and float(f.max()) <= 2 ** 0.25 * (1 + 1e-6)


def test_absolute_h15_bounds():
    s, valid = _rows(seed=3)
    w = lops.weights_absolute_h15(s, valid)
    assert float(w.max()) <= 1.0 and bool((w.sum(-1) >= 0.75).all())


def test_probability_grids_are_powers_of_two():
    s, valid = _rows(seed=4)
    for mode, step in (("pq_log2_b8", 1.0), ("pq_logsqrt2_b8", 0.5)):
        w = lops.weight_fn(mode)(s, valid)
        lw = -torch.log2(w[valid]) / step
        assert torch.equal(lw, torch.round(lw)) and float(lw.max()) <= 255


def test_tiled_semantics_single_tile_equals_whole_row():
    s, valid = _rows(seed=5, q=40, k=128)
    for mode in ("h15tiled", "h15tiled_bf16"):
        assert torch.equal(lops.weight_fn(mode)(s, valid), lops.weights_rowmax_h15(s, valid))


def test_conditional_rescaling_moves_the_anchor():
    """Two key tiles; the second raises the row maximum by 0.7 octave (< the BF16 rescale threshold of 8).
    FP8 semantics (update every tile): the maximal entry keeps weight 1 relative to the final maximum.
    BF16 semantics: the anchor stays at the first tile's maximum, so the maximal entry gets 1.5 instead of 2^0.7."""
    k = 256
    sigma = torch.full((1, 1, 1, k), -3.0)
    sigma[..., 5] = 0.0                      # first tile maximum
    sigma[..., 200] = 0.7                    # second tile maximum
    s = sigma / LOG2E
    valid = torch.ones_like(s, dtype=torch.bool)
    w8 = lops.weights_rowmax_h15_tiled(s, valid)
    w16 = lops.weights_rowmax_h15_tiled_bf16(s, valid)
    assert float(w8[..., 200]) == 1.0
    assert float(w16[..., 200]) == 1.5 and abs(math.log2(1.5) - 0.7) > 0.1


def test_install_into_attention_backend():
    lops.install(attention)
    s, valid = _rows(seed=6, q=16, k=160)
    for mode in ("ilpot", "ilpot_ha", "ilpot_h15", "pq_log2_b8", "pq_logsqrt2_b8", "h15rowmax", "h15tiled",
                 "h15tiled_bf16", "schraudolph_rowmax", "schraudolph_rowmax_k2"):
        p, M = attention._alt_probs(s, valid, mode, 16, layer_idx=0)
        assert torch.allclose(p.sum(-1), torch.ones(p.shape[:-1]), atol=1e-5)
        assert not bool((p[~valid] != 0).any())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
