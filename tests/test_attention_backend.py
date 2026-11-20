#!/usr/bin/env python3
"""CPU tests of the Section 4 operators of softmax_approx.attention, configured exactly as the pipeline configures
them (characterization.configure_condition).  Run with pytest or directly."""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from softmax_approx import attention  # noqa: E402
from softmax_approx import characterization as cm  # noqa: E402


def _rows(seed=0, q=48, k=256, scale=4.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, 2, q, k, generator=g) * scale
    valid = (torch.arange(k).view(1, -1) <= torch.arange(q).view(-1, 1) + (k - q)).view(1, 1, q, k).expand(1, 2, q, k)
    return s.masked_fill(~valid, float("-inf")), valid


def _probs(spec, s, valid):
    cm.configure_condition(attention, spec, 2, 0)
    return attention._alt_probs(s, valid, attention.CFG.layer_mode[1], s.shape[-2], layer_idx=1)


def test_temperature_one_is_softmax():
    s, valid = _rows()
    p, _ = _probs(cm.c_temp(1.0), s, valid)
    assert torch.allclose(p, F.softmax(s, -1), atol=1e-7)


def test_pot_error_bound():
    """|log w_hat - log w| <= ln2 / (2m) for the PoT perturbation (Eq. 18)."""
    s, valid = _rows(seed=1)
    for m in (0.5, 1.0, 2.0, 4.0):
        p, _ = _probs(cm.c_pot(m), s, valid)
        D = (s.amax(-1, keepdim=True) - s)[valid]
        lw_hat = torch.log(p[valid].double()) - torch.log(p.amax(-1, keepdim=True).expand_as(p)[valid].double())
        err = (lw_hat + D.double()).abs().max()
        assert float(err) <= math.log(2) / (2 * m) + 1e-5


def test_mean_threshold_same_support():
    """Softmax and uniform weighting keep the same support: the valid scores above the row mean (Eq. 10)."""
    s, valid = _rows(seed=2)
    p_soft, m_soft = _probs(cm.c_binary(True), s, valid)
    p_unif, m_unif = _probs(cm.c_binary(False), s, valid)
    assert torch.equal(m_soft, m_unif)
    n = m_unif.sum(-1, keepdim=True).float()
    assert torch.allclose(p_unif, m_unif.float() / n)
    assert torch.allclose(p_soft.sum(-1), torch.ones(p_soft.shape[:-1]), atol=1e-6)


def test_topk_count():
    s, valid = _rows(seed=3)
    for r in (0.5, 0.15):
        _, M = _probs(cm.c_topk(r, True), s, valid)
        assert torch.equal(M.sum(-1), (valid.sum(-1).float() * r).ceil().long().clamp(min=1))


def test_full_range_grid_converges_to_softmax():
    s, valid = _rows(seed=4)
    err = [float((_probs(cm.c_multibit(K, 4.0, "nearest"), s, valid)[0] - F.softmax(s, -1)).abs().max())
           for K in (16, 64, 256)]
    assert err[0] > err[1] > err[2]


def test_rowmax_pot_weights():
    """Rowmax-PoT: weights 2^-d with integer d in [0, K_max]; the row maximum has weight 1 (Eq. 20)."""
    s, valid = _rows(seed=5)
    for tail, K in (("clamp", 20), ("drop", 12)):
        p, _ = _probs(cm.c_rowmax_pot(K, tail_mode=tail), s, valid)
        w = p / p.amax(-1, keepdim=True)
        d = -torch.log2(w[valid & (w > 0)])
        assert torch.equal(d, torch.round(d)) and float(d.max()) <= K


def test_query_chunking_and_mask_paths_agree():
    mod = types.SimpleNamespace(layer_idx=1, num_key_value_groups=2, scaling=0.125)
    g = torch.Generator().manual_seed(6)
    q = torch.randn(1, 4, 64, 32, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 2, 64, 32, generator=g).to(torch.bfloat16)
    v = torch.randn(1, 2, 64, 32, generator=g).to(torch.bfloat16)
    mask = torch.zeros(1, 1, 64, 64).masked_fill(torch.triu(torch.ones(64, 64, dtype=torch.bool), 1),
                                                 torch.finfo(torch.float32).min)
    for spec in (cm.c_baseline(), cm.c_multibit(21, 4.0, "nearest"), cm.c_pot(1.0)):
        outs = []
        for qc, am in ((0, None), (16, None), (0, mask), (16, mask)):
            cm.configure_condition(attention, spec, 2, qc)
            outs.append(attention.attention_forward(mod, q, k, v, am)[0].float())
        for o in outs[1:]:
            assert torch.allclose(o, outs[0], atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
