#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-tile correctness gate (Appendix B.2, Figure S1A): kernel probabilities versus the Rowmax-H15 reference.

flash_attn.cute exposes no attention probabilities, and the kernel cannot be fed an exact FP32 score tensor
(QK^T runs on tensor cores), so P is read back through the PV product: with a one-hot V (v[0, j, 0, :] = e_j) and
head_dim_v == seqlen_k, O[0, i, 0, j] == P[i, j] up to the V/O storage precision.  With seqlen <= 128 the row is a
single key tile, so the running maximum equals the final maximum and the whole-row reference applies.  The scores
are recovered from the stock-FA4 run (A0): log P0 shifted to the row maximum is sigma - sigma_max in nats.

The tolerance is set by the FP8 E4M3 storage grid of V and O, not by the operator's envelope.  Measured on B200:
mean L1 per row 4.253e-3, maximum 2.469e-2 (128 rows).  Each arm runs in its own process because
FA_USE_POT_H15 is read when the kernel is built.  L1_DUMP=<file.pt> saves the per-element arrays behind Figure S1A.

    python kernel/correctness/gate_single_tile.py --seqlen 128 --max-offset 8
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))

_ARM_SRC = r'''
import os, sys
os.environ["FA_USE_POT_H15"] = sys.argv[1]
import torch
from flash_attn.cute.interface import flash_attn_func
S = int(sys.argv[3]); out = sys.argv[2]
H, D = 1, S                      # head_dim_v must equal seqlen for the one-hot trick
g = torch.Generator(device="cuda").manual_seed(0)
def mk():
    t = (torch.randn(1,S,H,D, generator=g, device="cuda") * 0.5).clamp(-6, 6)
    return t.to(torch.float8_e4m3fn)
q, k = mk(), mk()
v = torch.zeros(1,S,H,D, device="cuda")
v[0, torch.arange(S), 0, torch.arange(S)] = 1.0
v = v.to(torch.float8_e4m3fn)
o = flash_attn_func(q, k, v, causal=True)[0]
torch.cuda.synchronize()
torch.save({"P": o[0,:,0,:].float().cpu()}, out)
'''

MEAN_L1_TOL = 1.0e-2      # FP8 E4M3 storage grid; measured 4.253e-3
MAX_L1_TOL  = 5.0e-2      # measured 2.469e-2


def run_arm(arm: str, seqlen: int, tmpdir: str) -> "torch.Tensor":
    import torch
    src = Path(tmpdir) / "_arm.py"; src.write_text(_ARM_SRC)
    out = str(Path(tmpdir) / f"P{arm}.pt")
    subprocess.run([sys.executable, str(src), arm, out, str(seqlen)],
                   check=True, stderr=subprocess.DEVNULL)
    return torch.load(out)["P"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=128, help="must be <= n_block_size (single tile)")
    ap.add_argument("--max-offset", type=int, default=8, help="FP8 path uses 8")
    a = ap.parse_args()
    assert a.seqlen <= 128, "single tile only; see module docstring"

    import torch
    from h15_reference import h15_probs_from_scores

    with tempfile.TemporaryDirectory() as td:
        P0 = run_arm("0", a.seqlen, td)          # stock FA4 = exact softmax
        P1 = run_arm("1", a.seqlen, td)          # patched = H15

    S = P0.shape[0]
    valid = torch.tril(torch.ones(S, S, dtype=torch.bool))
    logP0 = torch.where(valid, P0.clamp_min(1e-30).log(), torch.tensor(float("-inf")))
    scores = logP0 - logP0.max(-1, keepdim=True).values      # == sigma - sigma_max, in nats
    ref = h15_probs_from_scores(scores[None, None], valid[None, None],
                                max_offset=a.max_offset)[0, 0]

    l1 = (P1 - ref).abs().sum(-1)
    mean_l1, max_l1 = float(l1.mean()), float(l1.max())
    _dump = os.environ.get("L1_DUMP")
    if _dump:
        torch.save({"P0": P0, "P1": P1, "P_ref": ref, "valid": valid,
                    "l1_per_row": l1, "scores_recovered": scores,
                    "meta": dict(seqlen=S, max_offset=a.max_offset, dtype="fp8_e4m3",
                                 note="single tile; V=one-hot so O==P")}, _dump)
        print(f"dumped per-element arrays to {_dump}")
    ok = mean_l1 <= MEAN_L1_TOL and max_l1 <= MAX_L1_TOL

    print(f"rows={S}  A0 rowsum[:3]={P0.sum(-1)[:3].tolist()}")
    print(f"A1 vs reference: mean L1/row={mean_l1:.4e} (tol {MEAN_L1_TOL:.1e})  "
          f"max L1/row={max_l1:.4e} (tol {MAX_L1_TOL:.1e})")
    print(f"  A1  [10,:4] = {[round(x,6) for x in P1[10,:4].tolist()]}")
    print(f"  ref [10,:4] = {[round(x,6) for x in ref[10,:4].tolist()]}")
    print(f"  A0  [10,:4] = {[round(x,6) for x in P0[10,:4].tolist()]}")
    print("L1 PASS" if ok else "L1 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
