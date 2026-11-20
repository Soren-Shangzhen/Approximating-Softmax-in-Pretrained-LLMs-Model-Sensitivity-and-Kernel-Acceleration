#!/usr/bin/env python3
"""Multi-tile correctness gate (Appendix B.2, Figure S1B): seqlen 512 = 4 key tiles of 128.

P is read back with a blocked one-hot V (block b of V selects P[:, 128 b : 128 (b+1)]).  This exercises the
running-maximum and rescale path that the single-tile gate cannot reach.  The kernel's normalised probabilities
are compared with stock FA4: log2(P1/P0) must lie in the operator's octave envelope [-0.25, 0.335] (Appendix B.1)
widened by the E4M3 storage tolerance 0.09 (normalisation can shift whole rows; Figure S1B shows both the
normalised ratio and the recovered unnormalised error); the maximal entry of each row must be (nearly) exact and
the row sums close to 1.  L15_DUMP=<file.pt> saves the per-element arrays behind Figure S1B.

    python kernel/correctness/gate_multi_tile.py
"""
import os, sys, subprocess, tempfile
from pathlib import Path
_SRC = r'''
import os, sys, torch
os.environ["FA_USE_POT_H15"] = sys.argv[1]
from flash_attn.cute.interface import flash_attn_func
S, D, out = 512, 128, sys.argv[2]
g = torch.Generator(device="cuda").manual_seed(0)
mk = lambda: (torch.randn(1,S,1,D,generator=g,device="cuda")*0.5).clamp(-6,6).to(torch.float8_e4m3fn)
q, k = mk(), mk()
P = torch.zeros(S, S)
for b in range(S // D):                       # block b of V selects P[:, b*D:(b+1)*D]
    v = torch.zeros(1,S,1,D, device="cuda")
    idx = torch.arange(b*D, (b+1)*D, device="cuda")
    v[0, idx, 0, idx - b*D] = 1.0
    o = flash_attn_func(q, k, v.to(torch.float8_e4m3fn), causal=True)[0]
    torch.cuda.synchronize()
    P[:, b*D:(b+1)*D] = o[0,:,0,:].float().cpu()
torch.save(P, out)
'''
LO, HI = -0.25, 0.335                          # Rowmax-H15 envelope (octaves), Appendix B.1
E4M3 = 0.09                                    # ~2^-4 rel error on both P0 and P1 -> log2 slack
def run(arm, td):
    s = Path(td)/"_a.py"; s.write_text(_SRC); o = str(Path(td)/f"P{arm}.pt")
    subprocess.run([sys.executable, str(s), arm, o], check=True, stderr=subprocess.DEVNULL)
    import torch; return torch.load(o)
import torch
with tempfile.TemporaryDirectory() as td:
    P0, P1 = run("0", td), run("1", td)
S = P0.shape[0]; valid = torch.tril(torch.ones(S,S,dtype=torch.bool))
rs0, rs1 = P0.sum(-1), P1.sum(-1)
print(f"rowsum A0: [{rs0.min():.4f},{rs0.max():.4f}]  A1: [{rs1.min():.4f},{rs1.max():.4f}]")
m = valid & (P0 > 1e-4) & (P1 > 0)
e = torch.log2(P1[m] / P0[m])
out = ((e < LO-E4M3) | (e > HI+E4M3)).float().mean().item()
print(f"log2(P1/P0) over {m.sum().item()} entries: mean={e.mean():+.4f} std={e.std():.4f} "
      f"min={e.min():+.3f} max={e.max():+.3f}")
print(f"  outside [{LO-E4M3:+.2f},{HI+E4M3:+.2f}]: {100*out:.3f}%   (expect ~0)")
am = P0.masked_fill(~valid, -1).argmax(-1)
ea = torch.log2(P1[torch.arange(S), am] / P0[torch.arange(S), am])
print(f"argmax entry log2 err: mean={ea.mean():+.4f} max|.|={ea.abs().max():.4f}  (phase-aligned: ~0 ± E4M3)")
import os
_dump = os.environ.get("L15_DUMP")
if _dump:
    torch.save({"P0": P0, "P1": P1, "valid": valid,
                "log2_ratio": torch.where(m, torch.log2(P1.clamp_min(1e-30)/P0.clamp_min(1e-30)),
                                          torch.zeros_like(P0)),
                "mask_used": m, "argmax_idx": am, "argmax_log2_err": ea,
                "meta": dict(seqlen=S, n_block=128, dtype="fp8_e4m3",
                             envelope=(LO, HI), e4m3_slack=E4M3)}, _dump)
    print(f"dumped per-element arrays to {_dump}")
ok = out < 0.005 and ea.abs().max() < 0.2 and abs(rs1.mean()-1) < 0.02
print("L1.5 PASS" if ok else "L1.5 FAIL")
sys.exit(0 if ok else 1)
