#!/usr/bin/env python3
"""Fixed-input smoke test of one arm (run once per arm, each in its own process; Appendix B.3).

    python kernel/correctness/smoke_test.py 0 A0.pt     # stock FA4
    python kernel/correctness/smoke_test.py 1 A1.pt     # patched (Rowmax-H15)

FP8 E4M3, causal, seqlen 4096, 16 heads, head dimension 128, fixed seed.  Prints the mean absolute output,
which must differ between the arms (it did not differ when an arm switch was silently ignored) and was
bit-for-bit reproducible across measurement sessions (paper: absmean 0.188664 for A0 and 0.188038 for A1).
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("out")
ap.add_argument("--dtype", default="fp8_e4m3")
ap.add_argument("--seqlen", type=int, default=4096)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--hd", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.environ["FA_USE_POT_H15"] = a.arm          # must be set before flash_attn is imported
import torch  # noqa: E402
from flash_attn.cute.interface import flash_attn_func  # noqa: E402

g = torch.Generator(device="cuda").manual_seed(a.seed)
shp = (1, a.seqlen, a.heads, a.hd)             # flash_attn.cute layout is (b, s, h, d)
_DT = {"fp8_e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16}[a.dtype]


def _mk():
    t = torch.randn(shp, generator=g, device="cuda", dtype=torch.float32)
    if _DT is torch.float8_e4m3fn:
        t = (t * 0.5).clamp(-6.0, 6.0)      # keep well inside the E4M3 normal range
    return t.to(_DT)


q, k, v = _mk(), _mk(), _mk()
o = flash_attn_func(q, k, v, causal=True)[0]
torch.cuda.synchronize()
torch.save({"o": o.float().cpu(), "arm": a.arm,
            "cfg": dict(dtype=a.dtype, seqlen=a.seqlen, heads=a.heads, hd=a.hd, seed=a.seed)}, a.out)
print(f"arm={a.arm} saved {a.out} absmean={o.float().abs().mean().item():.6f}")
