#!/usr/bin/env python3
"""One FlashAttention-4 configuration as a profiling target (Nsight Systems: Table B4, Figure S2B; Nsight Compute).

The arm is fixed through FA_USE_POT_H15 before flash_attn is imported.  Fixed-seed inputs, --warmup calls and a
synchronization, then --iters timed calls between two host timestamps followed by a synchronization; prints and
optionally writes (--out, JSON) the host time per timed call.  For Table B4 the paper traced every kernel and
length in its own process with `nsys profile --trace=cuda` (50 warm-up + 200 timed calls): the kernel duration is
the median of the cuda_gpu_kern_sum report over all 250 traced calls, the call latency the host time per timed call.
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["0", "1"], default="1")
ap.add_argument("--seqlen", type=int, default=8192)
ap.add_argument("--causal", type=int, default=1)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--hd", type=int, default=128)
ap.add_argument("--dtype", default="fp8_e4m3")
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--out", default=None)
a = ap.parse_args()

os.environ["FA_USE_POT_H15"] = a.arm             # must precede the import
import torch  # noqa: E402
from flash_attn.cute.interface import flash_attn_func  # noqa: E402

DT = {"fp8_e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
g = torch.Generator(device="cuda").manual_seed(0)
shp = (1, a.seqlen, a.heads, a.hd)                # flash_attn.cute layout is (b, s, h, d)


def mk():
    t = torch.randn(shp, generator=g, device="cuda", dtype=torch.float32)
    if DT is torch.float8_e4m3fn:
        t = (t * 0.5).clamp(-6, 6)
    return t.to(DT)


q, k, v = mk(), mk(), mk()
causal = bool(a.causal)
for _ in range(a.warmup):
    flash_attn_func(q, k, v, causal=causal)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(a.iters):
    flash_attn_func(q, k, v, causal=causal)
torch.cuda.synchronize()
host_us = 1e6 * (time.perf_counter() - t0) / max(a.iters, 1)
res = dict(arm=f"A{a.arm}", dtype=a.dtype, seqlen=a.seqlen, causal=int(causal), heads=a.heads, hd=a.hd,
           warmup=a.warmup, iters=a.iters, host_us_per_call=host_us)
print(json.dumps(res))
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
