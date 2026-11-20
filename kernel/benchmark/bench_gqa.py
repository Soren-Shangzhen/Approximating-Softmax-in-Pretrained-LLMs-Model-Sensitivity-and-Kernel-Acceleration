#!/usr/bin/env python3
"""Grouped-query attention call latency (32 query / 8 key-value heads; Table B5e), one arm per process.
Same timing protocol as bench_arm.py.  Usage: bench_gqa.py <arm> <out.csv> [--hq 32 --hkv 8 ...]"""
import os, sys, csv, argparse, subprocess
ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("out")
ap.add_argument("--hq", type=int, default=32); ap.add_argument("--hkv", type=int, default=8)
ap.add_argument("--seqlens", default="8192"); ap.add_argument("--hd", type=int, default=128)
ap.add_argument("--dtype", default="fp8_e4m3"); ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--causal", type=int, default=1); ap.add_argument("--runs", type=int, default=7)
ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--tag", default="gqa")
a = ap.parse_args()
os.environ["FA_USE_POT_H15"] = a.arm          # must precede import
import torch
from flash_attn.cute.interface import flash_attn_func
DT = {"fp8_e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
def gpu_stat():
    try:
        s = subprocess.check_output(["nvidia-smi","--query-gpu=clocks.sm,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"], timeout=5).decode().strip().split(",")
        return int(float(s[0])), int(float(s[1])), float(s[2])
    except Exception: return -1, -1, -1.0
rows = []
for S in [int(x) for x in a.seqlens.split(",")]:
    g = torch.Generator(device="cuda").manual_seed(0)
    def mk(h):
        t = torch.randn(a.batch, S, h, a.hd, generator=g, device="cuda", dtype=torch.float32)
        if DT is torch.float8_e4m3fn: t = (t*0.5).clamp(-6, 6)
        return t.to(DT)
    q, k, v = mk(a.hq), mk(a.hkv), mk(a.hkv)
    causal = bool(a.causal)
    for _ in range(a.warmup): flash_attn_func(q,k,v,causal=causal)
    torch.cuda.synchronize()
    for r in range(a.runs):
        st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
        for _ in range(a.iters): flash_attn_func(q,k,v,causal=causal)
        en.record(); torch.cuda.synchronize(); ms = st.elapsed_time(en)/a.iters
        fl = 2*2*a.batch*a.hq*S*S*a.hd*(0.5 if causal else 1.0)
        clk,tmp,pw = gpu_stat()
        rows.append(dict(arm=f"A{a.arm}", tag=a.tag, dtype=a.dtype, seqlen=S, hq=a.hq, hkv=a.hkv,
                         hd=a.hd, batch=a.batch, causal=int(causal), run=r,
                         latency_ms=round(ms,5), tflops=round(fl/(ms*1e-3)/1e12,1),
                         sm_clock=clk, temp_c=tmp, power_w=pw))
        print(f"A{a.arm} GQA {a.hq}q/{a.hkv}kv {a.dtype} s={S} run{r}: {ms:.4f} ms {rows[-1]['tflops']} TF")
    del q,k,v; torch.cuda.empty_cache()
with open(a.out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {a.out} ({len(rows)} rows)")
