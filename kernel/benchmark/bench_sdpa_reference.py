#!/usr/bin/env python3
"""External references (Table B5g): PyTorch SDPA with the cuDNN, flash or memory-efficient backend, BF16,
same timing protocol as bench_arm.py (no clock record).  These compute softmax attention with their own
tiling, so they give context, not a controlled baseline.  SDPA layout is (B,H,S,D); flash_attn.cute uses (B,S,H,D)."""
import argparse, csv
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="cudnn", choices=["cudnn","torch_flash","mem_eff"])
ap.add_argument("--out", required=True); ap.add_argument("--dtype", default="bf16")
ap.add_argument("--seqlens", default="4096,8192,16384"); ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--hd", type=int, default=128); ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--causal", type=int, default=1); ap.add_argument("--runs", type=int, default=7)
ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--iters", type=int, default=100)
a = ap.parse_args()
BE = {"cudnn":SDPBackend.CUDNN_ATTENTION,"torch_flash":SDPBackend.FLASH_ATTENTION,
      "mem_eff":SDPBackend.EFFICIENT_ATTENTION}[a.backend]
DT = {"bf16":torch.bfloat16,"fp16":torch.float16}[a.dtype]
F = torch.nn.functional.scaled_dot_product_attention
rows=[]
for S in [int(x) for x in a.seqlens.split(",")]:
    g = torch.Generator(device="cuda").manual_seed(0)
    q,k,v = [torch.randn(a.batch,a.heads,S,a.hd,generator=g,device="cuda",dtype=DT) for _ in range(3)]
    causal = bool(a.causal)
    try:
        with sdpa_kernel(BE):
            for _ in range(a.warmup): F(q,k,v,is_causal=causal)
            torch.cuda.synchronize()
            for r in range(a.runs):
                s0=torch.cuda.Event(True); e0=torch.cuda.Event(True); s0.record()
                for _ in range(a.iters): F(q,k,v,is_causal=causal)
                e0.record(); torch.cuda.synchronize(); ms=s0.elapsed_time(e0)/a.iters
                fl = 2*2*a.batch*a.heads*S*S*a.hd*(0.5 if causal else 1.0)
                rows.append(dict(backend=a.backend,dtype=a.dtype,seqlen=S,heads=a.heads,hd=a.hd,
                                 batch=a.batch,causal=int(causal),run=r,latency_ms=round(ms,5),
                                 tflops=round(fl/(ms*1e-3)/1e12,1)))
                print(f"{a.backend} {a.dtype} s={S} run{r}: {ms:.4f} ms {rows[-1]['tflops']} TF")
    except Exception as e:
        print(f"FAILED {a.backend} s={S}: {type(e).__name__}: {str(e)[:120]}")
    del q,k,v; torch.cuda.empty_cache()
if rows:
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
