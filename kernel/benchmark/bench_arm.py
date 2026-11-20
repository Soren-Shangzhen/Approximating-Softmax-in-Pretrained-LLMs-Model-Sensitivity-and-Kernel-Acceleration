#!/usr/bin/env python3
"""Attention-forward call latency of one FlashAttention-4 arm (Section 5.3, Appendix B.3), one process per arm.

For each dtype and sequence length: fixed-seed inputs (FP8 inputs scaled by 0.5 and clipped to [-6, 6]),
--warmup calls and a synchronization, then --runs windows of --iters back-to-back calls of the FA4 Python
interface timed with two CUDA events; the window time divided by --iters is the call latency (it includes launch
and dispatch gaps, not input generation or compilation).  SM clock, temperature and board power are read with
nvidia-smi after each window.  --freq / --start-frg override FA4's polynomial-emulation settings of the stock
kernel (arm 0) for this head dimension (Figure 4D, Table B5d).

    python kernel/benchmark/bench_arm.py --arm 0 --out A0.csv --dtypes fp8_e4m3 --seqlens 8192
"""
import os, sys, csv, argparse, subprocess
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dtypes", default="fp8_e4m3,bf16"); ap.add_argument("--seqlens", default="4096,8192,16384")
ap.add_argument("--heads", type=int, default=16); ap.add_argument("--hd", type=int, default=128)
ap.add_argument("--batch", type=int, default=1); ap.add_argument("--causal", type=int, default=1)
ap.add_argument("--runs", type=int, default=7); ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--iters", type=int, default=100); ap.add_argument("--tag", default="")
ap.add_argument("--freq", type=int, default=-1, help="A0 only: override ex2_emu_freq for this hd")
ap.add_argument("--start-frg", type=int, default=-1)
a = ap.parse_args()
os.environ["FA_USE_POT_H15"] = a.arm            # must precede import
import torch
from flash_attn.cute.interface import flash_attn_func
if a.freq >= 0:                                  # monkey-patch tuning dicts before first construction
    import flash_attn.cute.flash_fwd_sm100 as fw
    for D in (fw._TUNING_CONFIG, fw._FP8_TUNING_CONFIG):
        for k in list(D.keys()):
            if k[2] == a.hd:
                D[k] = {**D[k], "ex2_emu_freq": a.freq}
                if a.start_frg >= 0: D[k]["ex2_emu_start_frg"] = a.start_frg
def gpu_stat():
    try:
        s = subprocess.check_output(["nvidia-smi","--query-gpu=clocks.sm,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"], timeout=5).decode().strip().split(",")
        return int(float(s[0])), int(float(s[1])), float(s[2])
    except Exception: return -1, -1, -1.0
DT = {"fp8_e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp8_e5m2": torch.float8_e5m2}
rows = []
for dtype in a.dtypes.split(","):
    for S in [int(x) for x in a.seqlens.split(",")]:
        dt = DT[dtype]; g = torch.Generator(device="cuda").manual_seed(0)
        shp = (a.batch, S, a.heads, a.hd)
        def mk():
            t = torch.randn(shp, generator=g, device="cuda", dtype=torch.float32)
            if dt is torch.float8_e4m3fn: t = (t*0.5).clamp(-6,6)
            return t.to(dt)
        q,k,v = mk(),mk(),mk(); causal = bool(a.causal)
        try:
            for _ in range(a.warmup): flash_attn_func(q,k,v,causal=causal)
            torch.cuda.synchronize()
            for r in range(a.runs):
                st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
                for _ in range(a.iters): flash_attn_func(q,k,v,causal=causal)
                en.record(); torch.cuda.synchronize(); ms = st.elapsed_time(en)/a.iters
                fl = 2*2*a.batch*a.heads*S*S*a.hd*(0.5 if causal else 1.0)
                clk,tmp,pw = gpu_stat()
                rows.append(dict(arm=f"A{a.arm}", tag=a.tag, dtype=dtype, seqlen=S, heads=a.heads, hd=a.hd,
                    batch=a.batch, causal=int(causal), freq=a.freq, start_frg=a.start_frg, run=r,
                    latency_ms=round(ms,5), tflops=round(fl/(ms*1e-3)/1e12,1), sm_clock=clk, temp_c=tmp, power_w=pw))
                print(f"A{a.arm} {a.tag} {dtype} s={S} h={a.heads} hd={a.hd} c={int(causal)} f={a.freq} run{r}: "
                      f"{ms:.4f} ms {rows[-1]['tflops']} TF  {clk}MHz {tmp}C {pw}W")
        except Exception as e:
            print(f"FAILED A{a.arm} {a.tag} {dtype} s={S}: {type(e).__name__}: {e}")
        del q,k,v; torch.cuda.empty_cache()
if rows:
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
