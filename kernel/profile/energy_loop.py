#!/usr/bin/env python3
"""GPU board energy per attention forward (Table 2, Table B5f; Appendix B.3), one arm per process.

FP8 causal, seqlen 16384, 16 heads, head dimension 128.  After 30 warm-up calls, nvidia-smi samples power.draw,
clocks.sm and temperature.gpu at a requested 100 ms interval from 2 s before to 1 s after a loop of forwards in
chunks of 50 calls (each followed by a synchronization) that runs for SECONDS (45 in the paper).  With host loop
time T, N completed forwards and P the mean of the samples without the first 20 and last 10, the energy per
forward is P T / N (no idle power subtracted).  Writes <out> (summary) and <out>.samples.csv (raw samples).

    python kernel/profile/energy_loop.py <arm 0|1> <out.csv> [SECONDS]
"""
import os, sys, time, subprocess, csv, statistics as st
arm, out, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 45
os.environ["FA_USE_POT_H15"] = arm
import torch
from flash_attn.cute.interface import flash_attn_func
S,H,D=16384,16,128; g=torch.Generator(device="cuda").manual_seed(0)
mk=lambda:(torch.randn(1,S,H,D,generator=g,device="cuda")*0.5).clamp(-6,6).to(torch.float8_e4m3fn)
q,k,v=mk(),mk(),mk()
for _ in range(30): flash_attn_func(q,k,v,causal=True)
torch.cuda.synchronize()
log=out+".samples.csv"
smp=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu",
    "--format=csv,noheader,nounits","-lms","100"], stdout=open(log,"w"), stderr=subprocess.DEVNULL)
time.sleep(2); t0=time.time(); n=0
while time.time()-t0 < secs:
    for _ in range(50): flash_attn_func(q,k,v,causal=True)
    torch.cuda.synchronize(); n+=50
t1=time.time(); time.sleep(1); smp.terminate()
P=[float(r[1]) for r in csv.reader(open(log)) if len(r)>=2]
P=P[20:-10]  # drop ramp/tail
mean_p=st.mean(P); dt=t1-t0; per_fwd_ms=1000*dt/n; energy_mj=mean_p*dt/n*1000
res=dict(arm=f"A{arm}",n_forwards=n,elapsed_s=round(dt,2),per_fwd_ms=round(per_fwd_ms,4),
         mean_power_w=round(mean_p,1),p_std_w=round(st.pstdev(P),1),energy_per_fwd_mJ=round(energy_mj,2),
         n_samples=len(P),power_min=min(P),power_max=max(P))
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
print(res)
