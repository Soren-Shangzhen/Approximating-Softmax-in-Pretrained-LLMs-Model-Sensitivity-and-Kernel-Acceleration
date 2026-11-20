#!/usr/bin/env python3
"""Measured model-level fidelity of the patched kernel on the BF16 path (Table 2, Table B2; Appendix B.2.1).

One model, one arm, one process (FA_USE_POT_H15 is read when the kernel is built).  The model runs with BF16
weights and the FlashAttention-4 forward as its attention (fa4cute_attn.py) on the 97 aligned 2048-token blocks of
the WikiText-103 stream (softmax_approx.data, 200,000 tokens), and the per-block NLL (the model's own loss over
2047 predictions) is written to --out.  The ΔNLL of a model is the paired difference A1 - A0 per block
(experiments/lattice_operators/analyze.py for Qwen2.5-1.5B; paper/tables/check_tables.py for all five models).

    python kernel/fidelity/run_ppl_arm.py --arm 0 --model Qwen/Qwen2.5-1.5B --out ppl_A0.csv
    python kernel/fidelity/run_ppl_arm.py --arm 1 --model Qwen/Qwen2.5-1.5B --out ppl_A1.csv
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["0", "1"])
ap.add_argument("--model", required=True, help="Hugging Face repo id or local snapshot directory")
ap.add_argument("--tag", default=None, help="name recorded in the CSV and used for the token cache (default: --model)")
ap.add_argument("--out", required=True)
ap.add_argument("--n-tokens", type=int, default=200_000)
ap.add_argument("--blocks", type=int, default=0, help="0 = all blocks")
a = ap.parse_args()
os.environ["FA_USE_POT_H15"] = a.arm               # must precede the flash_attn import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface  # noqa: E402
import fa4cute_attn  # noqa: E402
from softmax_approx.data import build_tokens  # noqa: E402

AttentionInterface.register("fa4cute", fa4cute_attn.fa4cute_attention)
try:
    from transformers import masking_utils as _MU
    for n in ("eager_mask", "eager_mask_function"):
        if hasattr(_MU, n):
            _MU.AttentionMaskInterface.register("fa4cute", getattr(_MU, n))
            break
except Exception as e:  # noqa: BLE001
    print("mask registration skipped:", e)

tag = a.tag or a.model
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="fa4cute").eval()
cfg = model.config
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(f"model={tag} head_dim={hd} q_heads={cfg.num_attention_heads} "
      f"kv_heads={getattr(cfg, 'num_key_value_heads', None)} softcap={getattr(cfg, 'attn_logit_softcapping', None)}")
assert hd == 128, "the correctness gates were validated only at head dimension 128"
assert not getattr(cfg, "attn_logit_softcapping", None), "logit softcapping is not supported on this path"

ids = build_tokens(tok, n_tokens=a.n_tokens, dataset="wikitext", tag=tag)
SEQ = 2048
rows = []
with torch.no_grad():
    for bi, i in enumerate(range(0, len(ids) - SEQ, SEQ)):
        if a.blocks and bi >= a.blocks:
            break
        c = ids[i:i + SEQ].unsqueeze(0).cuda()
        rows.append(dict(model=tag, arm=f"A{a.arm}", block=bi, seq_len=SEQ, n_pred=SEQ - 1,
                         nll=float(model(c, labels=c).loss)))
        if bi % 20 == 0:
            print(f"  block {bi}: {rows[-1]['nll']:.6f}", flush=True)
m = sum(r["nll"] for r in rows) / len(rows)
print(f"model={tag} arm=A{a.arm} blocks={len(rows)} mean_nll={m:.8f} ppl={math.exp(m):.6f}")
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
