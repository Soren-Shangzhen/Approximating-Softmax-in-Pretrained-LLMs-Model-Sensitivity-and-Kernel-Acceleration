"""Transformers attention backend that calls the (patched) FlashAttention-4 forward, for model-level NLL.

FA_USE_POT_H15 = 0|1   arm (stock FA4 / Rowmax-H15); read when the kernel is built, so one arm per process
FA_EVAL_FP8    = 0|1   quantize Q/K/V to E4M3 with one symmetric scale per tensor (the direct FP8 attempt of
                       Appendix B.2; its per-tensor quantization error dominated, so the paper reports the FP8 path
                       only as a semantic simulation).  The scales are folded out exactly: softmax_scale /= s_q s_k,
                       output /= s_v, and both arms get the identical quantization.
Supports causal attention without logit softcapping (head dimension 128 was validated).
"""
import os
import torch
from flash_attn.cute.interface import flash_attn_func

ARM = os.environ.get("FA_USE_POT_H15", "0")
FP8 = os.environ.get("FA_EVAL_FP8", "0") == "1"
_MAX = 448.0
_HR = float(os.environ.get("FA_EVAL_FP8_HEADROOM", "1.0"))


def _q8(t):
    amax = t.abs().amax().clamp_min(1e-6)
    s = (_MAX * _HR) / amax
    return (t.float() * s).clamp(-_MAX, _MAX).to(torch.float8_e4m3fn), float(s)


def fa4cute_attention(module, query, key, value, attention_mask,
                      scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = getattr(module, "scaling", None) or (query.shape[-1] ** -0.5)
    if kwargs.get("attn_logit_softcapping") is not None:
        raise NotImplementedError("logit softcapping unsupported on this path")
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    if not FP8:
        ret = flash_attn_func(q, k, v, softmax_scale=float(scaling), causal=True)
        o = ret[0] if isinstance(ret, (tuple, list)) else ret
        return o.contiguous(), None
    q8, sq = _q8(q); k8, sk = _q8(k); v8, sv = _q8(v)
    ret = flash_attn_func(q8, k8, v8, softmax_scale=float(scaling) / (sq * sk), causal=True)
    o = ret[0] if isinstance(ret, (tuple, list)) else ret
    return (o.float() / sv).to(query.dtype).contiguous(), None
