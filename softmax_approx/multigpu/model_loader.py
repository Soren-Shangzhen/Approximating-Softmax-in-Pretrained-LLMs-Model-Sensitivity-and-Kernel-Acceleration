"""Multi-GPU model loading for the 70B/72B models, with the intervention attention backend registered.

Same backend and registration as softmax_approx.model_loader; the model is sharded with device_map="balanced"
over all visible GPUs, keeping APPROX_SOFTMAX_GPU_RESERVE_GB (default 15) GiB free per GPU, and the resulting
device map is checked to use CUDA devices only (APPROX_SOFTMAX_REQUIRE_ALL_VISIBLE_GPUS=1 also requires every
visible GPU to be used).  With one visible GPU it behaves like softmax_approx.model_loader.load.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from softmax_approx.model_loader import ATTN_IMPLEMENTATION, MASK_REGISTERED  # registers the backend

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1","true","yes","y","on"}


def _normalize_device_map_value(v):
    if isinstance(v, int):
        return ("cuda", int(v))
    s = str(v).strip().lower()
    if s.isdigit():
        return ("cuda", int(s))
    if s.startswith("cuda:"):
        try:
            return ("cuda", int(s.split(":",1)[1]))
        except Exception:
            return ("other", s)
    return ("other", s)


def _device_kwargs() -> dict:
    nvis = int(torch.cuda.device_count())
    if nvis <= 1:
        return dict(device_map="cuda")
    reserve_gb = float(os.environ.get("APPROX_SOFTMAX_GPU_RESERVE_GB","15"))
    if reserve_gb < 0:
        raise ValueError("APPROX_SOFTMAX_GPU_RESERVE_GB must be non-negative")
    max_memory = {}
    for i in range(nvis):
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        usable = int(total - reserve_gb)
        if usable <= 0:
            raise RuntimeError(f"GPU {i}: total={total:.2f} GiB reserve={reserve_gb:.2f} GiB")
        max_memory[i] = f"{usable}GiB"
    print(f"[load] balanced sharding; reserve={reserve_gb:.1f} GiB/GPU max_memory={max_memory}")
    return dict(device_map="balanced", max_memory=max_memory)


def _validate_gpu_only_device_map(model) -> None:
    nvis = int(torch.cuda.device_count())
    if nvis <= 1:
        return
    dm = getattr(model, "hf_device_map", None)
    if not isinstance(dm, dict) or not dm:
        raise RuntimeError("multi-GPU load requested but hf_device_map is missing/empty")
    used=set(); bad=[]
    for module_name,value in dm.items():
        kind,parsed=_normalize_device_map_value(value)
        if kind=="cuda":
            used.add(int(parsed))
        else:
            bad.append((module_name,str(value)))
    if bad:
        raise RuntimeError(f"GPU-only sharding gate failed; non-CUDA placements: {bad[:12]}")
    if not used:
        raise RuntimeError("GPU-only sharding gate failed: no CUDA devices used")
    if not used.issubset(set(range(nvis))):
        raise RuntimeError(f"unexpected CUDA devices: used={sorted(used)} visible={list(range(nvis))}")
    if _truthy_env("APPROX_SOFTMAX_REQUIRE_ALL_VISIBLE_GPUS","0"):
        expected=set(range(nvis))
        if used != expected:
            raise RuntimeError(f"not all visible GPUs used: used={sorted(used)} expected={sorted(expected)}")
    print(f"[load] hf_device_map uses CUDA devices {sorted(used)} / visible={nvis}")


def load(name=DEFAULT_MODEL, impl=ATTN_IMPLEMENTATION, trust_remote_code=False):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        attn_implementation=impl,
        trust_remote_code=trust_remote_code,
        **_device_kwargs(),
    )
    model.eval()
    _validate_gpu_only_device_map(model)
    return model, tok
