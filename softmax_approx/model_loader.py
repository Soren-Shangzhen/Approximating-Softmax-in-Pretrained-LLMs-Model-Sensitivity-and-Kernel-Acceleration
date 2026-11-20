"""Model loading with the intervention attention backend registered in Transformers (5.x)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

from .attention import attention_forward

ATTN_IMPLEMENTATION = "approx_softmax"

AttentionInterface.register(ATTN_IMPLEMENTATION, attention_forward)

# Also register the eager mask interface, so that Transformers builds the same 4D mask for this backend as
# for "eager".  Without it the mask is None and attention.py builds its own causal mask; both agree bit for
# bit without padding, but only the registered mask handles padding (batched zero-shot evaluation).
try:
    from transformers import masking_utils as _MU
    _mask_fn = None
    for _name in ("eager_mask", "eager_mask_function"):
        if hasattr(_MU, _name):
            _mask_fn = getattr(_MU, _name)
            break
    if _mask_fn is None:
        raise AttributeError("no eager mask function found in transformers.masking_utils")
    _MU.AttentionMaskInterface.register(ATTN_IMPLEMENTATION, _mask_fn)
    MASK_REGISTERED = True
except Exception:
    MASK_REGISTERED = False

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


def load(name=DEFAULT_MODEL, impl=ATTN_IMPLEMENTATION, trust_remote_code=False):
    """BF16 weights on one GPU, evaluation mode."""
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=impl,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tok
