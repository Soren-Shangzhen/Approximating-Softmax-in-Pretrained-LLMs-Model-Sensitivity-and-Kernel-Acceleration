#!/usr/bin/env python3
"""Entry point of the characterization pipeline for Llama-3.1-70B and Qwen2.5-72B (multi-GPU).

Registers the two model aliases with the modern_core profile (T00, T03, T04, T06, T08, T10, T13) plus the
allocation_extended feature (D_R(16), D_R(32), I_R), and the long-context extension used by T11 (App. C):
T11 runs the deep-reference long-context implementation for these two models, and T12 needs only T00, T03 and
T06.  Everything else is softmax_approx.multigpu.characterization unchanged.

    python experiments/characterization/large_model_entry.py --model qwen72 --out-root runs/characterization --resume
    python experiments/characterization/large_model_entry.py --model qwen72 --tasks T11,T12 \
        --contexts 2048,8192,16384 --out-root runs/characterization_large_T11_T12 --resume
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softmax_approx.multigpu import characterization as cm  # noqa: E402

cm.MODEL_REGISTRY["qwen72"] = dict(
    hf="Qwen/Qwen2.5-72B", family="Qwen", approx_params="72B",
    profile="modern_core", extras=["allocation_extended", "longctx_large_anchor"],
)
cm.MODEL_REGISTRY["llama70"] = dict(
    hf="meta-llama/Llama-3.1-70B", family="Llama", approx_params="70B",
    profile="modern_core", extras=["allocation_extended", "longctx_large_anchor"],
)

_original_run_T11 = cm.CrossModelRunner.run_T11


def _run_T11_large_model(self):
    """T11 for the large models reuses the deep-reference long-context implementation unchanged."""
    if "longctx_large_anchor" not in self.features:
        return _original_run_T11(self)
    already_deep = "deep_reference" in self.features
    self.features.add("deep_reference")
    try:
        return _original_run_T11(self)
    finally:
        if not already_deep:
            self.features.discard("deep_reference")


cm.CrossModelRunner.run_T11 = _run_T11_large_model

# T12 evaluates mean_sparse_softmax itself and reads no T01 output, so T01 is not a dependency here.
cm.TASK_DEPS["T12"] = ["T00", "T03", "T06"]


if __name__ == "__main__":
    cm.main()
