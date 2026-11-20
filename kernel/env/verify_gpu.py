#!/usr/bin/env python3
"""Check that the GPU is a B200 (sm_100), not B300/GB300 (sm_103) or consumer Blackwell (sm_120)."""
import sys
import torch
def main():
    assert torch.cuda.is_available(), "no CUDA device"
    cap=torch.cuda.get_device_capability(); name=torch.cuda.get_device_name(0)
    major,minor=cap
    print(f"device: {name}  compute capability: sm_{major}{minor}")
    props=torch.cuda.get_device_properties(0)
    print(f"SMs: {props.multi_processor_count}  total mem: {props.total_memory/1e9:.1f} GB")
    print(f"driver/runtime: torch cuda {torch.version.cuda}")
    try:
        import flash_attn.cute as fac
        import flash_attn
        print(f"flash_attn: {getattr(flash_attn,'__version__','?')}  cute pkg: {fac.__file__}")
    except Exception as e:
        print(f"flash_attn.cute import: FAILED ({e})")
    # HARD GATE: must be sm_100 (B200). sm_103 = B300 (native fast exp2, ex2_emu_freq=0 -> no point).
    if (major,minor)!=(10,0):
        print(f"FATAL: expected sm_100 (B200), got sm_{major}{minor}. "
              f"sm_103=B300/GB300 has fast native exp2 (ex2_emu_freq=0) -> H15 gains nothing there; "
              f"sm_120=consumer Blackwell (no TMEM). The Rowmax-H15 measurements need a B200.")
        return 2
    print("OK: sm_100 (B200) confirmed.")
    return 0
if __name__=="__main__": sys.exit(main())
