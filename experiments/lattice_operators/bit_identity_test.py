#!/usr/bin/env python3
"""Finite bit-level test of the add-and-shift construction (Appendix B.1-B.2), CPU only (about a minute).

For k = 0..13 the FP32 pattern of the patched kernel generalised to k fractional bits,

    y = fp32(2^k x + 0x4B400000)  (round to nearest even);   w_bits = (bits(y) << (23 - k)) + 0x3F800000,

is compared bit for bit with the reference S(Q_{2^-k}(x)) = S_k(x) of softmax_approx.lattice_operators, where
Q_{2^-k}(x) = round_RNE(2^k x) / 2^k and S(x) = 2^floor(x) (1 + x - floor(x)).  Rowmax-H15 is k = 1.  The inputs,
the same 2,295,040 FP32 values for every k, are 2,000,000 uniform draws in [-126, 0] (numpy default_rng(0)), the
integers 0, -1, ..., -126 and every multiple of 7 * 2^-14 from 0 down to -126, which contain round-to-nearest-even
ties of 2^k x for every k <= 13; both sides take x after the clamp at -126.  Positive x (possible under FA4's
deferred rescale) are not tested.  This is a finite test on the listed inputs, not a proof.

Expected output: "k=.. mismatches=0/2295040" for every k.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softmax_approx.lattice_operators import S_k  # noqa: E402

MAGIC = np.float32(12582912.0)          # 0x4B400000 = 1.5 * 2^23


def addshift_bits(x, k):
    """2^k x + magic (RNE), bits << (23 - k), + 0x3F800000, on a numpy float32 array."""
    y = (x * np.float32(2.0 ** k)).astype(np.float32) + MAGIC
    b = y.view(np.int32).astype(np.int64)
    return (((b << (23 - k)) + 0x3F800000) & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def main():
    rng = np.random.default_rng(0)
    x = np.concatenate([-rng.random(2_000_000) * 126.0, -np.arange(0, 127, dtype=np.float64),
                        -np.arange(0, 126 * 2**14 + 1, 7, dtype=np.float64) / 2**14]).astype(np.float32)
    x = np.maximum(x, np.float32(-126.0))
    bad_total = 0
    for k in range(0, 14):
        w = addshift_bits(x, k)
        ref = S_k(torch.from_numpy(x), k).numpy()
        bad = int((w.view(np.uint32) != ref.view(np.uint32)).sum())
        bad_total += bad
        print(f"k={k:2d} mismatches={bad}/{x.size}")
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
