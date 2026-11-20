# Rowmax-H15 patch for FlashAttention-4 (commit `0dc2cb48`)

## What the patch changes

FA4's forward kernel for sm_100 evaluates most exponentials of the online softmax with the MUFU `ex2`
instruction and a fraction of them with a polynomial emulation (`ex2_emulation_2`: magic-number floor plus the
cubic `POLY_EX2[3]`), selected by the tuning parameters `ex2_emu_freq` / `ex2_emu_start_frg`. The patch
replaces the exponential of **every** element (no emulation predicate, no MUFU) by `pot_h15_emulation_2`
(`pot_h15.py`): clamp, one packed `add.rn` with the magic constant `0x4B400000`, and one combine. Arm A0 is stock
FA4; arm A1 is the patched kernel, selected by `FA_USE_POT_H15=1` when the kernel is constructed.

Ten anchored text edits in `flash_attn/cute/softmax.py` and `flash_attn/cute/flash_fwd_sm100.py`
(`apply_patch.py`; the resulting diffs are `softmax.diff` and `flash_fwd_sm100.diff`), plus `pot_h15.py` copied
into `flash_attn/cute/`. `apply_patch.py --dry-run` verifies that every anchor occurs exactly once;
`--revert` restores the `.orig` backups byte for byte.

Everything else is unchanged: running maximum and rescaling, row sum and log-sum-exp, pipeline, TMEM layout,
warp specialization, FP8 scaling of P (`max_offset`). The row sum and the PV accumulator consume the same
approximated weights.

## The operator (Eq. 5 / Eq. 28)

With x the base-two exponent that the online exponential receives (relative to the running row maximum,
including the softmax scale and the configuration offset),

    n = round_RNE(2x),   w = (1 + (n mod 2)/2) * 2^floor(n/2),

so even n gives 2^k and odd n gives 1.5 * 2^k. In the kernel the doubled input 2x comes from
`scale_subtract_rowmax`; `add.rn` with `0x4B400000` (2^23 + 2^22) places n in the low mantissa bits;
`(bits << 22) + 0x3F800000` moves bit 0 of n to the leading mantissa position and the rest into the exponent
field. The anchor is the running maximum that online softmax already tracks, so no scale is calibrated.

## Implementation rules

* **Only the exponential path is doubled.** `scale_subtract_rowmax` computes local
  `scale_log2_exp = 2 * scale_log2` and `max_offset_exp = 2 * max_offset` when `use_pot_h15` is set and uses
  them only in its own `fma_packed_f32x2`. Doubling the shared `scale_log2` / `max_offset` instead would square
  the rescale factor in `update_row_max`, halve the rescale threshold, corrupt the log-sum-exp in `finalize` and
  trip the FP8 range assertion `max_offset + rescale_threshold < _LOG2_DTYPE_MAX`.
* **Clamp at -252, not -254.** x is already doubled. -254 would give floor(n/2) = -127, i.e. exponent field 0,
  and the shifted value would wrap; -252 gives 2^-126, still a normal number (-126 on the original scale).
* **Round to nearest even (`add.rn`), not floor (`add.rm`).** Rowmax-H15 rounds to the nearest half octave.
  The PyTorch reference uses `torch.round`, which also rounds ties to even, so it matches the kernel bit for bit.
* **Why a 22-bit shift.** The low 22 bits of the magic constant are zero, so
  `(magic_int + n) << 22 == (n & 0x3FF) << 22`.

## Running the two arms

* `FA_USE_POT_H15` is read when the kernel object is built and passed as a `cutlass.Constexpr`; the CuTe DSL
  JIT caches the compiled kernel per process. Changing the variable inside a process silently reuses the first
  arm. **Every A0/A1 comparison runs each arm in its own process** (all scripts in `kernel/` do).
* The patched kernel anchors each key tile (128 keys at head dimension 128) at its *running* maximum and rescales
  earlier tiles by an exact float factor, so multi-tile rows are not bit-identical to a whole-row reference:
  the single-tile gate is bit-level, the multi-tile gate checks the octave envelope
  (`kernel/correctness/`). `softmax_approx.lattice_operators` implements both semantics in PyTorch
  (`h15tiled`, FP8 path; `h15tiled_bf16`, BF16 path).
* On the BF16 path FA4 skips the online rescale while the row maximum grows by at most `rescale_threshold = 8.0`
  octaves, so the row's largest weight can leave the lattice; on the FP8 path (`rescale_threshold = 0`) it stays
  exact ("Conditional rescaling moves the anchor", Appendix B.2; `tests/test_lattice_operators.py` shows both
  cases on a synthetic multi-tile row).

## Installation note

The FA4 CuTe DSL package can be installed without building the FlashAttention-2 CUDA extension
(`FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`). If `import flash_attn.cute` then fails because the top-level
`flash_attn/__init__.py` imports the missing extension, wrap that import in `try/except` (keep a backup and
restore it afterwards; this edit is outside `apply_patch.py`).
