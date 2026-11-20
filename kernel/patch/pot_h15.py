# Rowmax-H15 exp2 replacement for FlashAttention-4 (Section 5.2, Appendix B.1 of the paper).
#
# Drop-in replacement for utils.ex2_emulation_2: given x already in DOUBLED log2 units
# (produced by scale_subtract_rowmax with scale_log2_exp = 2*scale_log2, max_offset_exp = 2*max_offset),
# quantize to the half-octave 1.5-lattice and reconstruct 2^{floor(n/2)} * (1.0 or 1.5) by bit ops.
# No MUFU / ex2.approx is used: per element only fmax (FMNMX), one packed add.rn (FADD.RN.f32x2),
# and one combine (LEA).  See kernel/patch/PATCH_NOTES.md.
#
# This file is meant to be copied into flash_attn/cute/ by apply_patch.py, so its imports mirror
# utils.py; it runs only inside the CuTe DSL environment of FlashAttention-4 on sm_100 (B200).

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

CLAMP_LO = -252.0                      # floor(n/2) = -126 -> 2^-126 (normalized); NOT -254 (see notes)
MAGIC = float(2 ** 23 + 2 ** 22)       # 0x4B400000


@dsl_user_op
def combine_h15(x_rounded: Float32, *, loc=None, ip=None) -> Float32:
    """(x_rounded_int << 22) + 0x3F800000, reinterpreted as f32.
    magic 0x4B400000 has its low 22 bits zero, so (magic + n) << 22 == (n & 0x3FF) << 22:
    n bit0 -> bit22 (top mantissa bit), n bits1..9 -> bits23..31 (exponent).  Adding 0x3F800000
    yields (1.0 or 1.5) * 2^{floor(n/2)}.  Mirrors utils.combine_int_frac_ex2 with shl 22 and a
    fixed fractional immediate."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x_rounded).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .s32 x_rounded_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 22;\n\t"
            "add.s32 out_i, x_rounded_e, 0x3F800000;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pot_h15_emulation_2(x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None):
    """H15 replacement for ex2_emulation_2.  Signature/return match ex2_emulation_2 (poly_degree is
    accepted and ignored, so callers need no change).  x, y are already DOUBLED log2 units.
    round-to-nearest-even (add.rn), NOT floor (add.rm): H15 rounds to the nearest half-octave."""
    xy_clamped = (cute.arch.fmax(x, CLAMP_LO), cute.arch.fmax(y, CLAMP_LO))
    xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (MAGIC, MAGIC), rnd="rn")
    return combine_h15(xy_rounded[0], loc=loc, ip=ip), combine_h15(xy_rounded[1], loc=loc, ip=ip)
