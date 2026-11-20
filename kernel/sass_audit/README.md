# Static instruction audit (Appendix B.5)

`exp_emu_regcount.cu` isolates the two exponential sequences as CUDA C++ device functions with the same
harness: FA4's emulation (`ex2_emulation_2` / `combine_int_frac_ex2` of FA4 commit `0dc2cb48`: clamp, magic-number
floor `add.rm`, fractional part, cubic `POLY_EX2[3]` by Horner, exponent combine) and Rowmax-H15 (clamp,
`add.rn` with the magic constant, `shl 22` + `add 0x3F800000`). Single-application kernels isolate the instruction
sequence; 32-element register-resident fragment kernels show register pressure.

`compile.sh` compiles for sm_86 with `-O3 -Xptxas -v` and disassembles the cubin. The paper's counts
(`results/kernel/sass_audit/`):

| | FA4 emulation | Rowmax-H15 |
|---|---|---|
| instructions per element | 8 (7 on the FP pipe: FMNMX, FADD.RM, FADD, FADD, 3 FFMA; 1 LEA) | 3 (FMNMX, FADD.RN; 1 LEA) |
| dependency depth | 8 | 3 |
| registers, single application | 14 | 12 |
| registers, 32-element fragment | 40 | 40 |

This concerns the operator in isolation, not the fused B200 kernel, and does not rule out register pressure or
another mechanism there.
