// Static instruction audit (Appendix B.5): FA4's polynomial exp2 emulation vs the Rowmax-H15 sequence,
// as isolated sm_86 micro-kernels (not the fused B200 kernel).
// Faithful to flash_attn/cute/utils.py (commit 0dc2cb48): ex2_emulation_2 / e2e_asm2 magic-number
// path + POLY_EX2[3] cubic; H15 = add.rn + shl 22 + add.s32 0x3F800000.
// Coefficients (verified): p0=0x3F800000 p1=0x3F31F519 p2=0x3E6906A4 p3=0x3D9DF09D ; magic=0x4B400000.
// Inputs x are already in log2 units (row-relative), matching apply_exp2_convert.
#include <cstdint>

__device__ __forceinline__ float bitf(uint32_t b){ float f; asm("mov.b32 %0,%1;":"=f"(f):"r"(b)); return f; }

#define MAGIC   0x4B400000u
#define P0      0x3F800000u
#define P1      0x3F31F519u
#define P2      0x3E6906A4u
#define P3      0x3D9DF09Du

// ---- FA4 emulation, 2 elements (mirrors ex2_emulation_2 / combine_int_frac_ex2) ----
__device__ __forceinline__ void fa4_ex2_emu2(float& x, float& y){
    const float magic = bitf(MAGIC);
    const float p0 = bitf(P0), p1 = bitf(P1), p2 = bitf(P2), p3 = bitf(P3);
    float xc = fmaxf(x, -127.0f), yc = fmaxf(y, -127.0f);
    float xr, yr;
    asm("add.rm.ftz.f32 %0, %1, %2;" : "=f"(xr) : "f"(xc), "f"(magic));   // round-down floor
    asm("add.rm.ftz.f32 %0, %1, %2;" : "=f"(yr) : "f"(yc), "f"(magic));
    float xrb = xr - magic, yrb = yr - magic;
    float xf = xc - xrb, yf = yc - yrb;
    float px = fmaf(fmaf(fmaf(p3, xf, p2), xf, p1), xf, p0);              // Horner, 3 FMA
    float py = fmaf(fmaf(fmaf(p3, yf, p2), yf, p1), yf, p0);
    float ox, oy;
    asm("{ .reg .s32 a,b,c; mov.b32 a,%1; shl.b32 a,a,23; mov.b32 b,%2; add.s32 c,a,b; mov.b32 %0,c; }"
        : "=f"(ox) : "f"(xr), "f"(px));
    asm("{ .reg .s32 a,b,c; mov.b32 a,%1; shl.b32 a,a,23; mov.b32 b,%2; add.s32 c,a,b; mov.b32 %0,c; }"
        : "=f"(oy) : "f"(yr), "f"(py));
    x = ox; y = oy;
}

// ---- Rowmax-H15, 2 elements (x already doubled; the clamp constant does not change the instruction count) ----
__device__ __forceinline__ void h15_emu2(float& x, float& y){
    const float magic = bitf(MAGIC);
    float xc = fmaxf(x, -254.0f), yc = fmaxf(y, -254.0f);
    float xr, yr;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(xr) : "f"(xc), "f"(magic));   // round-nearest-even
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(yr) : "f"(yc), "f"(magic));
    float ox, oy;
    asm("{ .reg .s32 a; mov.b32 a,%1; shl.b32 a,a,22; add.s32 a,a,0x3F800000; mov.b32 %0,a; }"
        : "=f"(ox) : "f"(xr));
    asm("{ .reg .s32 a; mov.b32 a,%1; shl.b32 a,a,22; add.s32 a,a,0x3F800000; mov.b32 %0,a; }"
        : "=f"(oy) : "f"(yr));
    x = ox; y = oy;
}

// ---- single-application kernels (isolate the instruction sequence for nvdisasm) ----
extern "C" __global__ void k_fa4_single(const float* in, float* out){
    int i = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    float x = in[i], y = in[i+1];
    fa4_ex2_emu2(x, y);
    out[i] = x; out[i+1] = y;
}
extern "C" __global__ void k_h15_single(const float* in, float* out){
    int i = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    float x = in[i], y = in[i+1];
    h15_emu2(x, y);
    out[i] = x; out[i+1] = y;
}

// ---- 32-element register-resident fragment kernels (measure register pressure) ----
#define FRG 32
extern "C" __global__ void k_fa4_frag(const float* in, float* out, int reps){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    float f[FRG];
    #pragma unroll
    for(int k=0;k<FRG;k++) f[k] = in[t*FRG+k];
    for(int r=0;r<reps;r++){
        #pragma unroll
        for(int k=0;k<FRG;k+=2) fa4_ex2_emu2(f[k], f[k+1]);
    }
    #pragma unroll
    for(int k=0;k<FRG;k++) out[t*FRG+k] = f[k];
}
extern "C" __global__ void k_h15_frag(const float* in, float* out, int reps){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    float f[FRG];
    #pragma unroll
    for(int k=0;k<FRG;k++) f[k] = in[t*FRG+k];
    for(int r=0;r<reps;r++){
        #pragma unroll
        for(int k=0;k<FRG;k+=2) h15_emu2(f[k], f[k+1]);
    }
    #pragma unroll
    for(int k=0;k<FRG;k++) out[t*FRG+k] = f[k];
}
