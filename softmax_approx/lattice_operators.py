"""Lattice operators of Section 5 and Appendix B, evaluated through the same attention backend as Section 4.

Each operator maps the FP32 scores of a row (``scores``; ``valid`` marks the keys allowed by the mask) to
unnormalised non-negative weights; the backend normalises them over the row.  Base-two exponents are
sigma = s * log2(e).  The conditions of the paper (Tables B1 and B2, the exponential-map ablation) are:

Absolute anchors (Section 5.1, Table B1; Table B2 references)
  "ilpot"            octave lattice on an absolute scale: n = floor(sigma + 1/2), w = 2^(n - max n)
                     (condition ilpot_v1, "absolute-lattice ILPoT").
  "ilpot_ha"         half-octave lattice {1, sqrt2} x 2^k on an absolute scale whose maximum entry is set to
                     its exact weight (condition ilpot_ha_v1, "ILPoT-H-A").
  "ilpot_h15"        {1, 1.5} x 2^k lattice on an absolute scale (condition ilpot_h15_v1, Table B2
                     "Absolute-anchored {1,1.5} x 2^k lattice").
  "pq_log2_b{b}"     probability-domain grid: d = round(-log2 p) clamped to [0, 2^b - 1], w = 2^-d
                     (FQ-ViT's log2 structure; b = 8 in Table B1).
  "pq_logsqrt2_b{b}" probability-domain grid: d = round(-2 log2 p) clamped to [0, 2^b - 1], w = 2^(-d/2)
                     (RepQ-ViT's log-sqrt2 structure; b = 8 in Table B1).
  The rowmax-anchored counterparts PoT m=1 / m=2 are perturbations of the attention backend (pot_m1,
  response_pot_m2; attention.py mode "perturb").

Rowmax-H15 semantics (Section 5.2, Eq. 5 / Eq. 28; Table B2)
  "h15rowmax"        offline whole-row Rowmax-H15: x = sigma - max_row sigma, w = H(x) with the kernel's bit
                     construction (round-to-nearest-even of 2x, clamp at -252, 22-bit shift).
  "h15tiled"         FP8-path semantics of the patched FlashAttention-4 kernel: keys in tiles of 128, each
                     entry quantized relative to the running maximum of its own tile, tiles rescaled to the
                     final maximum by the exact factor 2^(m_t - m_final) (rescale_threshold = 0).
  "h15tiled_bf16"    BF16-path semantics: as h15tiled, but the running maximum is only updated when it grows
                     by more than 8 octaves (FA4's rescale_threshold = 8.0), decided per row.

Exponential maps (Appendix B.1-B.2, offline whole-row path)
  "schraudolph_rowmax"       unshifted Schraudolph map S(x) = 2^floor(x) (1 + x - floor(x)), x >= -126.
  "schraudolph_rowmax_k{k}"  k-bit staircase S(Q_{2^-k}(x)); k = 1 is Rowmax-H15, H(x) = S(Q_{1/2}(x)).

``install(attention_module)`` adds these modes to the backend's ``_alt_probs``; it must run before
``characterization.install_probe_patch`` (called by ``CrossModelRunner.load_runtime``) so that the
pipeline's probes see their output exactly as they see every other operator.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG2E = 1.4426950408889634          # 1 / ln 2
SQRT2 = 1.4142135623730951
E_MIN = -126                        # smallest base-two exponent kept by the absolute lattices
CLAMP_LO_H15 = -252.0               # Rowmax-H15 clamps its doubled input 2x at -252 (x >= -126)
CLAMP_LO_X = CLAMP_LO_H15 / 2.0     # the same lower bound expressed in x
H15_NBLOCK = 128                    # FA4 key tile at head dimension 128 (n_block_size)
RESCALE_THRESHOLD_BF16 = 8.0        # FA4 rescale_threshold for 16-bit dtypes (log2 units)
_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# absolute anchors
# ---------------------------------------------------------------------------
@torch.no_grad()
def _ilpot_exponents(scores: torch.Tensor, valid: torch.Tensor):
    """(e, M, sigma) on the absolute octave lattice; masked entries have e = -inf."""
    sigma = scores.float() * LOG2E
    sigma = sigma.masked_fill(~valid, float("-inf"))
    n = torch.floor(sigma + 0.5)                # -inf stays -inf
    M = n.amax(dim=-1, keepdim=True)
    e = n - M
    return e, M, sigma


@torch.no_grad()
def weights_absolute_octave(scores, valid):
    """"ilpot": w = 2^e, e = floor(sigma + 1/2) - max_j floor(sigma_j + 1/2) <= 0; 0 below 2^-126."""
    e, M, sigma = _ilpot_exponents(scores, valid)
    keep = valid & (e >= E_MIN)
    w = torch.exp2(e.clamp(min=float(E_MIN)))   # integer exponent -> exact power of two
    w = torch.where(keep, w, torch.zeros_like(w))
    return w


@torch.no_grad()
def _weights_absolute_half_octave(scores, valid):
    """Absolute half-octave lattice {1, sqrt2} x 2^q with an even maximum: n = floor(2 sigma + 1/2),
    M = 2 ceil(max n / 2), e = n - M <= 0, w = 2^floor(e/2) * (sqrt2 if e is odd)."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, float("-inf"))
    n = torch.floor(2.0 * sigma + 0.5)
    M = 2.0 * torch.ceil(n.amax(dim=-1, keepdim=True) / 2.0)
    e = n - M                                   # <= 0, integer-valued (half octaves)
    e_safe = torch.where(valid, e, torch.full_like(e, -300.0))
    ei = e_safe.to(torch.int64)
    q = torch.div(ei, 2, rounding_mode="floor")
    r = ei - 2 * q                              # 0 or 1
    keep = valid & (q >= E_MIN)
    w = torch.exp2(q.clamp(min=E_MIN).to(torch.float32))
    w = w * torch.where(r == 1, torch.full_like(w, SQRT2), torch.ones_like(w))
    w = torch.where(keep, w, torch.zeros_like(w))
    return w


def _first_argmax(x):
    return (x == x.amax(dim=-1, keepdim=True)).float().argmax(dim=-1, keepdim=True)


@torch.no_grad()
def weights_absolute_half_octave_exact_max(scores, valid):
    """"ilpot_ha" (ILPoT-H-A): the absolute half-octave lattice with the first maximal entry set to its exact
    weight f = 2^(sigma_max - M/2) relative to the lattice anchor."""
    w = _weights_absolute_half_octave(scores, valid)
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    smax = sigma.amax(dim=-1, keepdim=True)
    n_max = torch.floor(2.0 * smax + 0.5)
    M = 2.0 * torch.ceil(n_max / 2.0)
    idx = _first_argmax(sigma)
    f = torch.exp2(smax - M / 2.0)
    w = w.scatter(-1, idx, f)
    return w


@torch.no_grad()
def weights_absolute_h15(scores, valid):
    """"ilpot_h15": {1, 1.5} x 2^q on an absolute scale.  n = floor(2 sigma + 1/2), M = 2 ceil(max n / 2),
    e = n - M, w = (1 + (e mod 2)/2) 2^floor(e/2); 0 below 2^-126."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    n = torch.floor(2.0 * sigma + 0.5)
    M = 2.0 * torch.ceil(n.amax(dim=-1, keepdim=True) / 2.0)
    e = n - M
    ei = torch.where(valid, e, torch.full_like(e, -300.0)).to(torch.int64)
    q = torch.div(ei, 2, rounding_mode="floor")
    r = ei & 1                                   # two's complement: e = -1 -> 1
    keep = valid & (q >= E_MIN)
    w = (1.0 + 0.5 * r.to(torch.float32)) * torch.exp2(q.clamp(min=E_MIN).to(torch.float32))
    w = torch.where(keep, w, torch.zeros_like(w))
    return w


def _softmax_p(scores, valid):
    return F.softmax(scores.float().masked_fill(~valid, _NEG_INF), dim=-1)


@torch.no_grad()
def weights_prob_log2(scores, valid, bits: int):
    """"pq_log2_b{bits}": d = round(-log2 p) clamped to [0, 2^bits - 1], w = 2^-d (p = softmax)."""
    cap = float(2 ** int(bits) - 1)
    p = _softmax_p(scores, valid)
    raw = torch.round(-torch.log2(p))
    d = raw.clamp(0, cap)
    w = torch.exp2(-d)
    return torch.where(valid, w, torch.zeros_like(w))


@torch.no_grad()
def weights_prob_logsqrt2(scores, valid, bits: int):
    """"pq_logsqrt2_b{bits}": d = round(-2 log2 p) clamped to [0, 2^bits - 1], w = 2^(-d/2) (p = softmax)."""
    cap = float(2 ** int(bits) - 1)
    p = _softmax_p(scores, valid)
    raw = torch.round(-2.0 * torch.log2(p))
    d = raw.clamp(0, cap)
    w = torch.exp2(-d / 2.0)
    return torch.where(valid, w, torch.zeros_like(w))


# ---------------------------------------------------------------------------
# Rowmax-H15 (bit-level construction of Appendix B.1, kernel semantics of Appendix B.2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def h15_bits(x):
    """Rowmax-H15 of a doubled exponent x = 2 (sigma - m): n = round_RNE(max(x, -252)) and the FP32 bit pattern
    ((n & 0x3FF) << 22) + 0x3F800000, i.e. (1 + (n mod 2)/2) 2^floor(n/2) = w_H15 of Eq. (5)."""
    xc = torch.clamp(x, min=CLAMP_LO_H15)
    n = torch.round(xc).to(torch.int32)                 # ties to even (the kernel's add.rn)
    bits = ((n & 0x3FF) << 22) + 0x3F800000
    return bits.view(torch.float32)


@torch.no_grad()
def weights_rowmax_h15(scores, valid, max_offset=0):
    """"h15rowmax": every entry quantized relative to the final (whole-row) maximum."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    smax = sigma.amax(dim=-1, keepdim=True)
    x = 2.0 * (sigma - smax) + 2.0 * float(max_offset)
    x = torch.where(valid, x, torch.full_like(x, _NEG_INF))
    w = torch.where(valid, h15_bits(torch.where(valid, x, torch.zeros_like(x))), torch.zeros_like(x))
    return w


@torch.no_grad()
def weights_rowmax_h15_tiled(scores, valid, n_block=H15_NBLOCK, max_offset=0):
    """"h15tiled": FA4 online semantics of the FP8 path (rescale_threshold = 0, the maximum is updated at every
    tile).  Each entry is quantized relative to the running maximum m_t of its own key tile (contiguous blocks
    of n_block keys; causality enters through ``valid``) and then rescaled to the final maximum by the exact
    factor 2^(m_t - m_final), not re-quantized.  max_offset cancels in the normalisation."""
    b, h, q, kv = scores.shape
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    ntiles = (kv + n_block - 1) // n_block
    pad = ntiles * n_block - kv
    sp = F.pad(sigma, (0, pad), value=_NEG_INF) if pad else sigma
    tile_max = sp.reshape(b, h, q, ntiles, n_block).amax(dim=-1)       # [b,h,q,ntiles]
    run_max = torch.cummax(tile_max, dim=-1).values                    # running max up to tile t
    m_final = run_max[..., -1:]                                        # [b,h,q,1]
    tile_of = (torch.arange(kv, device=scores.device) // n_block)      # [kv]
    mj = run_max.index_select(-1, tile_of)                             # running max of each entry's tile
    x = 2.0 * (sigma - mj) + 2.0 * float(max_offset)
    xw = torch.where(valid, x, torch.zeros_like(x))
    w = h15_bits(xw)
    # exact rescale of each tile to the final maximum (kernel: O, l *= exp2(m - m_new))
    rescale = torch.exp2(torch.where(torch.isfinite(mj) & torch.isfinite(m_final), mj - m_final,
                                     torch.zeros_like(mj)))
    w = w * rescale
    w = torch.where(valid, w, torch.zeros_like(w))
    return w


@torch.no_grad()
def weights_rowmax_h15_tiled_bf16(scores, valid, n_block=H15_NBLOCK, max_offset=0,
                                  threshold=RESCALE_THRESHOLD_BF16):
    """"h15tiled_bf16": as weights_rowmax_h15_tiled, but with FA4's update rule for 16-bit dtypes: the running
    maximum keeps its old value while it grows by at most ``threshold`` octaves, so entries above it get x > 0
    and the row's largest weight can leave the lattice (Appendix B.2, "Conditional rescaling moves the anchor").
    The decision is made per row; FA4 makes it per warp of 32 rows."""
    b, h, q, kv = scores.shape
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    ntiles = (kv + n_block - 1) // n_block
    pad = ntiles * n_block - kv
    sp = F.pad(sigma, (0, pad), value=_NEG_INF) if pad else sigma
    tile_max = sp.reshape(b, h, q, ntiles, n_block).amax(dim=-1)        # [b,h,q,ntiles]

    anchors = []
    m = torch.full_like(tile_max[..., 0], _NEG_INF)                     # [b,h,q]
    for t in range(ntiles):
        cand = torch.maximum(m, tile_max[..., t])
        first = ~torch.isfinite(m)                                      # m == -inf: must adopt
        grow = cand - torch.where(first, cand, m)                       # 0 when first (avoids nan)
        update = first | (grow > threshold)
        m = torch.where(update & torch.isfinite(cand), cand, m)
        anchors.append(m)
    run_max = torch.stack(anchors, dim=-1)                              # [b,h,q,ntiles]

    m_final = run_max[..., -1:]
    tile_of = (torch.arange(kv, device=scores.device) // n_block)
    mj = run_max.index_select(-1, tile_of)                              # [b,h,q,kv]
    x = 2.0 * (sigma - mj) + 2.0 * float(max_offset)
    xw = torch.where(valid, x, torch.zeros_like(x))
    w = h15_bits(xw)
    rescale = torch.exp2(torch.where(torch.isfinite(mj) & torch.isfinite(m_final), mj - m_final,
                                     torch.zeros_like(mj)))
    w = w * rescale
    return torch.where(valid, w, torch.zeros_like(w))


# ---------------------------------------------------------------------------
# exponential maps (offline whole-row path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def S(x: torch.Tensor) -> torch.Tensor:
    """Unshifted Schraudolph map: 2^floor(x) (1 + x - floor(x))."""
    fl = torch.floor(x)
    return torch.exp2(fl) * (1.0 + (x - fl))


@torch.no_grad()
def Q_half(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest half-integer, ties to even."""
    return torch.round(2.0 * x) / 2.0


@torch.no_grad()
def H(x: torch.Tensor) -> torch.Tensor:
    """Ideal Rowmax-H15 operator H(x) = S(Q_{1/2}(x))."""
    return S(Q_half(x))


@torch.no_grad()
def Q_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Round to the nearest multiple of 2^-k, ties to even (power-of-two scaling is exact in FP32)."""
    return torch.round(x * float(2 ** k)) / float(2 ** k)


@torch.no_grad()
def S_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-bit staircase S(Q_{2^-k}(x))."""
    return S(Q_k(x, k))


@torch.no_grad()
def weights_schraudolph_rowmax(scores, valid):
    """"schraudolph_rowmax": x = sigma - max_row sigma clamped at -126 (as Rowmax-H15), w = S(x)."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    smax = sigma.amax(dim=-1, keepdim=True)
    x = sigma - smax
    xw = torch.where(valid, x, torch.zeros_like(x))
    w = S(torch.clamp(xw, min=CLAMP_LO_X))
    return torch.where(valid, w, torch.zeros_like(w))


@torch.no_grad()
def weights_schraudolph_rowmax_k(scores, valid, k: int):
    """"schraudolph_rowmax_k{k}": as weights_schraudolph_rowmax with the map S_k."""
    sigma = (scores.float() * LOG2E).masked_fill(~valid, _NEG_INF)
    smax = sigma.amax(dim=-1, keepdim=True)
    x = sigma - smax
    xw = torch.where(valid, x, torch.zeros_like(x))
    w = S_k(torch.clamp(xw, min=CLAMP_LO_X), k)
    return torch.where(valid, w, torch.zeros_like(w))


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
_FIXED = {
    "ilpot": weights_absolute_octave,
    "ilpot_ha": weights_absolute_half_octave_exact_max,
    "ilpot_h15": weights_absolute_h15,
    "h15rowmax": weights_rowmax_h15,
    "h15tiled": weights_rowmax_h15_tiled,
    "h15tiled_bf16": weights_rowmax_h15_tiled_bf16,
    "schraudolph_rowmax": weights_schraudolph_rowmax,
}
_PARAMETRIC = (
    ("pq_logsqrt2_b", weights_prob_logsqrt2),
    ("pq_log2_b", weights_prob_log2),
    ("schraudolph_rowmax_k", weights_schraudolph_rowmax_k),
)


def weight_fn(mode):
    """Weight function of a mode name, or None if the mode is not a lattice operator."""
    if not isinstance(mode, str):
        return None
    if mode in _FIXED:
        return _FIXED[mode]
    for prefix, fn in _PARAMETRIC:
        if mode.startswith(prefix) and mode[len(prefix):].isdigit():
            arg = int(mode[len(prefix):])
            return lambda scores, valid, _fn=fn, _arg=arg: _fn(scores, valid, _arg)
    return None


@torch.no_grad()
def probs(scores, valid, mode):
    """Normalised probabilities and kept-entry mask of a lattice operator."""
    w = weight_fn(mode)(scores, valid)
    return w / w.sum(dim=-1, keepdim=True), (w > 0)


_INSTALLED = {}


def install(attention_module) -> None:
    """Wrap attention_module._alt_probs with the lattice modes (idempotent)."""
    if _INSTALLED.get("module") is attention_module:
        return
    inner = attention_module._alt_probs

    def alt_probs_with_lattices(scores, valid, mode, q_len, layer_idx=-1, q_start=0, offset_full=0):
        if weight_fn(mode) is not None:
            return probs(scores, valid, mode)
        return inner(scores, valid, mode, q_len, layer_idx, q_start, offset_full)

    attention_module._alt_probs = alt_probs_with_lattices
    _INSTALLED["module"] = attention_module
