"""Attention backend for frozen-model interventions on the softmax map (Sections 3-4, Appendix A).

``attention_forward`` follows the signature of Transformers' ``eager_attention_forward`` and is registered
as the attention implementation ``ATTN_IMPLEMENTATION`` by ``softmax_approx.model_loader``.  For every head
and layer it computes the scores q.k^T in FP32 with the model's own scaling, attention-logit softcap and
causal mask, and replaces only the score-to-probability map by the operator selected for that layer in
``CFG.layer_mode``.  The probabilities are cast to the value dtype before the PV product.  The softmax
baseline runs through the same code path as every intervention.

Operators (``CFG.layer_mode[layer]``; set per condition by ``characterization.configure_condition``):

  "softmax"      softmax over the valid keys (baseline, Eq. 1).
  "binary_soft"  mean-threshold support (scores above the row mean of the valid scores), softmax weights
                 renormalised on that support (App. A.3, Eq. 10).
  "binary"       the same support with uniform weights.
  "topk_soft"    top-k support, k = ceil(r |V|) with r = CFG.topk_ratio, softmax weights.
  "topk"         top-k support with uniform weights.
  "multibit"     full-range grid (App. A.4-A.5): [s_min, s_max] of each row is divided into K = CFG.n_levels
                 intervals whose widths shrink geometrically towards the row maximum with ratio
                 R = CFG.width_ratio (Eq. 13; R = 1 gives equal widths).  The interval index maps to a weight
                 through an exponential ("exp") or linear ("linear") level-to-weight map, with upper-edge
                 ("up"), nearest-boundary ("nearest") or weight-domain interpolation ("lerp") reconstruction
                 (CFG.quant_round, Eq. 17).
  "perturb"      perturbations of the softmax log weights (App. A.6-A.7): temperature scaling of the scores
                 (CFG.pert_arm = "temp", alpha = CFG.alpha, Eq. 22) or rounding of the rowmax-relative base-two
                 log weight to m steps per octave without a tail cutoff (CFG.pert_arm = "pot", m = CFG.pert_m,
                 Eq. 18).
  "pow2_max"     Rowmax-PoT (Eq. 20): weights 2^-d with d = min(K_max, round(Delta / ln 2)) measured from the row
                 maximum; K_max = CFG.n_levels, tail entries clamped to 2^-K_max or dropped (CFG.tail_mode).

The lattice operators of Section 5 and Appendix B (absolute versus row-maximum anchors, Rowmax-H15 semantics,
exponential maps) are added by ``softmax_approx.lattice_operators.install``, which wraps ``_alt_probs``.
``characterization.install_probe_patch`` wraps ``_alt_probs``, ``_attend`` and ``_log_sparsity`` to collect
divergence, error-geometry, support and high-mass statistics; both look these functions up as module
attributes at call time.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

SUPPORTS_ATTN_LOGIT_SOFTCAPPING = True


@dataclass
class AttentionConfig:
    layer_mode: dict = field(default_factory=dict)          # {layer_idx: operator}; layers not listed use default_mode
    default_mode: str = "softmax"
    # full-range grid ("multibit") and Rowmax-PoT ("pow2_max")
    n_levels: int = 1                                       # K (grid intervals) or K_max (Rowmax-PoT tail index)
    n_levels_per_layer: dict = field(default_factory=dict)  # optional per-layer override of n_levels (unused: empty)
    threshold_mode: str = "none"                            # grids keep every valid entry (no support cut)
    coverage: str = "max"                                   # grids span [s_min, s_max] of each row
    width_ratio: float = 1.0                                # R = h_1 / h_K (lowest-score / top interval width)
    level_weight: str = "exp"                               # "exp" | "linear" level-to-weight map
    quant_round: str = "up"                                 # reconstruction: "up" | "nearest" | "lerp" (exp map only)
    use_std: bool = True                                    # set by configure_condition; grids use no row std
    round_mode: str = "floor"                               # Rowmax-PoT exponent rounding: "round" (= floor(x+1/2)) | "floor"
    fine_m: int = 1                                         # Rowmax-PoT: m steps per octave below fine_T octaves
    fine_T: float = 0.0
    tail_mode: str = "clamp"                                # Rowmax-PoT tail beyond K_max: "clamp" | "drop"
    alpha: float = 1.0                                      # temperature (perturb/temp); Rowmax-PoT octave = ln2/alpha
    # log-weight perturbations ("perturb")
    pert_arm: str = "temp"                                  # "temp" | "pot"
    pert_m: float = 1.0                                     # PoT steps per octave
    pert_c: float = 0.0                                     # set by configure_condition; not used by "temp"/"pot"
    pert_seed: int = 0                                      # set by configure_condition; not used by "temp"/"pot"
    # support interventions
    topk_ratio: float | None = None
    # execution
    probe_sparsity: bool = False                            # compute the operator but return softmax (T00 identity probe)
    fp32_scores: bool = True                                # q.k^T in FP32 (the paper's protocol)
    q_chunk: int = 0                                        # >0: process queries in chunks (row statistics stay within a row)
    stats: dict = field(default_factory=dict)               # support density per layer (see _log_sparsity)

    def reset(self):
        fresh = AttentionConfig()
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(fresh, f.name))


CFG = AttentionConfig()

UNIFORM_MODES = {"binary", "topk"}
SOFT_MODES = {"binary_soft", "topk_soft"}
THRESHOLD_SEL = {"binary", "binary_soft"}
TOPK_SEL = {"topk", "topk_soft"}


def reset_stats():
    CFG.stats = {}


def _log_sparsity(layer_idx, M, valid):
    """Mean fraction of valid keys kept per row (replaced by the characterization pipeline's accumulator)."""
    kept = M.sum(-1).float()
    total = valid.sum(-1).float().clamp(min=1)
    s = (kept / total).mean().item()
    acc = CFG.stats.setdefault(layer_idx, [0.0, 0])
    acc[0] += s
    acc[1] += 1


def sparsity_report():
    return {k: v[0] / max(v[1], 1) for k, v in CFG.stats.items()}


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def attention_forward(module, query, key, value, attention_mask,
                      scaling=None, dropout=0.0, **kwargs):
    """Signature of transformers' eager_attention_forward; returns (output [b, q, h, d], None)."""
    layer_idx = getattr(module, "layer_idx", -1)
    mode = CFG.layer_mode.get(layer_idx, CFG.default_mode)

    n_rep = getattr(module, "num_key_value_groups", None)
    if n_rep is None:
        n_rep = query.shape[1] // key.shape[1]
    k = repeat_kv(key, n_rep)
    v = repeat_kv(value, n_rep)

    if scaling is None:
        scaling = getattr(module, "scaling", None) or (query.shape[-1] ** -0.5)

    # Gemma-2 and compatible architectures soft-cap the raw attention logits.  The cap is applied in
    # _attend BEFORE the mask is added; applying tanh to a large negative mask would make masked keys finite.
    if "attn_logit_softcapping" not in kwargs:
        softcap = getattr(module, "attn_logit_softcapping", None)
        if softcap is None:
            cfg = getattr(module, "config", None)
            softcap = getattr(cfg, "attn_logit_softcapping", None) if cfg is not None else None
        if softcap is None and "softcap" in kwargs:
            softcap = kwargs.get("softcap")
        if softcap is not None:
            kwargs["attn_logit_softcapping"] = float(softcap)

    q_len, kv_len = query.shape[-2], k.shape[-2]
    qc = int(CFG.q_chunk or 0)

    if qc <= 0 or q_len <= qc:
        out = _attend(query, k, v, attention_mask, scaling, mode, layer_idx,
                      q_start=0, q_len_full=q_len, kv_len=kv_len, **kwargs)
    else:
        # Query chunking: every row statistic depends on that row only, so chunking is exact.
        parts = []
        for s in range(0, q_len, qc):
            e = min(s + qc, q_len)
            am = None if attention_mask is None else attention_mask[..., s:e, :]
            parts.append(_attend(query[..., s:e, :], k, v, am, scaling, mode, layer_idx,
                                 q_start=s, q_len_full=q_len, kv_len=kv_len, **kwargs))
        out = torch.cat(parts, dim=2)

    return out.transpose(1, 2).contiguous(), None


def _attend(q, k, v, attention_mask, scaling, mode, layer_idx,
            q_start, q_len_full, kv_len, **kwargs):
    """One query chunk; returns [b, h, q_chunk, d]."""
    if CFG.fp32_scores:
        scores = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling
    else:
        scores = (torch.matmul(q, k.transpose(2, 3)) * scaling).float()

    softcap = kwargs.get("attn_logit_softcapping", kwargs.get("softcap", None))
    if softcap is not None:
        softcap = float(softcap)
        if softcap <= 0:
            raise ValueError(f"attn_logit_softcapping must be positive, got {softcap}")
        scores = torch.tanh(scores / softcap) * softcap

    qb = q.shape[-2]
    offset_full = kv_len - q_len_full          # 0 without a KV cache

    if attention_mask is not None:
        am = attention_mask[..., :kv_len]
        if am.dtype == torch.bool:
            valid = am.expand_as(scores)
            scores = scores.masked_fill(~valid, float("-inf"))
        else:
            valid = (am > -1e30).expand_as(scores)
            scores = scores + am.float()
    elif q_len_full > 1:
        # Transformers builds no mask for a custom attention without a registered mask interface;
        # fall back to a causal mask (identical to the registered eager mask without padding).
        sw = kwargs.get("sliding_window", None)
        if sw is not None and sw < kv_len:
            raise RuntimeError(f"sliding_window={sw} < kv_len={kv_len}: sliding-window masking is not implemented")
        gq = torch.arange(q_start, q_start + qb, device=scores.device).view(-1, 1)
        ki = torch.arange(kv_len, device=scores.device).view(1, -1)
        valid = (ki <= gq + offset_full).view(1, 1, qb, kv_len).expand_as(scores)
        scores = scores.masked_fill(~valid, float("-inf"))
    else:
        valid = torch.ones_like(scores, dtype=torch.bool)

    if mode == "softmax" and not CFG.probe_sparsity:
        probs = F.softmax(scores, dim=-1)
    else:
        alt_probs, M = _alt_probs(scores, valid, mode, qb, layer_idx,
                                  q_start=q_start, offset_full=offset_full)
        if M is not None:
            _log_sparsity(layer_idx, M, valid)
        probs = F.softmax(scores, dim=-1) if CFG.probe_sparsity else alt_probs
        del alt_probs

    del scores, valid
    return torch.matmul(probs.to(v.dtype), v)


def _alt_probs(scores, valid, mode, q_len, layer_idx=-1, q_start=0, offset_full=0):
    """Probabilities of an intervention; returns (p, M) with M the kept support (or the valid mask)."""
    kv_len = scores.shape[-1]

    if mode == "perturb":
        # Perturbation of the softmax log weights: with D = s_max - s >= 0, softmax is softmax(-D);
        # the perturbed weights are exp(-D + eps) on the valid entries only (Eqs. 18 and 22).
        rmx = scores.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
        D = (rmx - scores).clamp(min=0.0)
        arm = CFG.pert_arm
        if arm == "temp":
            # softmax(alpha s) = softmax(-alpha D):  eps = (1 - alpha) D
            eps = (1.0 - float(CFG.alpha)) * D
        elif arm == "pot":
            # w_hat = 2^{-round(m D / ln2) / m}:  eps = D - round(m D / ln2) ln2 / m,  |eps| <= ln2 / (2m)
            m = float(CFG.pert_m)
            ln2 = math.log(2.0)
            q = torch.round(D * m / ln2)
            eps = D - q * ln2 / m
        else:
            raise ValueError(f"unknown pert_arm: {arm}")
        eps = torch.where(valid, eps, torch.zeros_like(eps))
        neg = torch.full_like(D, float("-inf"))
        lwh = torch.where(valid, -D + eps, neg)
        ph = F.softmax(lwh, dim=-1)
        return ph, valid

    if mode == "pow2_max":
        # Rowmax-PoT: base-two weights anchored at the row maximum.  The tail clamp acts far below the
        # maximum, where the weights are small; only the row maximum is reduced, as in softmax.
        k = max(int(CFG.n_levels_per_layer.get(layer_idx, CFG.n_levels)), 1)
        rmx = scores.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
        delta = math.log(2.0) / max(float(CFG.alpha), 1e-9)       # one octave in score units
        x = ((rmx - scores) / delta).clamp(min=0.0)               # octaves below the row maximum
        m = max(int(CFG.fine_m), 1)
        rnd = (lambda t: torch.floor(t + 0.5)) if CFG.round_mode == "round" else torch.floor
        if m > 1 and CFG.fine_T > 0:
            xr = torch.where(x < float(CFG.fine_T), rnd(x * m) / m, rnd(x))
        else:
            xr = rnd(x)
        over = xr > float(k)                                      # tail beyond K_max octaves
        d = xr.clamp(max=float(k))
        w = torch.exp2(-d)
        if CFG.tail_mode == "drop":
            w = torch.where(over, torch.zeros_like(w), w)
        w = torch.where(valid, w, torch.zeros_like(w))
        M = w > 0
        # the row maximum has d = 0 and w = 1, so no row is empty
        return w / w.sum(-1, keepdim=True).clamp(min=1e-12), M

    if mode == "multibit":
        if CFG.threshold_mode != "none" or CFG.coverage != "max":
            raise ValueError("only full-range grids (threshold_mode='none', coverage='max') are implemented")
        k = max(int(CFG.n_levels_per_layer.get(layer_idx, CFG.n_levels)), 1)
        # Full range [s_min, s_max]: u = s - s_min in [0, C], C = s_max - s_min (Eq. 12).
        mu = scores.masked_fill(~valid, float("inf")).amin(-1, keepdim=True)
        rmx = scores.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
        Cov = (rmx - mu).clamp(min=1e-6)
        step = Cov / k
        above = valid                                             # no support cut: every valid entry is kept
        u = (scores - mu).clamp(min=0.0)

        R = float(CFG.width_ratio)
        _qr_ok = (CFG.level_weight == "exp" and mode == "multibit")
        NEAREST = (CFG.quant_round == "nearest" and _qr_ok)
        LERP = (CFG.quant_round == "lerp" and _qr_ok)
        w_direct = None                                           # LERP gives the weights directly
        if NEAREST or LERP:
            # Locate u between the boundaries e_lo <= u <= e_hi of its interval.  Nearest-boundary
            # reconstruction compares distances in u (= relative log-weight error); e_0 = 0 is a valid
            # reconstruction point for u near 0.
            if abs(R - 1.0) < 1e-9:
                stp = step
                lcont = u / stp.clamp(min=1e-12)
                ehi_fn = lambda lv: lv * stp
            else:
                rho = R ** (-1.0 / max(k - 1, 1))
                den = 1.0 - rho ** k
                frac = (u / Cov.clamp(min=1e-12)).clamp(min=0.0, max=1.0)
                arg = (1.0 - frac * den).clamp(min=1e-12)
                lcont = torch.log(arg) / math.log(rho)
                ehi_fn = lambda lv: Cov * (1.0 - rho ** lv) / den
            l_hi = torch.ceil(lcont).clamp(1.0, float(k))
            l_lo = (l_hi - 1.0).clamp(0.0, float(k))
            e_hi, e_lo = ehi_fn(l_hi), ehi_fn(l_lo)
            if NEAREST:
                near = torch.where((u - e_lo) <= (e_hi - u), l_lo, l_hi)   # ties go to the lower boundary
                level = torch.where(above, near, torch.zeros_like(near))
                edge = torch.where(above, ehi_fn(level), torch.zeros_like(level))
            else:
                # LERP of the WEIGHTS at the two boundaries (Eq. 17): w = (1-t) exp(e_lo) + t exp(e_hi),
                # t = (u - e_lo) / (e_hi - e_lo); normalised by the row maximum of exp(e_hi) to stay <= 1.
                t = ((u - e_lo) / (e_hi - e_lo).clamp(min=1e-12)).clamp(0.0, 1.0)
                em = torch.where(above, e_hi, torch.full_like(e_hi, float("-inf")))
                em = em.amax(dim=-1, keepdim=True)
                em = torch.where(torch.isfinite(em), em, torch.zeros_like(em))
                wl = (1.0 - t) * torch.exp(e_lo - em) + t * torch.exp(e_hi - em)
                w_direct = torch.where(above, wl, torch.zeros_like(wl))
                level = torch.where(above, l_hi, torch.zeros_like(l_hi))
                edge = torch.where(above, e_hi, torch.zeros_like(e_hi))
        elif abs(R - 1.0) < 1e-9:
            # equal widths, upper-edge reconstruction
            raw = u / step.clamp(min=1e-12)
            level = torch.where(above, raw.floor() + 1.0, torch.zeros_like(raw))
            level = level.clamp(min=0.0, max=float(k))
            edge = level * step
        else:
            # geometric widths h_a = h_1 rho^{a-1}, rho = R^{-1/(K-1)}; boundaries e_a = C (1-rho^a)/(1-rho^K)
            # (Eq. 13).  Upper-edge index: smallest a with e_a >= u.
            rho = R ** (-1.0 / max(k - 1, 1))
            den = 1.0 - rho ** k
            frac = (u / Cov.clamp(min=1e-12)).clamp(min=0.0, max=1.0)
            arg = (1.0 - frac * den).clamp(min=1e-12)
            lv = torch.ceil(torch.log(arg) / math.log(rho))
            level = torch.where(above, lv.clamp(1.0, float(k)), torch.zeros_like(lv))
            edge = Cov * (1.0 - rho ** level) / den
            edge = torch.where(above, edge, torch.zeros_like(edge))

        if w_direct is not None:
            w = w_direct
        elif CFG.level_weight in ("exp", "linear") and (NEAREST or abs(float(CFG.width_ratio) - 1.0) >= 1e-9):
            # non-uniform widths or nearest reconstruction: the weight comes from the boundary value
            if CFG.level_weight == "linear":
                w = level
            else:
                e = torch.where(above, edge, torch.full_like(edge, float("-inf")))
                emax = e.amax(dim=-1, keepdim=True)
                emax = torch.where(torch.isfinite(emax), emax, torch.zeros_like(emax))
                w = torch.where(above, torch.exp(e - emax), torch.zeros_like(e))
        elif CFG.level_weight == "linear":
            w = level
        elif CFG.level_weight == "exp":
            # w = exp(level * step), shifted by the row maximum before exp (cancels on normalisation)
            e = level * step
            e = torch.where(above, e, torch.full_like(e, float("-inf")))
            emax = e.amax(dim=-1, keepdim=True)
            emax = torch.where(torch.isfinite(emax), emax, torch.zeros_like(emax))
            w = torch.where(above, torch.exp(e - emax), torch.zeros_like(e))
        else:
            raise ValueError(f"unknown level_weight: {CFG.level_weight}")
        w = w * valid.float()

        M = w > 0
        empty = w.sum(-1, keepdim=True) == 0
        if bool(empty.any()):
            am = scores.masked_fill(~valid, float("-inf")).argmax(-1)
            oh = F.one_hot(am, kv_len).float()
            w = torch.where(empty, w + oh, w)
            M = M | (empty & oh.bool())
        return w / w.sum(-1, keepdim=True).clamp(min=1e-9), M

    # ---------- support interventions: step 1, select the support M ----------
    if mode in TOPK_SEL:
        assert CFG.topk_ratio is not None, "top-k modes need CFG.topk_ratio"
        cnt = valid.sum(-1, keepdim=True).clamp(min=1)
        kk = (cnt.float() * CFG.topk_ratio).ceil().long().clamp(min=1)
        s_fill = scores.masked_fill(~valid, float("-inf"))
        rank = s_fill.argsort(dim=-1, descending=True).argsort(dim=-1)
        M = (rank < kk) & valid
    elif mode in THRESHOLD_SEL:
        cnt = valid.sum(-1, keepdim=True).clamp(min=1).float()
        s0 = torch.where(valid, scores, torch.zeros_like(scores))
        mu = s0.sum(-1, keepdim=True) / cnt                       # row mean of the valid scores
        M = (scores > mu) & valid
    elif mode == "softmax":
        raise ValueError(
            "probe_sparsity=True needs a layer_mode that produces a support (multibit / binary / binary_soft / "
            "topk / topk_soft); the current mode is softmax.")
    else:
        raise ValueError(f"unknown mode: {mode}")

    # a row whose support would be empty keeps its maximum
    empty = M.sum(-1, keepdim=True) == 0
    if bool(empty.any()):
        am = scores.masked_fill(~valid, float("-inf")).argmax(-1)
        M = M | (empty & F.one_hot(am, kv_len).bool())

    # ---------- step 2: weights on the support ----------
    if mode in SOFT_MODES:
        # same support, softmax weights renormalised on it
        probs = F.softmax(scores.masked_fill(~M, float("-inf")), dim=-1)
    else:
        probs = M.float() / M.sum(-1, keepdim=True).clamp(min=1).float()

    return probs, M
