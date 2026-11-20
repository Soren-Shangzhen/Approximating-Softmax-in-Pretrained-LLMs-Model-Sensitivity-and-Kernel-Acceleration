"""Statistics of the paper (App. A.9), without torch: paired block bootstrap and perplexity change.

The statistical unit is the aligned block.  A contrast is the per-block difference d_b; its estimate is the mean
over blocks and its interval the two-sided 95% percentile interval of 5,000 bootstrap means, each resampling the
blocks with replacement (numpy default_rng(seed=0) indices shared across conditions).  This reproduces
characterization.Bootstrapper.mean_ci exactly.  A perplexity change is 100 (exp(dNLL) - 1) percent.
"""
from __future__ import annotations

import math

import numpy as np

REPS = 5000
SEED = 0
_IDX: dict[tuple[int, int, int], np.ndarray] = {}


def indices(n: int, reps: int = REPS, seed: int = SEED) -> np.ndarray:
    key = (int(n), int(reps), int(seed))
    if key not in _IDX:
        _IDX[key] = np.random.default_rng(seed).integers(0, n, size=(reps, n), endpoint=False)
    return _IDX[key]


def mean_ci(x, reps: int = REPS, seed: int = SEED, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lower, upper) of the paired block bootstrap."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = x[indices(len(x), reps, seed)].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(x.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def resolved(lo: float, hi: float) -> bool:
    """A contrast is resolved when its interval excludes zero."""
    return bool(lo > 0 or hi < 0)


def ppl_change(dnll: float) -> float:
    """Perplexity change in percent for a NLL change in nats."""
    return 100.0 * math.expm1(dnll)
