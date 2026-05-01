"""Statistical helpers for quality comparisons.

Wraps `bench.bootstrap_ci` for the (baseline, candidate) delta of a per-
sample score. Useful when lm-eval reports a stderr but we want a
non-parametric CI on the delta itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lmbench.bench import bootstrap_ci


@dataclass(frozen=True)
class DeltaCI:
    """Confidence interval for `mean(candidate) - mean(baseline)`."""

    mean_delta: float
    low: float
    high: float
    alpha: float
    significant: bool


def delta_bootstrap_ci(
    baseline: list[float],
    candidate: list[float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> DeltaCI:
    """Bootstrap CI for `mean(candidate) - mean(baseline)`.

    Bootstraps each sample independently and reports the conservative
    envelope around the delta. Significance = CI excludes 0.
    """
    if not baseline or not candidate:
        raise ValueError("delta_bootstrap_ci requires non-empty inputs")
    bl_low, bl_high = bootstrap_ci(
        baseline, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed
    )
    cand_low, cand_high = bootstrap_ci(
        candidate, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    bl_mean = sum(baseline) / len(baseline)
    cand_mean = sum(candidate) / len(candidate)
    mean_delta = cand_mean - bl_mean
    low = cand_low - bl_high
    high = cand_high - bl_low
    significant = (low > 0.0) or (high < 0.0)
    return DeltaCI(
        mean_delta=mean_delta,
        low=low,
        high=high,
        alpha=alpha,
        significant=significant,
    )


def is_within_tolerance(
    baseline: float,
    candidate: float,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> bool:
    """True if `candidate` is within `abs_tol` or `rel_tol` of `baseline`."""
    return math.isclose(baseline, candidate, abs_tol=abs_tol, rel_tol=rel_tol)
