"""Performance metric primitives.

Pure functions over per-request samples. The benchmark driver in `perf.py`
collects a list of `RequestSample` rows and feeds them to `summarize` to
produce a `PerfSummary`. The compare/report layers later consume those
summaries.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestSample:
    """One request's measurements."""

    ttft_s: float
    """Time-to-first-token (seconds). Wall time from request send to first token."""

    itl_s: tuple[float, ...]
    """Inter-token latencies (seconds), one per token after the first."""

    e2e_s: float
    """End-to-end wall time (seconds), request send to last token."""

    output_tokens: int
    """Number of output tokens (excludes prompt)."""

    success: bool = True


@dataclass(frozen=True)
class LatencyStats:
    """Mean + percentile bundle for a latency distribution."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float


@dataclass(frozen=True)
class PerfSummary:
    """Aggregate performance over a sweep of `RequestSample` rows."""

    n_requests: int
    n_success: int
    duration_s: float
    ttft: LatencyStats
    itl: LatencyStats
    tpot: LatencyStats
    e2e: LatencyStats
    output_tokens_total: int
    output_tokens_per_s: float
    request_rate_per_s: float


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (NumPy default). q in [0, 100]."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def latency_stats(values: list[float]) -> LatencyStats:
    """Compute mean + p50/p95/p99/min/max for a non-empty sequence."""
    if not values:
        raise ValueError("latency_stats() requires at least one value")
    return LatencyStats(
        count=len(values),
        mean=sum(values) / len(values),
        p50=percentile(values, 50.0),
        p95=percentile(values, 95.0),
        p99=percentile(values, 99.0),
        min=min(values),
        max=max(values),
    )


def _empty_stats() -> LatencyStats:
    return LatencyStats(count=0, mean=0.0, p50=0.0, p95=0.0, p99=0.0, min=0.0, max=0.0)


def summarize(samples: list[RequestSample], duration_s: float) -> PerfSummary:
    """Aggregate per-request samples into a `PerfSummary`.

    `duration_s` is the wallclock of the workload (used for throughput).
    Failed requests contribute to counts but are excluded from latency
    distributions.
    """
    n_requests = len(samples)
    successes = [s for s in samples if s.success]
    n_success = len(successes)

    ttfts = [s.ttft_s for s in successes]
    itls = [v for s in successes for v in s.itl_s]
    tpots = [
        (s.e2e_s - s.ttft_s) / max(s.output_tokens - 1, 1)
        for s in successes
        if s.output_tokens > 1
    ]
    e2es = [s.e2e_s for s in successes]
    output_tokens_total = sum(s.output_tokens for s in successes)

    if duration_s <= 0:
        output_tokens_per_s = 0.0
        request_rate_per_s = 0.0
    else:
        output_tokens_per_s = output_tokens_total / duration_s
        request_rate_per_s = n_success / duration_s

    return PerfSummary(
        n_requests=n_requests,
        n_success=n_success,
        duration_s=duration_s,
        ttft=latency_stats(ttfts) if ttfts else _empty_stats(),
        itl=latency_stats(itls) if itls else _empty_stats(),
        tpot=latency_stats(tpots) if tpots else _empty_stats(),
        e2e=latency_stats(e2es) if e2es else _empty_stats(),
        output_tokens_total=output_tokens_total,
        output_tokens_per_s=output_tokens_per_s,
        request_rate_per_s=request_rate_per_s,
    )


def bootstrap_ci(
    values: list[float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap (1 - alpha) confidence interval for the mean.

    Returns `(low, high)`. Uses a seeded `random.Random` so results are
    reproducible. Cheap and assumption-free; appropriate for small N.
    """
    if not values:
        raise ValueError("bootstrap_ci() requires at least one value")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    low = percentile(means, 100.0 * (alpha / 2))
    high = percentile(means, 100.0 * (1.0 - alpha / 2))
    return low, high
