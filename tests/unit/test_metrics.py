"""Unit tests for `lmbench.bench.metrics`."""

from __future__ import annotations

import math

import pytest

from lmbench.bench import (
    RequestSample,
    bootstrap_ci,
    latency_stats,
    percentile,
    summarize,
)


def test_percentile_single_value() -> None:
    assert percentile([5.0], 50.0) == 5.0
    assert percentile([5.0], 99.0) == 5.0


def test_percentile_p50_p95_p99() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50.0) == pytest.approx(50.5)
    assert percentile(values, 95.0) == pytest.approx(95.05)
    assert percentile(values, 99.0) == pytest.approx(99.01)


def test_percentile_validates_q() -> None:
    with pytest.raises(ValueError, match=r"q must be in"):
        percentile([1.0], -1.0)
    with pytest.raises(ValueError, match=r"q must be in"):
        percentile([1.0], 101.0)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 50.0)


def test_latency_stats_basic() -> None:
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.count == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.p50 == pytest.approx(3.0)
    assert stats.min == 1.0
    assert stats.max == 5.0


def test_latency_stats_rejects_empty() -> None:
    with pytest.raises(ValueError):
        latency_stats([])


def _sample(ttft: float, itls: tuple[float, ...], output_tokens: int) -> RequestSample:
    e2e = ttft + sum(itls)
    return RequestSample(ttft_s=ttft, itl_s=itls, e2e_s=e2e, output_tokens=output_tokens)


def test_summarize_basic() -> None:
    samples = [
        _sample(0.10, (0.02, 0.02, 0.02), 4),
        _sample(0.12, (0.03, 0.03, 0.03), 4),
        _sample(0.08, (0.01, 0.01, 0.01), 4),
    ]
    summary = summarize(samples, duration_s=1.0)
    assert summary.n_requests == 3
    assert summary.n_success == 3
    assert summary.ttft.count == 3
    assert summary.ttft.mean == pytest.approx(0.10)
    assert summary.itl.count == 9
    assert summary.tpot.count == 3
    assert summary.output_tokens_total == 12
    assert summary.output_tokens_per_s == pytest.approx(12.0)
    assert summary.request_rate_per_s == pytest.approx(3.0)


def test_summarize_skips_failed_for_latency() -> None:
    s_ok = _sample(0.10, (0.02,), 2)
    s_fail = RequestSample(
        ttft_s=0.0, itl_s=(), e2e_s=0.5, output_tokens=0, success=False
    )
    summary = summarize([s_ok, s_fail], duration_s=1.0)
    assert summary.n_requests == 2
    assert summary.n_success == 1
    assert summary.ttft.count == 1
    assert summary.output_tokens_total == 2


def test_summarize_zero_duration_yields_zero_throughput() -> None:
    samples = [_sample(0.1, (0.02,), 2)]
    summary = summarize(samples, duration_s=0.0)
    assert summary.output_tokens_per_s == 0.0
    assert summary.request_rate_per_s == 0.0


def test_summarize_all_failures() -> None:
    fails = [
        RequestSample(ttft_s=0.0, itl_s=(), e2e_s=0.1, output_tokens=0, success=False)
        for _ in range(3)
    ]
    summary = summarize(fails, duration_s=1.0)
    assert summary.n_success == 0
    assert summary.ttft.count == 0
    assert summary.itl.count == 0
    assert summary.output_tokens_per_s == 0.0


def test_summarize_single_token_output_excluded_from_tpot() -> None:
    samples = [_sample(0.1, (), 1), _sample(0.1, (0.02, 0.02), 3)]
    summary = summarize(samples, duration_s=1.0)
    assert summary.tpot.count == 1


def test_bootstrap_ci_envelope_contains_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = bootstrap_ci(values, alpha=0.05, n_bootstrap=500, seed=42)
    mean = sum(values) / len(values)
    assert low <= mean <= high


def test_bootstrap_ci_deterministic_with_seed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_ci(values, seed=7, n_bootstrap=200)
    b = bootstrap_ci(values, seed=7, n_bootstrap=200)
    assert a == b


def test_bootstrap_ci_validates_inputs() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], alpha=1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], n_bootstrap=0)


def test_summary_fields_finite() -> None:
    samples = [_sample(0.1, (0.02, 0.02), 3)]
    summary = summarize(samples, duration_s=1.0)
    for v in (
        summary.ttft.mean,
        summary.ttft.p50,
        summary.itl.mean,
        summary.tpot.mean,
        summary.e2e.mean,
    ):
        assert math.isfinite(v)
