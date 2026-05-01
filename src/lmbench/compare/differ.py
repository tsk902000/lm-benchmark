"""Pure (baseline, candidate) -> ComparisonReport differ.

Produces absolute and relative deltas with regression flags for the
report layer. Bigger-is-better for throughput and quality, smaller-is-
better for latency.
"""

from __future__ import annotations

from dataclasses import dataclass

from lmbench.bench import LatencyStats, PerfSummary, QualityResult, TaskScore


@dataclass(frozen=True)
class MetricDelta:
    """One scalar delta with regression flag."""

    name: str
    baseline: float
    candidate: float
    abs_delta: float
    rel_delta: float
    regression: bool

    @classmethod
    def make(
        cls,
        name: str,
        baseline: float,
        candidate: float,
        *,
        lower_is_better: bool,
        threshold_pct: float,
    ) -> MetricDelta:
        abs_d = candidate - baseline
        rel_d = (abs_d / baseline) if baseline else 0.0
        if lower_is_better:
            regression = rel_d > (threshold_pct / 100.0)
        else:
            regression = rel_d < -(threshold_pct / 100.0)
        return cls(
            name=name,
            baseline=baseline,
            candidate=candidate,
            abs_delta=abs_d,
            rel_delta=rel_d,
            regression=regression,
        )


def _stat_deltas(
    prefix: str,
    baseline: LatencyStats,
    candidate: LatencyStats,
    *,
    threshold_pct: float,
) -> tuple[MetricDelta, ...]:
    """Build deltas for a `LatencyStats` block (mean / p50 / p95 / p99)."""
    return tuple(
        MetricDelta.make(
            f"{prefix}.{field}",
            getattr(baseline, field),
            getattr(candidate, field),
            lower_is_better=True,
            threshold_pct=threshold_pct,
        )
        for field in ("mean", "p50", "p95", "p99")
    )


@dataclass(frozen=True)
class PerfComparison:
    """Side-by-side perf comparison for one (workload, concurrency) cell."""

    workload_name: str
    concurrency: int
    deltas: tuple[MetricDelta, ...]
    any_regression: bool


@dataclass(frozen=True)
class QualityComparison:
    """Side-by-side quality comparison."""

    suite_name: str
    deltas: tuple[MetricDelta, ...]
    any_regression: bool


@dataclass(frozen=True)
class ComparisonReport:
    """Full baseline-vs-candidate report."""

    perf: tuple[PerfComparison, ...]
    quality: tuple[QualityComparison, ...]

    @property
    def any_regression(self) -> bool:
        return any(p.any_regression for p in self.perf) or any(
            q.any_regression for q in self.quality
        )


def diff_perf(
    *,
    workload_name: str,
    concurrency: int,
    baseline: PerfSummary,
    candidate: PerfSummary,
    latency_threshold_pct: float = 5.0,
    throughput_threshold_pct: float = 5.0,
) -> PerfComparison:
    """Compute deltas for one (workload, concurrency) cell."""
    latency_deltas: list[MetricDelta] = []
    for prefix, b, c in (
        ("ttft", baseline.ttft, candidate.ttft),
        ("itl", baseline.itl, candidate.itl),
        ("tpot", baseline.tpot, candidate.tpot),
        ("e2e", baseline.e2e, candidate.e2e),
    ):
        latency_deltas.extend(
            _stat_deltas(prefix, b, c, threshold_pct=latency_threshold_pct)
        )
    throughput_delta = MetricDelta.make(
        "throughput.tokens_per_s",
        baseline.output_tokens_per_s,
        candidate.output_tokens_per_s,
        lower_is_better=False,
        threshold_pct=throughput_threshold_pct,
    )
    request_rate_delta = MetricDelta.make(
        "throughput.request_rate",
        baseline.request_rate_per_s,
        candidate.request_rate_per_s,
        lower_is_better=False,
        threshold_pct=throughput_threshold_pct,
    )
    deltas = (*latency_deltas, throughput_delta, request_rate_delta)
    return PerfComparison(
        workload_name=workload_name,
        concurrency=concurrency,
        deltas=deltas,
        any_regression=any(d.regression for d in deltas),
    )


def diff_quality(
    *,
    baseline: QualityResult,
    candidate: QualityResult,
    threshold_pct: float = 1.0,
) -> QualityComparison:
    """Compute task-level deltas; flags any task that drops > threshold_pct."""
    by_task_baseline: dict[str, TaskScore] = {s.task: s for s in baseline.scores}
    by_task_candidate: dict[str, TaskScore] = {s.task: s for s in candidate.scores}
    common_tasks = sorted(set(by_task_baseline) & set(by_task_candidate))
    deltas = tuple(
        MetricDelta.make(
            f"{task}.{by_task_baseline[task].metric}",
            by_task_baseline[task].value,
            by_task_candidate[task].value,
            lower_is_better=False,
            threshold_pct=threshold_pct,
        )
        for task in common_tasks
    )
    return QualityComparison(
        suite_name=baseline.suite_name,
        deltas=deltas,
        any_regression=any(d.regression for d in deltas),
    )
