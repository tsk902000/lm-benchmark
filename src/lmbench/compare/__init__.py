"""Baseline-vs-candidate result diffing and statistical comparison."""

from __future__ import annotations

from .differ import (
    ComparisonReport,
    MetricDelta,
    PerfComparison,
    QualityComparison,
    diff_perf,
    diff_quality,
)
from .offline import compare_result_dirs, load_perf_summary, load_quality_summary
from .stats import DeltaCI, delta_bootstrap_ci, is_within_tolerance

__all__ = [
    "ComparisonReport",
    "DeltaCI",
    "MetricDelta",
    "PerfComparison",
    "QualityComparison",
    "compare_result_dirs",
    "delta_bootstrap_ci",
    "diff_perf",
    "diff_quality",
    "is_within_tolerance",
    "load_perf_summary",
    "load_quality_summary",
]
