"""Unit tests for `lmbench.compare` and `lmbench.report`."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmbench.bench import LatencyStats, PerfSummary, QualityResult, TaskScore
from lmbench.compare import (
    ComparisonReport,
    MetricDelta,
    delta_bootstrap_ci,
    diff_perf,
    diff_quality,
    is_within_tolerance,
)
from lmbench.report import render_html, render_markdown, write_html, write_markdown


def _stats(mean: float) -> LatencyStats:
    return LatencyStats(
        count=10,
        mean=mean,
        p50=mean,
        p95=mean * 1.1,
        p99=mean * 1.2,
        min=mean * 0.9,
        max=mean * 1.3,
    )


def _summary(*, ttft: float, tokens_per_s: float) -> PerfSummary:
    return PerfSummary(
        n_requests=10,
        n_success=10,
        duration_s=1.0,
        ttft=_stats(ttft),
        itl=_stats(0.02),
        tpot=_stats(0.02),
        e2e=_stats(ttft * 4),
        output_tokens_total=1000,
        output_tokens_per_s=tokens_per_s,
        request_rate_per_s=10.0,
    )


# ---- MetricDelta ------------------------------------------------------


def test_metric_delta_lower_is_better_regression() -> None:
    d = MetricDelta.make(
        "ttft.mean",
        baseline=0.1,
        candidate=0.12,
        lower_is_better=True,
        threshold_pct=5.0,
    )
    assert d.regression is True
    assert d.abs_delta == pytest.approx(0.02)
    assert d.rel_delta == pytest.approx(0.2)


def test_metric_delta_higher_is_better_regression() -> None:
    d = MetricDelta.make(
        "tps", baseline=100.0, candidate=80.0, lower_is_better=False, threshold_pct=5.0
    )
    assert d.regression is True


def test_metric_delta_within_threshold() -> None:
    d = MetricDelta.make(
        "ttft.mean",
        baseline=0.1,
        candidate=0.103,
        lower_is_better=True,
        threshold_pct=5.0,
    )
    assert d.regression is False


def test_metric_delta_zero_baseline_handled() -> None:
    d = MetricDelta.make(
        "x", baseline=0.0, candidate=1.0, lower_is_better=False, threshold_pct=5.0
    )
    assert d.rel_delta == 0.0
    assert d.regression is False


# ---- diff_perf / diff_quality -----------------------------------------


def test_diff_perf_emits_all_metrics() -> None:
    base = _summary(ttft=0.1, tokens_per_s=1000.0)
    cand = _summary(ttft=0.09, tokens_per_s=1100.0)
    cmp = diff_perf(workload_name="w", concurrency=8, baseline=base, candidate=cand)
    assert cmp.workload_name == "w"
    assert cmp.concurrency == 8
    # 4 latency metrics * 4 stat fields + 2 throughput = 18
    assert len(cmp.deltas) == 18
    assert cmp.any_regression is False


def test_diff_perf_flags_regression() -> None:
    base = _summary(ttft=0.1, tokens_per_s=1000.0)
    bad = _summary(ttft=0.2, tokens_per_s=500.0)
    cmp = diff_perf(workload_name="w", concurrency=8, baseline=base, candidate=bad)
    assert cmp.any_regression is True


def test_diff_quality_filters_to_common_tasks() -> None:
    base = QualityResult(
        suite_name="s",
        served_model_name="m",
        scores=(
            TaskScore("mmlu", "acc", 0.5),
            TaskScore("gsm8k", "exact_match", 0.2),
        ),
        raw_results_path=Path("/tmp/r.json"),
    )
    cand = QualityResult(
        suite_name="s",
        served_model_name="m-q",
        scores=(
            TaskScore("mmlu", "acc", 0.49),
            TaskScore("hellaswag", "acc", 0.7),
        ),
        raw_results_path=Path("/tmp/r2.json"),
    )
    cmp = diff_quality(baseline=base, candidate=cand, threshold_pct=1.0)
    assert {d.name.split(".")[0] for d in cmp.deltas} == {"mmlu"}
    assert cmp.any_regression is True


# ---- delta_bootstrap_ci ----------------------------------------------


def test_delta_bootstrap_ci_significant_drop() -> None:
    baseline = [0.5] * 50
    candidate = [0.40] * 50
    ci = delta_bootstrap_ci(baseline, candidate, n_bootstrap=200, seed=0)
    assert ci.mean_delta == pytest.approx(-0.10)
    assert ci.significant is True


def test_delta_bootstrap_ci_no_significant_change() -> None:
    baseline = [float(v) for v in range(1, 21)]
    candidate = list(baseline)
    ci = delta_bootstrap_ci(baseline, candidate, n_bootstrap=200, seed=0)
    assert ci.significant is False


def test_delta_bootstrap_ci_rejects_empty() -> None:
    with pytest.raises(ValueError):
        delta_bootstrap_ci([], [1.0])


def test_is_within_tolerance() -> None:
    assert is_within_tolerance(1.0, 1.001, abs_tol=0.01)
    assert is_within_tolerance(100.0, 101.0, rel_tol=0.02)
    assert not is_within_tolerance(1.0, 1.5, abs_tol=0.1)


# ---- report rendering -------------------------------------------------


def _basic_report() -> ComparisonReport:
    base = _summary(ttft=0.1, tokens_per_s=1000.0)
    cand = _summary(ttft=0.11, tokens_per_s=1050.0)
    perf = (diff_perf(workload_name="w", concurrency=8, baseline=base, candidate=cand),)
    quality = (
        diff_quality(
            baseline=QualityResult(
                "s", "m", (TaskScore("mmlu", "acc", 0.5),), Path("/x")
            ),
            candidate=QualityResult(
                "s", "m-q", (TaskScore("mmlu", "acc", 0.495),), Path("/y")
            ),
        ),
    )
    return ComparisonReport(perf=perf, quality=quality)


def test_render_markdown_contains_headers_and_metrics() -> None:
    md = render_markdown(_basic_report(), title="my-run")
    assert "# my-run" in md
    assert "## Performance" in md
    assert "## Quality" in md
    assert "ttft.mean" in md
    assert "mmlu.acc" in md


def test_render_markdown_no_regression_banner() -> None:
    base = _summary(ttft=0.1, tokens_per_s=1000.0)
    cand = _summary(ttft=0.099, tokens_per_s=1010.0)
    report = ComparisonReport(
        perf=(
            diff_perf(workload_name="w", concurrency=1, baseline=base, candidate=cand),
        ),
        quality=(),
    )
    md = render_markdown(report)
    assert "No regressions" in md


def test_render_markdown_regression_banner() -> None:
    base = _summary(ttft=0.1, tokens_per_s=1000.0)
    bad = _summary(ttft=0.5, tokens_per_s=100.0)
    report = ComparisonReport(
        perf=(
            diff_perf(workload_name="w", concurrency=1, baseline=base, candidate=bad),
        ),
        quality=(),
    )
    md = render_markdown(report)
    assert "Regression detected" in md


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "report.md"
    written = write_markdown(_basic_report(), target)
    assert written == target
    assert target.exists()
    assert "lmbench" in target.read_text(encoding="utf-8")


def test_render_html_contains_title_and_tables() -> None:
    html = render_html(_basic_report(), title="my-run")
    assert "my-run" in html
    assert "<html" in html.lower()
    assert "Metric" in html or "metric" in html.lower()


def test_write_html_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "report.html"
    written = write_html(_basic_report(), target)
    assert written == target
    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
