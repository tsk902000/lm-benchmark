"""Load saved runner artifacts and build an offline comparison report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lmbench.bench import LatencyStats, PerfSummary, QualityResult, TaskScore

from .differ import (
    ComparisonReport,
    PerfComparison,
    QualityComparison,
    diff_perf,
    diff_quality,
)


def _require_mapping(value: Any, *, path: Path, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping at {key!r}")
    return value


def _latency_stats(payload: dict[str, Any], *, path: Path, key: str) -> LatencyStats:
    try:
        return LatencyStats(
            count=int(payload["count"]),
            mean=float(payload["mean"]),
            p50=float(payload["p50"]),
            p95=float(payload["p95"]),
            p99=float(payload["p99"]),
            min=float(payload["min"]),
            max=float(payload["max"]),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing {key}.{exc.args[0]}") from exc


def load_perf_summary(path: Path) -> tuple[str, int, PerfSummary]:
    """Load one saved `perf/*.json` artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = _require_mapping(payload, path=path, key="$")
    summary = _require_mapping(data.get("summary"), path=path, key="summary")
    try:
        workload_name = str(data["workload_name"])
        concurrency = int(data["concurrency"])
        return (
            workload_name,
            concurrency,
            PerfSummary(
                n_requests=int(summary["n_requests"]),
                n_success=int(summary["n_success"]),
                duration_s=float(summary["duration_s"]),
                ttft=_latency_stats(
                    _require_mapping(summary.get("ttft"), path=path, key="summary.ttft"),
                    path=path,
                    key="summary.ttft",
                ),
                itl=_latency_stats(
                    _require_mapping(summary.get("itl"), path=path, key="summary.itl"),
                    path=path,
                    key="summary.itl",
                ),
                tpot=_latency_stats(
                    _require_mapping(summary.get("tpot"), path=path, key="summary.tpot"),
                    path=path,
                    key="summary.tpot",
                ),
                e2e=_latency_stats(
                    _require_mapping(summary.get("e2e"), path=path, key="summary.e2e"),
                    path=path,
                    key="summary.e2e",
                ),
                output_tokens_total=int(summary["output_tokens_total"]),
                output_tokens_per_s=float(summary["output_tokens_per_s"]),
                request_rate_per_s=float(summary["request_rate_per_s"]),
            ),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing {exc.args[0]}") from exc


def load_quality_summary(path: Path) -> QualityResult:
    """Load one saved `quality/quality_summary.json` artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = _require_mapping(payload, path=path, key="$")
    scores_raw = data.get("scores")
    if not isinstance(scores_raw, list):
        raise ValueError(f"{path}: expected list at 'scores'")
    scores: list[TaskScore] = []
    for idx, score in enumerate(scores_raw):
        item = _require_mapping(score, path=path, key=f"scores[{idx}]")
        try:
            stderr_raw = item.get("stderr")
            scores.append(
                TaskScore(
                    task=str(item["task"]),
                    metric=str(item["metric"]),
                    value=float(item["value"]),
                    stderr=None if stderr_raw is None else float(stderr_raw),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}: missing scores[{idx}].{exc.args[0]}") from exc
    return QualityResult(
        suite_name=str(data.get("suite_name", "quality")),
        served_model_name=str(data.get("served_model_name", "")),
        scores=tuple(scores),
        raw_results_path=path,
    )


def _load_perf_dir(stage_dir: Path) -> dict[tuple[str, int], PerfSummary]:
    perf_dir = stage_dir / "perf"
    if not perf_dir.exists():
        return {}
    out: dict[tuple[str, int], PerfSummary] = {}
    for path in sorted(perf_dir.glob("*.json")):
        workload_name, concurrency, summary = load_perf_summary(path)
        out[(workload_name, concurrency)] = summary
    return out


def compare_result_dirs(baseline: Path, candidate: Path) -> ComparisonReport:
    """Compare two saved stage directories, e.g. `baseline/` and `quantized/`."""
    baseline_perf = _load_perf_dir(baseline)
    candidate_perf = _load_perf_dir(candidate)
    perf: list[PerfComparison] = []
    for workload_name, concurrency in sorted(set(baseline_perf) & set(candidate_perf)):
        perf.append(
            diff_perf(
                workload_name=workload_name,
                concurrency=concurrency,
                baseline=baseline_perf[(workload_name, concurrency)],
                candidate=candidate_perf[(workload_name, concurrency)],
            )
        )

    quality: list[QualityComparison] = []
    baseline_quality_path = baseline / "quality" / "quality_summary.json"
    candidate_quality_path = candidate / "quality" / "quality_summary.json"
    if baseline_quality_path.exists() and candidate_quality_path.exists():
        quality.append(
            diff_quality(
                baseline=load_quality_summary(baseline_quality_path),
                candidate=load_quality_summary(candidate_quality_path),
            )
        )

    return ComparisonReport(perf=tuple(perf), quality=tuple(quality))
