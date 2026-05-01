"""Quality benchmark driver.

Wraps the `lm-eval` (lm-evaluation-harness) CLI in `local-completions`
mode pointed at a running vLLM server. Parses the resulting `results_*.json`
into a `QualityResult` and persists a normalized summary JSON next to
the raw artifacts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lmbench.config import EvalSuite, ModelEntry


@dataclass(frozen=True)
class TaskScore:
    """One eval task's headline score."""

    task: str
    metric: str
    value: float
    stderr: float | None = None


@dataclass(frozen=True)
class QualityResult:
    """Parsed quality-eval output."""

    suite_name: str
    served_model_name: str
    scores: tuple[TaskScore, ...]
    raw_results_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "served_model_name": self.served_model_name,
            "scores": [
                {
                    "task": s.task,
                    "metric": s.metric,
                    "value": s.value,
                    "stderr": s.stderr,
                }
                for s in self.scores
            ],
            "raw_results_path": str(self.raw_results_path),
        }


_PRIMARY_METRIC_PRIORITY = (
    "acc_norm",
    "acc",
    "exact_match",
    "f1",
    "mc2",
    "pass@1",
    "pass_at_1",
)


def _strip_filter(metric_key: str) -> tuple[str, str]:
    """Split `acc,none` -> ("acc", "none"). Returns ("acc", "") if no filter."""
    if "," in metric_key:
        name, filt = metric_key.split(",", 1)
        return name, filt
    return metric_key, ""


def _pick_primary_metric(
    task_metrics: dict[str, Any],
) -> tuple[str, float, float | None]:
    """Choose a single primary metric from one task's metric block.

    lm-eval emits keys like `acc,none`, `acc_norm,none`, `acc_stderr,none`.
    Order of preference is `_PRIMARY_METRIC_PRIORITY`. Returns
    `(metric_name, value, stderr_or_None)`.
    """
    bare: dict[str, float] = {}
    stderrs: dict[str, float] = {}
    for raw_key, val in task_metrics.items():
        if not isinstance(val, (int, float)):
            continue
        name, _ = _strip_filter(raw_key)
        if name.endswith("_stderr"):
            stderrs[name[: -len("_stderr")]] = float(val)
        else:
            bare[name] = float(val)
    for candidate in _PRIMARY_METRIC_PRIORITY:
        if candidate in bare:
            return candidate, bare[candidate], stderrs.get(candidate)
    if not bare:
        raise ValueError(f"no numeric metrics found in task block: {task_metrics!r}")
    name = next(iter(bare))
    return name, bare[name], stderrs.get(name)


def parse_lm_eval_results(
    results_json: Path, *, suite_name: str, served_model_name: str
) -> QualityResult:
    """Parse an `lm-eval` `results_*.json` file into a `QualityResult`."""
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    results_block = payload.get("results")
    if not isinstance(results_block, dict):
        raise ValueError(f"{results_json}: missing or non-dict 'results' block")
    scores: list[TaskScore] = []
    for task_name, metrics in results_block.items():
        if not isinstance(metrics, dict):
            continue
        try:
            metric_name, value, stderr = _pick_primary_metric(metrics)
        except ValueError:
            continue
        scores.append(
            TaskScore(task=task_name, metric=metric_name, value=value, stderr=stderr)
        )
    return QualityResult(
        suite_name=suite_name,
        served_model_name=served_model_name,
        scores=tuple(scores),
        raw_results_path=results_json,
    )


def build_lm_eval_args(
    *,
    base_url: str,
    served_model_name: str,
    suite: EvalSuite,
    output_dir: Path,
    executable: str = "lm_eval",
) -> list[str]:
    """Build the `lm_eval` CLI argv for an `EvalSuite` against a vLLM server."""
    tasks = ",".join(suite.tasks)
    model_args = (
        f"base_url={base_url}/v1/completions,"
        f"model={served_model_name},"
        "tokenized_requests=False"
    )
    args = [
        executable,
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]
    if suite.limit is not None:
        args += ["--limit", str(suite.limit)]
    if suite.num_fewshot:
        # lm-eval accepts a single `--num_fewshot N` (applied to all tasks).
        # We collapse to the max requested value to keep tasks comparable;
        # callers needing per-task control should split the suite.
        args += ["--num_fewshot", str(max(suite.num_fewshot.values()))]
    return args


def _find_results_json(output_dir: Path) -> Path:
    """Locate the lm-eval `results_*.json` under output_dir."""
    candidates = sorted(output_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no results_*.json under {output_dir}; lm-eval may have failed"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_quality(
    *,
    suite: EvalSuite,
    model: ModelEntry,
    base_url: str,
    served_model_name: str,
    output_dir: Path,
    executable: str = "lm_eval",
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> QualityResult:
    """Run an `EvalSuite` and return a parsed `QualityResult`.

    Also writes a normalized `quality_summary.json` to `output_dir`.
    """
    del model  # reserved for tokenizer/family hints in later phases
    if executable == "lm_eval":
        resolved = shutil.which(executable) or shutil.which("lm-eval")
        if resolved is None:
            raise FileNotFoundError(
                "'lm_eval' not found on PATH; install the [gpu] extra "
                "(`uv sync --extra all`) on a CUDA host"
            )
    else:
        resolved = executable

    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_lm_eval_args(
        base_url=base_url,
        served_model_name=served_model_name,
        suite=suite,
        output_dir=output_dir,
        executable=resolved,
    )

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"lm_eval exited with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    results_path = _find_results_json(output_dir)
    parsed = parse_lm_eval_results(
        results_path, suite_name=suite.name, served_model_name=served_model_name
    )
    summary_path = output_dir / "quality_summary.json"
    summary_path.write_text(json.dumps(parsed.as_dict(), indent=2), encoding="utf-8")
    return parsed
