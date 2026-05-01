"""Unit tests for `lmbench.bench.quality`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lmbench.bench import (
    QualityResult,
    build_lm_eval_args,
    merged_task_list,
    parse_lm_eval_results,
    run_quality,
)
from lmbench.bench import quality as quality_mod
from lmbench.config import EvalSuite, ModelEntry


def _write_results(path: Path, results: dict[str, Any]) -> Path:
    payload = {"results": results, "config": {"model": "x"}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_picks_acc_norm_when_present(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path / "results_x.json",
        {
            "mmlu": {
                "acc,none": 0.42,
                "acc_stderr,none": 0.01,
                "acc_norm,none": 0.45,
                "acc_norm_stderr,none": 0.012,
            }
        },
    )
    result = parse_lm_eval_results(p, suite_name="s", served_model_name="m")
    assert result.scores[0].metric == "acc_norm"
    assert result.scores[0].value == pytest.approx(0.45)
    assert result.scores[0].stderr == pytest.approx(0.012)


def test_parse_falls_back_to_acc(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path / "results_x.json",
        {"mmlu": {"acc,none": 0.42, "acc_stderr,none": 0.01}},
    )
    result = parse_lm_eval_results(p, suite_name="s", served_model_name="m")
    assert result.scores[0].metric == "acc"
    assert result.scores[0].value == pytest.approx(0.42)


def test_parse_handles_exact_match(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path / "results_x.json",
        {
            "gsm8k": {
                "exact_match,strict-match": 0.18,
                "exact_match_stderr,strict-match": 0.02,
            }
        },
    )
    result = parse_lm_eval_results(p, suite_name="s", served_model_name="m")
    assert result.scores[0].metric == "exact_match"


def test_parse_skips_tasks_without_numeric_metrics(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path / "results_x.json",
        {
            "good": {"acc,none": 0.5},
            "bad": {"version": "v1.0", "alias": "bad"},
        },
    )
    result = parse_lm_eval_results(p, suite_name="s", served_model_name="m")
    tasks = {s.task for s in result.scores}
    assert tasks == {"good"}


def test_parse_rejects_missing_results_block(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text(json.dumps({"config": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or non-dict"):
        parse_lm_eval_results(p, suite_name="s", served_model_name="m")


def test_build_lm_eval_args_basic() -> None:
    suite = EvalSuite(name="s", tasks=("mmlu", "gsm8k"))
    argv = build_lm_eval_args(
        base_url="http://127.0.0.1:8000",
        served_model_name="m",
        suite=suite,
        output_dir=Path("/tmp/q"),
    )
    assert argv[0] == "lm_eval"
    i = argv.index("--model_args")
    assert "base_url=http://127.0.0.1:8000/v1/completions" in argv[i + 1]
    assert "model=m" in argv[i + 1]
    assert argv[argv.index("--tasks") + 1] == "mmlu,gsm8k"
    assert "--limit" not in argv
    assert "--num_fewshot" not in argv


def test_build_lm_eval_args_with_limit_and_fewshot() -> None:
    suite = EvalSuite(
        name="s",
        tasks=("mmlu", "gsm8k"),
        num_fewshot={"mmlu": 5, "gsm8k": 3},
        limit=100,
    )
    argv = build_lm_eval_args(
        base_url="http://x",
        served_model_name="m",
        suite=suite,
        output_dir=Path("/tmp/q"),
    )
    assert argv[argv.index("--limit") + 1] == "100"
    assert argv[argv.index("--num_fewshot") + 1] == "5"


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_quality_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = EvalSuite(name="s", tasks=("mmlu",))
    model = ModelEntry(name="opt", hf_id="facebook/opt-125m")
    output_dir = tmp_path / "q"

    def fake_run(_argv: list[str], **_kw: Any) -> _FakeCompleted:
        out = output_dir / "results_2026.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "results": {"mmlu": {"acc,none": 0.42, "acc_stderr,none": 0.01}},
                    "config": {},
                }
            ),
            encoding="utf-8",
        )
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(quality_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(quality_mod.shutil, "which", lambda x: x)

    result = run_quality(
        suite=suite,
        model=model,
        base_url="http://127.0.0.1:8000",
        served_model_name="opt",
        output_dir=output_dir,
    )
    assert isinstance(result, QualityResult)
    assert result.scores[0].task == "mmlu"
    summary_path = output_dir / "quality_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["scores"][0]["task"] == "mmlu"


def test_run_quality_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = EvalSuite(name="s", tasks=("mmlu",))
    model = ModelEntry(name="opt", hf_id="facebook/opt-125m")

    def fake_run(_argv: list[str], **_kw: Any) -> _FakeCompleted:
        return _FakeCompleted(returncode=2, stderr="boom")

    monkeypatch.setattr(quality_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(quality_mod.shutil, "which", lambda x: x)
    with pytest.raises(RuntimeError, match="exited with code 2"):
        run_quality(
            suite=suite,
            model=model,
            base_url="http://x",
            served_model_name="opt",
            output_dir=tmp_path / "q",
        )


def test_run_quality_executable_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality_mod.shutil, "which", lambda _x: None)
    suite = EvalSuite(name="s", tasks=("mmlu",))
    model = ModelEntry(name="opt", hf_id="facebook/opt-125m")
    with pytest.raises(FileNotFoundError, match="lm_eval"):
        run_quality(
            suite=suite,
            model=model,
            base_url="http://x",
            served_model_name="opt",
            output_dir=Path("/tmp/q"),
        )


def test_run_quality_missing_results_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = EvalSuite(name="s", tasks=("mmlu",))
    model = ModelEntry(name="opt", hf_id="facebook/opt-125m")
    monkeypatch.setattr(
        quality_mod.subprocess,
        "run",
        lambda *_a, **_kw: _FakeCompleted(returncode=0),
    )
    monkeypatch.setattr(quality_mod.shutil, "which", lambda x: x)
    with pytest.raises(FileNotFoundError, match="results_"):
        run_quality(
            suite=suite,
            model=model,
            base_url="http://x",
            served_model_name="opt",
            output_dir=tmp_path / "q",
        )


# ---- long-context / coding task wiring -------------------------------


def test_merged_task_list_concatenates_and_dedups() -> None:
    suite = EvalSuite(
        name="s",
        tasks=("mmlu", "gsm8k"),
        long_context=("ruler", "longbench", "mmlu"),  # mmlu is duplicate
    )
    merged = merged_task_list(suite)
    assert merged == ("mmlu", "gsm8k", "ruler", "longbench")


def test_merged_task_list_long_context_only() -> None:
    suite = EvalSuite(
        name="long-only",
        tasks=("livecodebench",),
        long_context=("ruler", "longbench"),
    )
    merged = merged_task_list(suite)
    assert merged == ("livecodebench", "ruler", "longbench")


def test_build_lm_eval_args_includes_long_context() -> None:
    suite = EvalSuite(
        name="s",
        tasks=("mmlu",),
        long_context=("ruler", "livecodebench"),
    )
    argv = build_lm_eval_args(
        base_url="http://x",
        served_model_name="m",
        suite=suite,
        output_dir=Path("/tmp/q"),
    )
    tasks_value = argv[argv.index("--tasks") + 1]
    assert tasks_value == "mmlu,ruler,livecodebench"


def test_seed_benchmarks_yaml_exposes_long_context_tasks() -> None:
    """The shipped seed config wires up RULER / LongBench / LiveCodeBench."""
    from lmbench.config import load_eval_suite

    suite = load_eval_suite(Path("configs/benchmarks.yaml"))
    assert "ruler" in suite.long_context
    assert "longbench" in suite.long_context
    assert "livecodebench" in suite.long_context
    # Combined task list passed to lm-eval includes them.
    merged = merged_task_list(suite)
    assert {"ruler", "longbench", "livecodebench"} <= set(merged)
    # And the standard tasks are still there.
    assert "mmlu" in merged
    assert "gsm8k" in merged
