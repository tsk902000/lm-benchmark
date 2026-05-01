"""Unit tests for `lmbench.runner.env` and `lmbench.runner.pipeline`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lmbench.bench import (
    LatencyStats,
    PerfResult,
    PerfSummary,
    Prompt,
    QualityResult,
    RequestSample,
    TaskScore,
)
from lmbench.config import RunPlan
from lmbench.quantize import QuantizedCheckpoint, VerifyResult
from lmbench.runner import EnvCapture, capture_environment, capture_to_path, run_plan
from lmbench.runner import env as env_mod
from lmbench.runner import pipeline as pipeline_mod
from lmbench.serve import ServerHandle

# ---- env capture ------------------------------------------------------


def test_capture_environment_returns_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_mod, "_safe_run", lambda _argv: "fake-output")
    monkeypatch.setattr(env_mod, "_package_versions", lambda: {"vllm": "0.7.0"})
    capture = capture_environment()
    assert isinstance(capture, EnvCapture)
    assert capture.os
    assert capture.python
    assert capture.packages == {"vllm": "0.7.0"}


def test_capture_to_path_writes_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(env_mod, "_safe_run", lambda _argv: None)
    monkeypatch.setattr(env_mod, "_package_versions", lambda: {})
    target = tmp_path / "env" / "env.json"
    capture = capture_to_path(target)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["os"] == capture.os
    assert payload["packages"] == {}


def test_safe_run_handles_missing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("not on PATH")

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
    assert env_mod._safe_run(["does-not-exist"]) is None


def test_safe_run_returns_none_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class _Proc:
        returncode: int = 2
        stdout: str = ""
        stderr: str = "boom"

    monkeypatch.setattr(env_mod.subprocess, "run", lambda *_a, **_kw: _Proc())
    assert env_mod._safe_run(["false"]) is None


# ---- pipeline orchestration ------------------------------------------


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


def _perf_result(workload: str, concurrency: int, *, ttft: float = 0.1) -> PerfResult:
    summary = PerfSummary(
        n_requests=4,
        n_success=4,
        duration_s=1.0,
        ttft=_stats(ttft),
        itl=_stats(0.02),
        tpot=_stats(0.02),
        e2e=_stats(ttft * 4),
        output_tokens_total=400,
        output_tokens_per_s=400.0,
        request_rate_per_s=4.0,
    )
    samples = tuple(
        RequestSample(
            ttft_s=ttft, itl_s=(0.02, 0.02), e2e_s=ttft + 0.04, output_tokens=3
        )
        for _ in range(4)
    )
    return PerfResult(
        samples=samples,
        summary=summary,
        gpu_summary={},
        concurrency=concurrency,
        workload_name=workload,
    )


def _quality_result(value: float) -> QualityResult:
    return QualityResult(
        suite_name="default",
        served_model_name="m",
        scores=(TaskScore(task="mmlu", metric="acc", value=value),),
        raw_results_path=Path("/tmp/r.json"),
    )


def _smoke_plan() -> RunPlan:
    return RunPlan.model_validate(
        {
            "name": "smoke",
            "models": [{"name": "opt", "hf_id": "facebook/opt-125m"}],
            "workloads": [
                {
                    "name": "w",
                    "kind": "random",
                    "num_prompts": 4,
                    "concurrency": [1, 2],
                    "input_len": 32,
                    "output_len": 8,
                    "warmup_prompts": 0,
                }
            ],
            "eval_suite": {"name": "s", "tasks": ["mmlu"]},
            "hardware": {
                "name": "hp",
                "gpu": "H100",
                "num_gpus": 1,
                "default_tp_size": 1,
            },
        }
    )


@contextmanager
def _fake_serve(*_a: Any, **_kw: Any) -> Iterator[ServerHandle]:
    @dataclass
    class _FakeProc:
        pid: int = 1
        returncode: int | None = None

        def poll(self) -> int | None:
            return None

    handle = ServerHandle(
        process=_FakeProc(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8000,
        served_model_name="opt",
        log_paths=None,
    )
    yield handle


def test_run_plan_skip_quantize_no_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(env_mod, "_safe_run", lambda _argv: None)
    monkeypatch.setattr(env_mod, "_package_versions", lambda: {})
    monkeypatch.setattr(pipeline_mod, "serve_model", _fake_serve)

    perf_calls: list[tuple[str, int]] = []

    def fake_run_workload(
        *,
        base_url: str,
        served_model_name: str,
        workload: Any,
        concurrency: int,
        prompts: Any,
        gpu_sampler: Any = None,
    ) -> PerfResult:
        del base_url, served_model_name, prompts, gpu_sampler
        perf_calls.append((workload.name, concurrency))
        return _perf_result(workload.name, concurrency)

    monkeypatch.setattr(pipeline_mod, "run_workload", fake_run_workload)
    monkeypatch.setattr(
        pipeline_mod,
        "generate",
        lambda spec: tuple(
            Prompt(text=f"p-{i}", expected_output_tokens=spec.output_len or 8)
            for i in range(spec.num_prompts)
        ),
    )

    result = run_plan(
        _smoke_plan(),
        output_dir=tmp_path / "out",
        skip_quality=True,
        skip_quantize=True,
    )
    assert result.plan_name == "smoke"
    assert len(result.models) == 1
    only = result.models[0]
    assert {(w, c) for w, c in perf_calls} == {("w", 1), ("w", 2)}
    assert only.baseline_quality is None
    assert only.quantized_perf == ()
    assert only.quantized_quality is None
    assert only.quantized_checkpoint is None
    assert only.report_md.exists()
    assert only.report_html.exists()
    assert (tmp_path / "out" / "env.json").exists()


def test_run_plan_with_quantize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(env_mod, "_safe_run", lambda _argv: None)
    monkeypatch.setattr(env_mod, "_package_versions", lambda: {})
    monkeypatch.setattr(pipeline_mod, "serve_model", _fake_serve)
    monkeypatch.setattr(
        pipeline_mod,
        "generate",
        lambda spec: tuple(
            Prompt(text="p", expected_output_tokens=spec.output_len or 8)
            for _ in range(spec.num_prompts)
        ),
    )

    def fake_run_workload(
        *,
        base_url: str,
        served_model_name: str,
        workload: Any,
        concurrency: int,
        prompts: Any,
        gpu_sampler: Any = None,
    ) -> PerfResult:
        del base_url, prompts, gpu_sampler
        ttft = 0.10 if "quant" not in served_model_name else 0.08
        return _perf_result(workload.name, concurrency, ttft=ttft)

    monkeypatch.setattr(pipeline_mod, "run_workload", fake_run_workload)
    monkeypatch.setattr(
        pipeline_mod,
        "run_quality",
        lambda **_kw: _quality_result(0.50),
    )

    fake_ckpt = QuantizedCheckpoint(
        output_dir=tmp_path / "ckpt",
        method="nvfp4",
        source_hf_id="facebook/opt-125m",
    )
    monkeypatch.setattr(pipeline_mod, "quantize_to_nvfp4", lambda **_kw: fake_ckpt)
    monkeypatch.setattr(
        pipeline_mod,
        "verify_checkpoint",
        lambda **_kw: VerifyResult(ok=True, completion="Paris"),
    )

    plan = RunPlan.model_validate(
        {
            **_smoke_plan().model_dump(),
            "quant_recipe": {
                "name": "r",
                "method": "nvfp4",
                "calibration": {"num_samples": 4},
            },
            "hardware": {
                "name": "hp",
                "gpu": "B300",
                "blackwell": True,
                "num_gpus": 1,
                "default_tp_size": 1,
            },
        }
    )

    result = run_plan(plan, output_dir=tmp_path / "out2", skip_quality=False)
    assert len(result.models) == 1
    only = result.models[0]
    assert only.baseline_quality is not None
    assert only.quantized_quality is not None
    assert only.quantized_checkpoint is fake_ckpt
    perf_files = list((tmp_path / "out2" / "opt").rglob("*.json"))
    assert any(p.name.startswith("w_c1") for p in perf_files)


def test_run_plan_aborts_on_failed_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(env_mod, "_safe_run", lambda _argv: None)
    monkeypatch.setattr(env_mod, "_package_versions", lambda: {})
    monkeypatch.setattr(pipeline_mod, "serve_model", _fake_serve)
    monkeypatch.setattr(
        pipeline_mod,
        "generate",
        lambda spec: tuple(
            Prompt(text="p", expected_output_tokens=spec.output_len or 8)
            for _ in range(spec.num_prompts)
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_workload",
        lambda **kw: _perf_result(kw["workload"].name, kw["concurrency"]),
    )
    monkeypatch.setattr(pipeline_mod, "run_quality", lambda **_kw: _quality_result(0.5))
    fake_ckpt = QuantizedCheckpoint(
        output_dir=tmp_path / "ckpt",
        method="nvfp4",
        source_hf_id="facebook/opt-125m",
    )
    monkeypatch.setattr(pipeline_mod, "quantize_to_nvfp4", lambda **_kw: fake_ckpt)
    monkeypatch.setattr(
        pipeline_mod,
        "verify_checkpoint",
        lambda **_kw: VerifyResult(ok=False, completion="", reason="empty"),
    )

    plan = RunPlan.model_validate(
        {
            **_smoke_plan().model_dump(),
            "quant_recipe": {"name": "r", "method": "nvfp4"},
            "hardware": {
                "name": "hp",
                "gpu": "B300",
                "blackwell": True,
                "num_gpus": 1,
                "default_tp_size": 1,
            },
        }
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        run_plan(plan, output_dir=tmp_path / "out3", skip_quality=True)


def test_perf_summary_to_dict_round_trip() -> None:
    result = _perf_result("w", 1)
    payload = pipeline_mod._perf_summary_to_dict(result)
    assert payload["n_requests"] == 4
    assert payload["ttft"]["mean"] == pytest.approx(0.1)


def test_save_perf_result_writes_json(tmp_path: Path) -> None:
    result = _perf_result("w", 4)
    out = pipeline_mod._save_perf_result(result, tmp_path / "perf")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["workload_name"] == "w"
    assert payload["concurrency"] == 4
    assert payload["n_samples"] == 4
