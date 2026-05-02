"""Smoke tests for the lmbench CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmbench import __version__
from lmbench.cli import app

runner = CliRunner()


def test_version_constant() -> None:
    assert __version__ == "0.1.0"


def test_root_help_runs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "lmbench" in result.output.lower()


def test_serve_help() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output.lower()


def test_bench_help() -> None:
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0


def test_quantize_help() -> None:
    result = runner.invoke(app, ["quantize", "--help"])
    assert result.exit_code == 0


def test_compare_help() -> None:
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0


def test_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_serve_stub_runs(empty_yaml: Path) -> None:
    result = runner.invoke(app, ["serve", "--config", str(empty_yaml), "--model", "demo"])
    assert result.exit_code == 0
    assert "serve" in result.output.lower()


def test_bench_stub_runs(empty_yaml: Path) -> None:
    result = runner.invoke(app, ["bench", "--config", str(empty_yaml), "--model", "demo"])
    assert result.exit_code == 0


def test_quantize_stub_runs(empty_yaml: Path) -> None:
    result = runner.invoke(app, ["quantize", "--config", str(empty_yaml), "--model", "demo"])
    assert result.exit_code == 0


def test_compare_writes_reports_for_saved_artifacts(tmp_path: Path) -> None:
    a = tmp_path / "baseline"
    a.mkdir()
    b = tmp_path / "candidate"
    b.mkdir()
    out = tmp_path / "report"
    result = runner.invoke(
        app,
        ["compare", "--baseline", str(a), "--candidate", str(b), "--output", str(out)],
    )
    assert result.exit_code == 0
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()


def test_run_dispatches_to_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`run` invokes `lmbench.runner.run_plan_from_file` with the plan path."""
    from lmbench.runner import PipelineResult
    from lmbench.runner.env import EnvCapture

    captured: dict[str, object] = {}

    def fake(
        path: Path,
        *,
        output_dir: Path | None = None,
        skip_quality: bool = False,
        skip_quantize: bool = False,
        skip_baseline: bool = False,
        progress: object | None = None,
    ) -> PipelineResult:
        captured["path"] = path
        captured["output_dir"] = output_dir
        captured["skip_quality"] = skip_quality
        captured["skip_quantize"] = skip_quantize
        captured["skip_baseline"] = skip_baseline
        captured["progress"] = progress
        env = EnvCapture(
            captured_at_unix=0.0,
            os="Linux",
            kernel="x",
            python="3.11",
            git_sha=None,
            git_dirty=False,
            nvidia_smi=None,
        )
        return PipelineResult(
            plan_name="smoke", output_dir=tmp_path, env=env, models=()
        )

    import lmbench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "run_plan_from_file", fake)
    plan = tmp_path / "plan.yaml"
    plan.write_text("plan: {}\n", encoding="utf-8")
    result = runner.invoke(
        app, ["run", "--plan", str(plan), "--skip-quality", "--skip-quantize"]
    )
    assert result.exit_code == 0, result.output
    assert captured["path"] == plan
    assert captured["skip_quality"] is True
    assert captured["skip_quantize"] is True
    assert captured["progress"] is not None


def test_unknown_command_fails() -> None:
    result = runner.invoke(app, ["nonexistent-cmd"])
    assert result.exit_code != 0
