"""Smoke tests for the lmbench CLI surface."""

from __future__ import annotations

from pathlib import Path

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


def test_compare_stub_runs(tmp_path: Path) -> None:
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


def test_run_stub_runs(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("plan: {}\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--plan", str(plan)])
    assert result.exit_code == 0


def test_unknown_command_fails() -> None:
    result = runner.invoke(app, ["nonexistent-cmd"])
    assert result.exit_code != 0
