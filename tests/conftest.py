"""Shared pytest fixtures and hooks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.gpu (requires CUDA hardware).",
    )
    parser.addoption(
        "--blackwell",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.blackwell (requires Blackwell GPU).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU-marked tests unless explicitly requested."""
    run_gpu = config.getoption("--gpu") or os.environ.get("LMBENCH_GPU") == "1"
    run_blackwell = (
        config.getoption("--blackwell") or os.environ.get("LMBENCH_BLACKWELL") == "1"
    )
    skip_gpu = pytest.mark.skip(reason="needs --gpu or LMBENCH_GPU=1")
    skip_blackwell = pytest.mark.skip(reason="needs --blackwell or LMBENCH_BLACKWELL=1")
    for item in items:
        if not run_gpu and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if not run_blackwell and "blackwell" in item.keywords:
            item.add_marker(skip_blackwell)


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """A throwaway results directory mirroring the production layout."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def empty_yaml(tmp_path: Path) -> Path:
    """An empty YAML config file for stub-CLI tests."""
    p = tmp_path / "config.yaml"
    p.write_text("models: []\n", encoding="utf-8")
    return p
