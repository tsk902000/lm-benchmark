"""Capture run-time environment metadata to `env.json`.

Each pipeline run snapshots the host (OS, kernel, Python, packages, GPU
state, git SHA) so we can reconcile perf deltas against driver / runtime
drift after the fact.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EnvCapture:
    """Snapshot of the host environment at run start."""

    captured_at_unix: float
    os: str
    kernel: str
    python: str
    git_sha: str | None
    git_dirty: bool
    nvidia_smi: str | None
    packages: dict[str, str | None] = field(default_factory=dict)


_PACKAGES_OF_INTEREST = (
    "vllm",
    "nvidia-modelopt",
    "modelopt",
    "transformers",
    "torch",
    "lm-eval",
    "datasets",
    "pynvml",
    "lmbench",
)


def _safe_run(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=15.0, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _git_sha() -> tuple[str | None, bool]:
    sha = _safe_run(["git", "rev-parse", "HEAD"])
    if sha is None:
        return None, False
    status = _safe_run(["git", "status", "--porcelain"])
    return sha, bool(status)


def _nvidia_smi() -> str | None:
    return _safe_run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ]
    )


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES_OF_INTEREST:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def capture_environment() -> EnvCapture:
    """Snapshot the current host's environment."""
    sha, dirty = _git_sha()
    return EnvCapture(
        captured_at_unix=time.time(),
        os=platform.system(),
        kernel=platform.release(),
        python=sys.version.split()[0],
        git_sha=sha,
        git_dirty=dirty,
        nvidia_smi=_nvidia_smi(),
        packages=_package_versions(),
    )


def write_env(capture: EnvCapture, output_path: Path) -> Path:
    """Persist a captured `EnvCapture` as JSON; return the written path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(capture), indent=2), encoding="utf-8")
    return output_path


def capture_to_path(output_path: Path) -> EnvCapture:
    """Convenience: capture and write in one step."""
    capture = capture_environment()
    write_env(capture, output_path)
    return capture
