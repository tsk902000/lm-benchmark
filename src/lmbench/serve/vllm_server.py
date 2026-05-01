"""Subprocess wrapper for `vllm serve`.

Three responsibilities:

1. Build a CLI argv from `ModelEntry` + `VLLMArgs`.
2. Start a `vllm serve` subprocess in its own process group so all child
   workers can be reliably killed via `os.killpg`.
3. Health-probe `/health` and `/v1/models` with `tenacity` backoff until
   the server is ready (or the timeout is reached).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)

from lmbench.config import ModelEntry


class ServerNotReady(RuntimeError):
    """Raised while a vLLM server has not yet responded as healthy."""


@dataclass(frozen=True)
class ServerHandle:
    """Handle to a running vLLM server process."""

    process: subprocess.Popen[bytes]
    host: str
    port: int
    served_model_name: str
    log_paths: tuple[Path, Path] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> int:
        return self.process.pid


def served_model_name(entry: ModelEntry) -> str:
    """Return the value passed to `--served-model-name`."""
    return entry.served_model_name or entry.name


def build_serve_args(
    entry: ModelEntry,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    executable: str = "vllm",
) -> list[str]:
    """Build the `vllm serve` CLI argv for a model entry."""
    args = [executable, "serve", entry.hf_id]
    if entry.revision is not None:
        args += ["--revision", entry.revision]
    args += ["--served-model-name", served_model_name(entry)]
    args += ["--host", host, "--port", str(port)]
    args += ["--tensor-parallel-size", str(entry.vllm.tensor_parallel_size)]
    if entry.vllm.pipeline_parallel_size > 1:
        args += ["--pipeline-parallel-size", str(entry.vllm.pipeline_parallel_size)]
    if entry.vllm.max_model_len is not None:
        args += ["--max-model-len", str(entry.vllm.max_model_len)]
    args += ["--gpu-memory-utilization", str(entry.vllm.gpu_memory_utilization)]
    args += ["--dtype", entry.vllm.dtype]
    if entry.vllm.kv_cache_dtype != "auto":
        args += ["--kv-cache-dtype", entry.vllm.kv_cache_dtype]
    if entry.vllm.quantization is not None:
        args += ["--quantization", entry.vllm.quantization]
    if entry.vllm.enforce_eager:
        args += ["--enforce-eager"]
    if entry.vllm.trust_remote_code:
        args += ["--trust-remote-code"]
    if not entry.vllm.enable_prefix_caching:
        args += ["--no-enable-prefix-caching"]
    for key, value in sorted(entry.vllm.extra_args.items()):
        flag = key if key.startswith("--") else f"--{key}"
        args += [flag, value]
    return args


def start_vllm_server(
    entry: ModelEntry,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    executable: str = "vllm",
    env: dict[str, str] | None = None,
    log_dir: Path | None = None,
) -> ServerHandle:
    """Spawn `vllm serve` as a subprocess in its own process group.

    The new process group is what lets `stop_vllm_server` SIGKILL all
    children (vLLM workers) at once if termination is required.
    """
    if executable == "vllm":
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                "'vllm' not found on PATH; install the [gpu] extra "
                "(`uv sync --extra all`) on a CUDA host"
            )
    else:
        resolved = executable

    argv = build_serve_args(entry, host=host, port=port, executable=resolved)

    stdout_target: int = subprocess.DEVNULL
    stderr_target: int = subprocess.DEVNULL
    log_paths: tuple[Path, Path] | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        name = served_model_name(entry).replace("/", "_")
        out_path = log_dir / f"{name}.stdout.log"
        err_path = log_dir / f"{name}.stderr.log"
        stdout_target = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        stderr_target = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        log_paths = (out_path, err_path)

    merged_env = {**os.environ, **(env or {})}
    start_new_session = sys.platform != "win32"
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        argv,
        stdout=stdout_target,
        stderr=stderr_target,
        env=merged_env,
        start_new_session=start_new_session,
    )
    return ServerHandle(
        process=proc,
        host=host,
        port=port,
        served_model_name=served_model_name(entry),
        log_paths=log_paths,
    )


def is_healthy(base_url: str, *, timeout_s: float = 2.0) -> bool:
    """Single-shot probe of `/health`. Swallows network errors."""
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout_s)
    except httpx.RequestError:
        return False
    return r.status_code == 200


def lists_model(base_url: str, model_name: str, *, timeout_s: float = 2.0) -> bool:
    """Single-shot probe of `/v1/models`; True if the model is listed."""
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=timeout_s)
    except httpx.RequestError:
        return False
    if r.status_code != 200:
        return False
    try:
        payload = r.json()
    except ValueError:
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return False
    return any(isinstance(m, dict) and m.get("id") == model_name for m in data)


def wait_for_ready(
    base_url: str,
    model_name: str,
    *,
    timeout_s: float = 300.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until `/health` is 200 AND `/v1/models` lists the model.

    Raises `ServerNotReady` if the deadline passes. The `sleep` parameter
    is exposed so tests can inject a no-op and avoid wallclock waits.
    """
    for attempt in Retrying(
        stop=stop_after_delay(timeout_s),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
        retry=retry_if_exception_type(ServerNotReady),
        reraise=True,
        sleep=sleep,
    ):
        with attempt:
            if not is_healthy(base_url):
                raise ServerNotReady(f"{base_url}/health not yet 200")
            if not lists_model(base_url, model_name):
                raise ServerNotReady(
                    f"{base_url}/v1/models does not list {model_name!r} yet"
                )


def stop_vllm_server(handle: ServerHandle, *, timeout_s: float = 30.0) -> int:
    """Terminate the process group; SIGKILL after `timeout_s` if needed.

    Returns the final exit code (or -9 on SIGKILL).
    """
    proc = handle.process
    if proc.poll() is not None:
        return int(proc.returncode)

    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    else:
        proc.terminate()

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return int(proc.returncode)
        time.sleep(0.1)

    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    else:
        proc.kill()

    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5.0)
    return int(proc.returncode if proc.returncode is not None else -9)
