"""Context manager for clean vLLM server startup and teardown.

`serve_model(entry)` starts a server, waits until it's ready, and yields
its `ServerHandle`. On exit (success or exception) the server's process
group is terminated. If startup fails, the handle is also stopped before
the exception propagates so no orphan processes are left behind.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lmbench.config import ModelEntry

from .vllm_server import (
    ServerHandle,
    start_vllm_server,
    stop_vllm_server,
    wait_for_ready,
)


@contextmanager
def serve_model(
    entry: ModelEntry,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    executable: str = "vllm",
    env: dict[str, str] | None = None,
    log_dir: Path | None = None,
    stream_logs: bool = False,
    startup_timeout_s: float = 600.0,
    shutdown_timeout_s: float = 30.0,
) -> Iterator[ServerHandle]:
    """Start a vLLM server, yield its handle, then tear it down.

    The teardown happens whether the body raises or returns. If
    `wait_for_ready` itself raises, the spawned process is still
    terminated before we propagate the exception.
    """
    handle = start_vllm_server(
        entry,
        host=host,
        port=port,
        executable=executable,
        env=env,
        log_dir=log_dir,
        stream_logs=stream_logs,
    )
    try:
        wait_for_ready(
            handle.base_url,
            handle.served_model_name,
            timeout_s=startup_timeout_s,
        )
        yield handle
    finally:
        stop_vllm_server(handle, timeout_s=shutdown_timeout_s)
