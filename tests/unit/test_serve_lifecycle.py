"""Unit tests for the serve lifecycle (start, stop, context manager)."""

from __future__ import annotations

import signal
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from lmbench.config import ModelEntry, VLLMArgs
from lmbench.serve import ServerHandle, serve_model
from lmbench.serve import lifecycle as lc
from lmbench.serve import vllm_server as vs


class _FakePopen:
    """Stand-in for `subprocess.Popen` covering the surface used by lmbench."""

    instances: ClassVar[list[_FakePopen]] = []

    def __init__(self, argv: list[str], **_: Any) -> None:
        self.argv = argv
        self.pid = 12345 + len(_FakePopen.instances)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.poll_calls = 0
        # If True, terminate() leaves the process "running" so the timeout fires.
        self.refuse_terminate = False
        _FakePopen.instances.append(self)

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.refuse_terminate:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _FakePopen.instances = []
    monkeypatch.setattr(vs.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(vs.shutil, "which", lambda x: x)
    monkeypatch.setattr(vs, "wait_for_ready", lambda *_a, **_kw: None)
    monkeypatch.setattr(lc, "wait_for_ready", lambda *_a, **_kw: None)
    monkeypatch.setattr(vs.sys, "platform", "linux")
    monkeypatch.setattr(vs.signal, "SIGKILL", 9, raising=False)

    def fake_killpg(pgid: int, sig: int) -> None:
        for fake in _FakePopen.instances:
            if fake.pid != pgid:
                continue
            if sig == signal.SIGTERM:
                fake.terminated = True
                if not fake.refuse_terminate:
                    fake.returncode = 0
            elif sig == signal.SIGKILL:
                fake.killed = True
                fake.returncode = -9
            return

    monkeypatch.setattr(vs.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(vs.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(vs.time, "sleep", lambda _s: None)
    yield


def _entry() -> ModelEntry:
    return ModelEntry(
        name="opt",
        hf_id="facebook/opt-125m",
        vllm=VLLMArgs(enforce_eager=True),
    )


def test_start_vllm_server_returns_handle() -> None:
    handle = vs.start_vllm_server(_entry(), port=9999)
    assert isinstance(handle, ServerHandle)
    assert handle.port == 9999
    assert handle.served_model_name == "opt"
    assert handle.base_url == "http://127.0.0.1:9999"
    assert handle.pid > 0
    assert _FakePopen.instances[0].argv[:3] == ["vllm", "serve", "facebook/opt-125m"]


def test_start_vllm_server_writes_log_paths(tmp_path: Path) -> None:
    handle = vs.start_vllm_server(_entry(), log_dir=tmp_path / "logs")
    assert handle.log_paths is not None
    out, err = handle.log_paths
    assert out.exists()
    assert err.exists()
    assert out.name == "opt.stdout.log"


def test_start_vllm_server_streams_logs_to_parent() -> None:
    handle = vs.start_vllm_server(_entry(), stream_logs=True)
    assert handle.log_paths is None
    fake = _FakePopen.instances[0]
    assert fake.argv[:3] == ["vllm", "serve", "facebook/opt-125m"]


def test_stop_vllm_server_terminates_cleanly() -> None:
    handle = vs.start_vllm_server(_entry())
    rc = vs.stop_vllm_server(handle, timeout_s=1.0)
    assert rc == 0
    fake = _FakePopen.instances[0]
    assert fake.terminated is True
    assert fake.killed is False


def test_stop_vllm_server_kills_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = vs.start_vllm_server(_entry())
    fake = _FakePopen.instances[0]
    fake.refuse_terminate = True
    times = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(vs.time, "monotonic", lambda: next(times))
    rc = vs.stop_vllm_server(handle, timeout_s=1.0)
    assert fake.terminated is True
    assert fake.killed is True
    assert rc == -9


def test_stop_vllm_server_noop_if_already_exited() -> None:
    handle = vs.start_vllm_server(_entry())
    fake = _FakePopen.instances[0]
    fake.returncode = 0
    rc = vs.stop_vllm_server(handle, timeout_s=1.0)
    assert rc == 0
    assert fake.terminated is False


def test_serve_model_context_manager_tears_down_on_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"), serve_model(_entry(), port=9000):
        raise RuntimeError("boom")
    fake = _FakePopen.instances[0]
    assert fake.terminated is True


def test_serve_model_context_manager_yields_handle() -> None:
    with serve_model(_entry(), port=9001) as handle:
        assert handle.port == 9001
        assert handle.served_model_name == "opt"
    fake = _FakePopen.instances[0]
    assert fake.terminated is True


def test_start_vllm_server_raises_when_executable_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vs.shutil, "which", lambda _x: None)
    with pytest.raises(FileNotFoundError, match="vllm"):
        vs.start_vllm_server(_entry())


def test_stop_vllm_server_handles_nonexistent_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = vs.start_vllm_server(_entry())
    fake = _FakePopen.instances[0]
    fake.refuse_terminate = True

    def raise_lookup(*_a: Any, **_kw: Any) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(vs.os, "killpg", raise_lookup)
    times = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(vs.time, "monotonic", lambda: next(times))
    rc = vs.stop_vllm_server(handle, timeout_s=0.1)
    assert rc == -9
    assert fake.killed is True


def test_stop_vllm_server_handles_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = vs.start_vllm_server(_entry())
    fake = _FakePopen.instances[0]
    fake.refuse_terminate = True

    def wait_raises(timeout: float | None = None) -> int:
        del timeout
        raise subprocess.TimeoutExpired(cmd="vllm", timeout=5.0)

    monkeypatch.setattr(fake, "wait", wait_raises)
    times = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(vs.time, "monotonic", lambda: next(times))
    rc = vs.stop_vllm_server(handle, timeout_s=0.1)
    assert rc == -9
