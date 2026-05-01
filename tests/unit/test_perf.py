"""Unit tests for `lmbench.bench.perf`."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from lmbench.bench import Prompt, run_workload
from lmbench.config import WorkloadSpec


def _spec(**overrides: object) -> WorkloadSpec:
    base: dict[str, object] = {
        "name": "w",
        "kind": "random",
        "num_prompts": 5,
        "input_len": 32,
        "output_len": 8,
        "warmup_prompts": 0,
        "concurrency": (1,),
    }
    base.update(overrides)
    return WorkloadSpec.model_validate(base)


class _FakeStreamResponse:
    def __init__(self, status_code: int, lines: list[str]) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    """Minimal stand-in for `httpx.AsyncClient` used by the perf driver."""

    def __init__(
        self,
        chunks_per_request: list[list[str]] | None = None,
        status_code: int = 200,
    ) -> None:
        self._chunks = chunks_per_request or []
        self._status = status_code
        self._call_index = 0

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    @asynccontextmanager
    async def stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncIterator[_FakeStreamResponse]:
        del method, url, kwargs
        idx = self._call_index
        self._call_index += 1
        lines = self._chunks[idx] if idx < len(self._chunks) else []
        yield _FakeStreamResponse(self._status, lines)


def _ok_chunks(n_tokens: int) -> list[str]:
    out = [
        f"data: {json.dumps({'choices': [{'text': f'tok{i}'}]})}\n"
        for i in range(n_tokens)
    ]
    out.append("data: [DONE]\n")
    return out


def _patch_async_client(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], _FakeAsyncClient],
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", factory)


def test_run_workload_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_ok_chunks(3) for _ in range(5)]
    _patch_async_client(monkeypatch, lambda: _FakeAsyncClient(chunks))
    prompts = tuple(Prompt(text=f"p{i}", expected_output_tokens=3) for i in range(5))
    result = run_workload(
        base_url="http://x",
        served_model_name="m",
        workload=_spec(num_prompts=5, warmup_prompts=0),
        concurrency=2,
        prompts=prompts,
    )
    assert result.summary.n_requests == 5
    assert result.summary.n_success == 5
    assert result.concurrency == 2
    assert result.workload_name == "w"
    assert all(s.output_tokens == 3 for s in result.samples)
    assert all(s.success for s in result.samples)


def test_run_workload_warmup_split(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_ok_chunks(2) for _ in range(8)]
    _patch_async_client(monkeypatch, lambda: _FakeAsyncClient(chunks))
    prompts = tuple(Prompt(text=f"p{i}", expected_output_tokens=2) for i in range(8))
    result = run_workload(
        base_url="http://x",
        served_model_name="m",
        workload=_spec(num_prompts=8, warmup_prompts=3),
        concurrency=1,
        prompts=prompts,
    )
    # 3 warmup + 5 measured: only the 5 are recorded
    assert result.summary.n_requests == 5
    assert len(result.samples) == 5


def test_run_workload_handles_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_async_client(monkeypatch, lambda: _FakeAsyncClient([], status_code=503))
    prompts = (Prompt(text="p", expected_output_tokens=2),)
    result = run_workload(
        base_url="http://x",
        served_model_name="m",
        workload=_spec(num_prompts=1, warmup_prompts=0),
        concurrency=1,
        prompts=prompts,
    )
    assert result.summary.n_success == 0
    assert all(not s.success for s in result.samples)


def test_run_workload_handles_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenClient(_FakeAsyncClient):
        @asynccontextmanager
        async def stream(
            self, method: str, url: str, **kwargs: Any
        ) -> AsyncIterator[_FakeStreamResponse]:
            del method, url, kwargs
            raise httpx.ConnectError("refused")
            yield  # pragma: no cover

    _patch_async_client(monkeypatch, _BrokenClient)
    prompts = (Prompt(text="p", expected_output_tokens=2),)
    result = run_workload(
        base_url="http://x",
        served_model_name="m",
        workload=_spec(num_prompts=1, warmup_prompts=0),
        concurrency=1,
        prompts=prompts,
    )
    assert result.summary.n_success == 0


def test_run_workload_skips_malformed_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        [
            "data: not-json\n",
            f"data: {json.dumps({'choices': [{'text': 'tok0'}]})}\n",
            "data: {\"no_choices\": true}\n",
            f"data: {json.dumps({'choices': [{'text': ''}]})}\n",
            f"data: {json.dumps({'choices': [{'text': 'tok1'}]})}\n",
            "data: [DONE]\n",
        ]
    ]
    _patch_async_client(monkeypatch, lambda: _FakeAsyncClient(chunks))
    prompts = (Prompt(text="p", expected_output_tokens=2),)
    result = run_workload(
        base_url="http://x",
        served_model_name="m",
        workload=_spec(num_prompts=1, warmup_prompts=0),
        concurrency=1,
        prompts=prompts,
    )
    assert result.summary.n_success == 1
    assert result.samples[0].output_tokens == 2


def test_run_workload_validates_inputs() -> None:
    spec = _spec(num_prompts=1, warmup_prompts=0)
    with pytest.raises(ValueError, match="non-empty"):
        run_workload(
            base_url="http://x",
            served_model_name="m",
            workload=spec,
            concurrency=1,
            prompts=(),
        )
    with pytest.raises(ValueError, match="concurrency"):
        run_workload(
            base_url="http://x",
            served_model_name="m",
            workload=spec,
            concurrency=0,
            prompts=(Prompt(text="p", expected_output_tokens=1),),
        )


def test_run_workload_warmup_consumes_all_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_ok_chunks(1) for _ in range(2)]
    _patch_async_client(monkeypatch, lambda: _FakeAsyncClient(chunks))
    prompts = (Prompt(text="p", expected_output_tokens=1),) * 2
    spec = WorkloadSpec.model_validate(
        {
            "name": "w",
            "kind": "random",
            "num_prompts": 5,
            "input_len": 32,
            "output_len": 8,
            "warmup_prompts": 5,
            "concurrency": (1,),
        }
    )
    with pytest.raises(ValueError, match="consumes them all"):
        run_workload(
            base_url="http://x",
            served_model_name="m",
            workload=spec,
            concurrency=1,
            prompts=prompts,
        )
