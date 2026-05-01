"""Unit tests for the health-probe layer in `lmbench.serve.vllm_server`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lmbench.serve import is_healthy, lists_model, wait_for_ready
from lmbench.serve.vllm_server import ServerNotReady


class _FakeResponse:
    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _patch_get(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str], _FakeResponse],
) -> None:
    def fake_get(url: str, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        return handler(url)

    monkeypatch.setattr(httpx, "get", fake_get)


def test_is_healthy_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, lambda _u: _FakeResponse(200))
    assert is_healthy("http://x") is True


def test_is_healthy_false_on_500(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, lambda _u: _FakeResponse(500))
    assert is_healthy("http://x") is False


def test_is_healthy_false_on_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raiser(*_a: Any, **_kw: Any) -> _FakeResponse:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", raiser)
    assert is_healthy("http://x") is False


def test_lists_model_true_when_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(
        monkeypatch,
        lambda _u: _FakeResponse(200, {"data": [{"id": "m"}, {"id": "n"}]}),
    )
    assert lists_model("http://x", "m") is True


def test_lists_model_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(
        monkeypatch,
        lambda _u: _FakeResponse(200, {"data": [{"id": "n"}]}),
    )
    assert lists_model("http://x", "m") is False


def test_lists_model_false_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, lambda _u: _FakeResponse(200))
    assert lists_model("http://x", "m") is False


def test_lists_model_false_on_non_dict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, lambda _u: _FakeResponse(200, ["not", "a", "dict"]))
    assert lists_model("http://x", "m") is False


def test_lists_model_false_on_non_list_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, lambda _u: _FakeResponse(200, {"data": "oops"}))
    assert lists_model("http://x", "m") is False


def test_lists_model_false_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, lambda _u: _FakeResponse(503, {"data": []}))
    assert lists_model("http://x", "m") is False


def test_lists_model_false_on_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raiser(*_a: Any, **_kw: Any) -> _FakeResponse:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", raiser)
    assert lists_model("http://x", "m") is False


def test_wait_for_ready_succeeds_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(url: str) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 5:
            if url.endswith("/health"):
                return _FakeResponse(503)
            return _FakeResponse(200, {"data": []})
        if url.endswith("/health"):
            return _FakeResponse(200)
        return _FakeResponse(200, {"data": [{"id": "m"}]})

    _patch_get(monkeypatch, handler)
    wait_for_ready("http://x", "m", timeout_s=60.0, sleep=lambda _s: None)
    assert calls["n"] >= 5


def test_wait_for_ready_times_out() -> None:
    # Real httpx call to a closed port — should fail fast and time out.
    with pytest.raises(ServerNotReady):
        wait_for_ready(
            "http://127.0.0.1:1",
            "m",
            timeout_s=0.05,
            sleep=lambda _s: None,
        )
