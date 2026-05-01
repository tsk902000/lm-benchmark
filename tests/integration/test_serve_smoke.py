"""End-to-end smoke test against `facebook/opt-125m`.

This test only runs when:
- The host has a CUDA GPU (`@pytest.mark.gpu`).
- The `vllm` executable is on PATH (i.e. the [gpu] extra is installed).

It spawns a real `vllm serve` subprocess, performs one OpenAI-compatible
`POST /v1/completions` request, and tears the server down via the
`serve_model` context manager.
"""

from __future__ import annotations

import shutil

import httpx
import pytest

from lmbench.config import ModelEntry, VLLMArgs
from lmbench.serve import serve_model

pytestmark = pytest.mark.gpu


@pytest.fixture
def opt_125m_entry() -> ModelEntry:
    return ModelEntry(
        name="opt-125m-smoke",
        hf_id="facebook/opt-125m",
        vllm=VLLMArgs(
            tensor_parallel_size=1,
            max_model_len=2048,
            dtype="auto",
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        ),
    )


def _vllm_available() -> bool:
    return shutil.which("vllm") is not None


@pytest.mark.skipif(not _vllm_available(), reason="vllm executable not on PATH")
def test_serve_opt_125m_round_trip(opt_125m_entry: ModelEntry) -> None:
    with serve_model(
        opt_125m_entry,
        port=8123,
        startup_timeout_s=600.0,
        shutdown_timeout_s=60.0,
    ) as handle:
        resp = httpx.post(
            f"{handle.base_url}/v1/completions",
            json={
                "model": handle.served_model_name,
                "prompt": "Hello",
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        choices = data.get("choices")
        assert isinstance(choices, list) and choices
        assert isinstance(choices[0].get("text"), str)
