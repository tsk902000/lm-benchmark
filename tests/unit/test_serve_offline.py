"""Unit tests for `lmbench.serve.vllm_offline.build_llm_kwargs`."""

from __future__ import annotations

from typing import Any

from lmbench.config import ModelEntry, VLLMArgs
from lmbench.serve import build_llm_kwargs


def _entry(**vllm_overrides: Any) -> ModelEntry:
    return ModelEntry(
        name="opt",
        hf_id="facebook/opt-125m",
        vllm=VLLMArgs(**vllm_overrides),
    )


def test_build_llm_kwargs_minimal() -> None:
    kw = build_llm_kwargs(_entry())
    assert kw["model"] == "facebook/opt-125m"
    assert kw["tensor_parallel_size"] == 1
    assert kw["dtype"] == "auto"
    assert "max_model_len" not in kw
    assert "kv_cache_dtype" not in kw
    assert "quantization" not in kw
    assert "revision" not in kw


def test_build_llm_kwargs_with_overrides() -> None:
    entry = ModelEntry(
        name="m",
        hf_id="x/y",
        revision="v1",
        vllm=VLLMArgs(
            tensor_parallel_size=2,
            max_model_len=4096,
            dtype="bfloat16",
            kv_cache_dtype="fp8",
            quantization="modelopt_fp4",
            enforce_eager=True,
        ),
    )
    kw = build_llm_kwargs(entry)
    assert kw["tensor_parallel_size"] == 2
    assert kw["max_model_len"] == 4096
    assert kw["dtype"] == "bfloat16"
    assert kw["kv_cache_dtype"] == "fp8"
    assert kw["quantization"] == "modelopt_fp4"
    assert kw["enforce_eager"] is True
    assert kw["revision"] == "v1"
