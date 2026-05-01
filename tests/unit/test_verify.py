"""Unit tests for `lmbench.quantize.verify`."""

from __future__ import annotations

from pathlib import Path

from lmbench.config import ModelEntry, VLLMArgs
from lmbench.quantize import (
    QuantizedCheckpoint,
    build_quant_entry,
    classify_completion,
)


def test_classify_rejects_empty() -> None:
    ok, why = classify_completion("")
    assert ok is False
    assert "empty" in why


def test_classify_rejects_repeating() -> None:
    ok, why = classify_completion("aaaaaaaaa")
    assert ok is False
    assert "degenerate" in why


def test_classify_accepts_normal_completion() -> None:
    ok, why = classify_completion(" Paris and is the largest city.")
    assert ok is True
    assert why == ""


def test_classify_accepts_short_unique_strings() -> None:
    ok, _ = classify_completion("Hi")
    assert ok is True


def test_build_quant_entry_carries_args() -> None:
    base = ModelEntry(
        name="opt-125m",
        hf_id="facebook/opt-125m",
        vllm=VLLMArgs(
            tensor_parallel_size=2,
            max_model_len=4096,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        ),
    )
    ckpt = QuantizedCheckpoint(
        output_dir=Path("/data/q/opt-125m"),
        method="nvfp4",
        source_hf_id="facebook/opt-125m",
    )
    entry = build_quant_entry(ckpt, base)
    assert entry.hf_id == "/data/q/opt-125m"
    assert entry.served_model_name == "opt-125m-quant"
    assert entry.vllm.tensor_parallel_size == 2
    assert entry.vllm.max_model_len == 4096
    assert entry.vllm.dtype == "bfloat16"
    assert entry.vllm.quantization == "modelopt_fp4"
    assert entry.vllm.enforce_eager is True


def test_build_quant_entry_skips_quantization_for_other_methods() -> None:
    base = ModelEntry(name="m", hf_id="x/y")
    ckpt = QuantizedCheckpoint(
        output_dir=Path("/p"), method="nvfp4_llmcompressor", source_hf_id="x/y"
    )
    entry = build_quant_entry(ckpt, base)
    assert entry.vllm.quantization is None
