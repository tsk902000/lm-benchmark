"""Thin wrapper around `vllm.LLM` for offline (in-process) generation.

The `vllm` package only ships Linux/CUDA wheels and lives behind the
`[gpu]` optional extra. This module imports it lazily so the harness
remains importable on dev boxes without GPU dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lmbench.config import ModelEntry

if TYPE_CHECKING:
    from vllm import LLM


def build_llm_kwargs(entry: ModelEntry) -> dict[str, Any]:
    """Map a `ModelEntry` to keyword args for `vllm.LLM(...)`."""
    kwargs: dict[str, Any] = {
        "model": entry.hf_id,
        "tensor_parallel_size": entry.vllm.tensor_parallel_size,
        "pipeline_parallel_size": entry.vllm.pipeline_parallel_size,
        "gpu_memory_utilization": entry.vllm.gpu_memory_utilization,
        "dtype": entry.vllm.dtype,
        "enforce_eager": entry.vllm.enforce_eager,
        "trust_remote_code": entry.vllm.trust_remote_code,
        "enable_prefix_caching": entry.vllm.enable_prefix_caching,
    }
    if entry.revision is not None:
        kwargs["revision"] = entry.revision
    if entry.vllm.max_model_len is not None:
        kwargs["max_model_len"] = entry.vllm.max_model_len
    if entry.vllm.kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = entry.vllm.kv_cache_dtype
    if entry.vllm.quantization is not None:
        kwargs["quantization"] = entry.vllm.quantization
    return kwargs


def load_offline_engine(entry: ModelEntry) -> LLM:
    """Instantiate an in-process `vllm.LLM` for the given model entry.

    Raises `ImportError` (with an actionable message) if the `[gpu]` extra
    is not installed.
    """
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - exercised via mock in unit tests
        raise ImportError(
            "vllm is not installed. Install the [gpu] extra on a CUDA host: "
            "`uv sync --extra all`."
        ) from exc
    return LLM(**build_llm_kwargs(entry))
