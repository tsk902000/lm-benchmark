"""Quick post-quantization sanity probe.

Loads the quantized checkpoint into vLLM, sends a fixed prompt, and
rejects the run early if the output is empty / degenerate. Catching
NVFP4 export failures here avoids burning hours on a broken benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from lmbench.config import ModelEntry, VLLMArgs
from lmbench.serve import serve_model

from .modelopt_nvfp4 import QuantizedCheckpoint


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a verification probe."""

    ok: bool
    completion: str
    reason: str = ""


_DEFAULT_PROMPT = "The capital of France is"
_MIN_NON_WHITESPACE = 1


def classify_completion(text: str) -> tuple[bool, str]:
    """Decide whether a completion looks healthy."""
    stripped = text.strip()
    if len(stripped) < _MIN_NON_WHITESPACE:
        return False, "completion is empty"
    if all(c == stripped[0] for c in stripped) and len(stripped) > 4:
        return False, f"degenerate repeating output: {stripped[:32]!r}"
    return True, ""


def build_quant_entry(
    checkpoint: QuantizedCheckpoint, base_entry: ModelEntry
) -> ModelEntry:
    """Construct a `ModelEntry` that points vLLM at the quantized checkpoint."""
    return ModelEntry(
        name=f"{base_entry.name}-quant",
        hf_id=checkpoint.vllm_id,
        served_model_name=f"{base_entry.name}-quant",
        vllm=VLLMArgs(
            tensor_parallel_size=base_entry.vllm.tensor_parallel_size,
            max_model_len=base_entry.vllm.max_model_len,
            dtype=base_entry.vllm.dtype,
            gpu_memory_utilization=base_entry.vllm.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=base_entry.vllm.trust_remote_code,
            quantization="modelopt_fp4" if checkpoint.method == "nvfp4" else None,
        ),
    )


def verify_checkpoint(
    *,
    checkpoint: QuantizedCheckpoint,
    base_entry: ModelEntry,
    prompt: str = _DEFAULT_PROMPT,
    max_tokens: int = 16,
    port: int = 8124,
    startup_timeout_s: float = 600.0,
) -> VerifyResult:
    """Boot the quantized checkpoint in vLLM, send one prompt, classify."""
    quant_entry = build_quant_entry(checkpoint, base_entry)
    with serve_model(
        quant_entry, port=port, startup_timeout_s=startup_timeout_s
    ) as handle:
        resp = httpx.post(
            f"{handle.base_url}/v1/completions",
            json={
                "model": handle.served_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            return VerifyResult(
                ok=False,
                completion="",
                reason=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return VerifyResult(ok=False, completion="", reason="no choices in response")
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if not isinstance(text, str):
            return VerifyResult(
                ok=False, completion="", reason="non-string completion"
            )
    ok, why = classify_completion(text)
    return VerifyResult(ok=ok, completion=text, reason=why)
