"""Workload generators: random, long-context, and ShareGPT.

Each generator returns a tuple of `Prompt` records. All sampling uses a
seeded `random.Random` so a given `(WorkloadSpec, seed)` produces the
exact same prompt sequence across runs — a hard requirement for the
"reproducible harness" charter in HANDOFF.md.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

from lmbench.config import WorkloadSpec


@dataclass(frozen=True)
class Prompt:
    """One workload entry."""

    text: str
    """The prompt text fed to the model (already chat-templated if needed)."""

    expected_output_tokens: int
    """Target `max_tokens` for the request. The driver passes this through."""


_WORDS = [
    "model", "serving", "benchmark", "token", "throughput", "latency",
    "context", "attention", "decoder", "layer", "kernel", "cuda", "memory",
    "cache", "prefix", "completion", "request", "concurrency", "batch",
    "warmup", "steady", "state", "percentile", "profile", "distribution",
    "blackwell", "tensor", "core", "quantization", "calibration",
    "accuracy", "quality",
]  # fmt: skip


def _random_words(rng: random.Random, n: int) -> str:
    """Generate `n` whitespace-separated tokens from a small lexicon.

    Token count is approximate (real BPE may merge or split). The driver
    treats `expected_output_tokens` as authoritative for `max_tokens`.
    """
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _random_filler(rng: random.Random, target_tokens: int) -> str:
    """Fill out roughly `target_tokens` tokens with a mix of words and noise.

    Uses a per-prompt salt so prompts of the same length are not byte-
    identical (which would let prefix caching distort perf measurements).
    """
    salt = "".join(rng.choices(string.ascii_lowercase, k=8))
    body = _random_words(rng, max(target_tokens - 1, 1))
    return f"{salt} {body}"


def gen_random(spec: WorkloadSpec) -> tuple[Prompt, ...]:
    """Random short-form prompts with controlled input/output lengths."""
    if spec.kind != "random":
        raise ValueError(f"gen_random requires kind='random', got {spec.kind!r}")
    if spec.input_len is None or spec.output_len is None:
        raise ValueError("random workload requires input_len and output_len")
    rng = random.Random(spec.seed)
    return tuple(
        Prompt(
            text=_random_filler(rng, spec.input_len),
            expected_output_tokens=spec.output_len,
        )
        for _ in range(spec.num_prompts)
    )


def gen_longctx(
    spec: WorkloadSpec, *, max_total_tokens: int | None = None
) -> tuple[Prompt, ...]:
    """Long-context workload with a token-budget guard.

    `max_total_tokens` (if provided) caps `input_len + output_len`. This
    mirrors the runtime `max_model_len` so a misconfigured workload
    cannot OOM the server.
    """
    if spec.kind != "longctx":
        raise ValueError(f"gen_longctx requires kind='longctx', got {spec.kind!r}")
    if spec.input_len is None or spec.output_len is None:
        raise ValueError("longctx workload requires input_len and output_len")
    if (
        max_total_tokens is not None
        and spec.input_len + spec.output_len > max_total_tokens
    ):
        raise ValueError(
            f"longctx input_len ({spec.input_len}) + output_len ({spec.output_len}) "
            f"exceeds max_total_tokens ({max_total_tokens})"
        )
    rng = random.Random(spec.seed)
    return tuple(
        Prompt(
            text=_random_filler(rng, spec.input_len),
            expected_output_tokens=spec.output_len,
        )
        for _ in range(spec.num_prompts)
    )


def _extract_sharegpt_text(record: object) -> str | None:
    """Pull the first human turn out of a ShareGPT-style record.

    Returns None for malformed entries (caller filters those out).
    """
    if not isinstance(record, dict):
        return None
    convs = record.get("conversations")
    if not isinstance(convs, list):
        return None
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role")
        value = turn.get("value") or turn.get("content")
        if role in ("human", "user") and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def gen_sharegpt(
    spec: WorkloadSpec,
    *,
    dataset_path: Path,
    default_output_tokens: int = 256,
) -> tuple[Prompt, ...]:
    """Sample real human turns from a ShareGPT-style JSON file."""
    if spec.kind != "sharegpt":
        raise ValueError(f"gen_sharegpt requires kind='sharegpt', got {spec.kind!r}")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{dataset_path}: expected a JSON list at the top level")
    candidates = [t for t in (_extract_sharegpt_text(r) for r in raw) if t]
    if not candidates:
        raise ValueError(f"{dataset_path}: no usable conversations after filtering")
    rng = random.Random(spec.seed)
    chosen = [rng.choice(candidates) for _ in range(spec.num_prompts)]
    out = spec.output_len if spec.output_len is not None else default_output_tokens
    return tuple(Prompt(text=t, expected_output_tokens=out) for t in chosen)


def generate(
    spec: WorkloadSpec,
    *,
    sharegpt_path: Path | None = None,
    max_total_tokens: int | None = None,
) -> tuple[Prompt, ...]:
    """Dispatch to the right generator based on `spec.kind`."""
    if spec.kind == "random":
        return gen_random(spec)
    if spec.kind == "longctx":
        return gen_longctx(spec, max_total_tokens=max_total_tokens)
    if spec.kind == "sharegpt":
        if sharegpt_path is None:
            raise ValueError("sharegpt workload requires `sharegpt_path`")
        return gen_sharegpt(spec, dataset_path=sharegpt_path)
    raise AssertionError(f"unreachable kind={spec.kind!r}")  # pragma: no cover
