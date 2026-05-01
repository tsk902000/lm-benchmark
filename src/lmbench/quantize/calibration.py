"""Calibration data loaders for post-training quantization.

Default: 512 samples from `cnn_dailymail` train, truncated to 2048 tokens
per HANDOFF.md. The `datasets` import is deferred so the harness module
imports cleanly on hosts without the [gpu] extra.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any, Protocol

from lmbench.config import CalibrationSpec


class _Tokenizer(Protocol):
    """Subset of `transformers.PreTrainedTokenizer` we rely on."""

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = ...,
        max_length: int | None = ...,
        return_tensors: str | None = ...,
    ) -> Any: ...


def _load_dataset_text(spec: CalibrationSpec) -> list[str]:
    """Load raw text rows for a calibration spec; lazy `datasets` import."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "calibration loaders need the `datasets` package. "
            "Install the [gpu] extra: `uv sync --extra all`."
        ) from exc
    ds = load_dataset(spec.dataset, spec.dataset_config, split=spec.split)
    field: str
    if spec.dataset == "cnn_dailymail":
        field = "article"
    elif spec.dataset == "wikitext":
        field = "text"
    else:
        sample = ds[0]
        candidate = next(
            (k for k, v in sample.items() if isinstance(v, str) and v.strip()),
            None,
        )
        if candidate is None:
            raise ValueError(
                f"could not infer text column for dataset {spec.dataset!r}"
            )
        field = candidate
    return [row for row in ds[field] if isinstance(row, str) and row.strip()]


def sample_calibration_text(spec: CalibrationSpec) -> tuple[str, ...]:
    """Return `spec.num_samples` deterministic-seeded text samples."""
    rows = _load_dataset_text(spec)
    if not rows:
        raise ValueError(f"calibration dataset {spec.dataset!r} produced no rows")
    rng = random.Random(spec.seed)
    if len(rows) <= spec.num_samples:
        return tuple(rows)
    return tuple(rng.sample(rows, spec.num_samples))


def tokenize_for_calibration(
    texts: Iterable[str], tokenizer: _Tokenizer, *, max_seq_len: int = 2048
) -> list[Any]:
    """Tokenize calibration texts into batched-by-1 tensors.

    Returns a list of tokenized inputs ready to feed into modelopt's
    `quantize(model, forward_loop=...)` callback.
    """
    encoded: list[Any] = []
    for text in texts:
        out = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        encoded.append(out)
    return encoded
