"""Unit tests for `lmbench.quantize.calibration`."""

from __future__ import annotations

from typing import Any

import pytest

from lmbench.config import CalibrationSpec
from lmbench.quantize import calibration as cal_mod
from lmbench.quantize import sample_calibration_text, tokenize_for_calibration


def test_sample_calibration_text_subsamples(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [f"row-{i}" for i in range(20)]
    monkeypatch.setattr(cal_mod, "_load_dataset_text", lambda _spec: rows)
    spec = CalibrationSpec(num_samples=5, seed=0)
    out = sample_calibration_text(spec)
    assert len(out) == 5
    assert all(s in rows for s in out)


def test_sample_calibration_text_returns_all_when_smaller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = ["a", "b", "c"]
    monkeypatch.setattr(cal_mod, "_load_dataset_text", lambda _spec: rows)
    spec = CalibrationSpec(num_samples=10)
    out = sample_calibration_text(spec)
    assert set(out) == {"a", "b", "c"}


def test_sample_calibration_text_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [f"row-{i}" for i in range(50)]
    monkeypatch.setattr(cal_mod, "_load_dataset_text", lambda _spec: rows)
    spec = CalibrationSpec(num_samples=10, seed=42)
    a = sample_calibration_text(spec)
    b = sample_calibration_text(spec)
    assert a == b


def test_sample_calibration_text_rejects_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cal_mod, "_load_dataset_text", lambda _spec: [])
    spec = CalibrationSpec(num_samples=5)
    with pytest.raises(ValueError, match="produced no rows"):
        sample_calibration_text(spec)


def test_tokenize_for_calibration_invokes_tokenizer() -> None:
    captured: list[dict[str, Any]] = []

    def fake_tokenizer(
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        captured.append(
            {
                "text": text,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        return {"input_ids": [[1, 2, 3]]}

    out = tokenize_for_calibration(["hello", "world"], fake_tokenizer, max_seq_len=128)
    assert len(out) == 2
    assert all(e == {"input_ids": [[1, 2, 3]]} for e in out)
    assert captured[0]["truncation"] is True
    assert captured[0]["max_length"] == 128
    assert captured[0]["return_tensors"] == "pt"
