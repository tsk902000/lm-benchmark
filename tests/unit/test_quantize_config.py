"""Unit tests for the pure helpers in `lmbench.quantize.modelopt_nvfp4`."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmbench.config import CalibrationSpec, ModelEntry, QuantRecipe
from lmbench.quantize import (
    QuantizedCheckpoint,
    build_export_metadata,
    quantized_output_dir,
    safe_dirname,
)
from lmbench.quantize.modelopt_nvfp4 import _select_modelopt_config


def test_safe_dirname_replaces_special_chars() -> None:
    assert safe_dirname("Llama-3.1/8B Instruct") == "Llama-3.1_8B_Instruct"
    assert safe_dirname("a:b?c") == "a_b_c"


def test_safe_dirname_falls_back_for_blank() -> None:
    assert safe_dirname("") == "model"
    assert safe_dirname("///") == "model"


def test_safe_dirname_preserves_dashes_and_dots() -> None:
    assert safe_dirname("opt-125m.v2") == "opt-125m.v2"


def test_quantized_output_dir_layout() -> None:
    model = ModelEntry(name="opt-125m", hf_id="facebook/opt-125m")
    recipe = QuantRecipe(name="nvfp4-default", output_dir=Path("results/quantized"))
    out = quantized_output_dir(model, recipe)
    assert out == Path("results/quantized/opt-125m/nvfp4-default")


def test_build_export_metadata_shape() -> None:
    model = ModelEntry(name="opt-125m", hf_id="facebook/opt-125m", revision="main")
    recipe = QuantRecipe(
        name="r",
        method="nvfp4",
        calibration=CalibrationSpec(num_samples=8, max_seq_len=128, seed=7),
    )
    meta = build_export_metadata(model, recipe)
    assert meta["method"] == "nvfp4"
    assert meta["source_hf_id"] == "facebook/opt-125m"
    assert meta["source_revision"] == "main"
    assert meta["calibration"]["num_samples"] == 8
    assert meta["calibration"]["seed"] == 7


def test_select_modelopt_config_nvfp4() -> None:
    class _FakeMtqConfig:
        NVFP4_DEFAULT_CFG: dict[str, str] = {"some": "config"}  # noqa: RUF012

    assert _select_modelopt_config("nvfp4", _FakeMtqConfig) == {"some": "config"}


def test_select_modelopt_config_missing_attr_raises() -> None:
    class _BareConfig:
        pass

    with pytest.raises(RuntimeError, match="NVFP4_DEFAULT_CFG"):
        _select_modelopt_config("nvfp4", _BareConfig)


def test_select_modelopt_config_llmcompressor_not_implemented() -> None:
    class _FakeMtqConfig:
        NVFP4_DEFAULT_CFG = object()

    with pytest.raises(NotImplementedError, match="llmcompressor"):
        _select_modelopt_config("nvfp4_llmcompressor", _FakeMtqConfig)


def test_select_modelopt_config_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown quantization method"):
        _select_modelopt_config("int4", object())


def test_quantized_checkpoint_vllm_id() -> None:
    ckpt = QuantizedCheckpoint(
        output_dir=Path("/data/q/opt"),
        method="nvfp4",
        source_hf_id="facebook/opt-125m",
    )
    assert ckpt.vllm_id == "/data/q/opt"
