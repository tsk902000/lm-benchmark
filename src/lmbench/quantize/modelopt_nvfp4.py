"""NVFP4 post-training quantization via `nvidia-modelopt`.

This module wraps the modelopt `mtq.quantize` flow:

1. Load the HF baseline checkpoint with `transformers`.
2. Load calibration data (default: cnn_dailymail x 512 samples).
3. Run `mtq.quantize(model, NVFP4_DEFAULT_CFG, forward_loop=...)`.
4. Export the quantized model in vLLM-consumable HF format via
   `model.save_pretrained` plus a sidecar `lmbench_quant_meta.json`.

The third-party imports (`torch`, `transformers`, `modelopt`) are lazy
so the module is importable on dev boxes without the [gpu]/[quant]
extras. Tests cover the pure-Python helpers and mock the heavy calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lmbench.config import ModelEntry, QuantRecipe

from .calibration import sample_calibration_text, tokenize_for_calibration


@dataclass(frozen=True)
class QuantizedCheckpoint:
    """Handle to a quantized checkpoint on disk."""

    output_dir: Path
    method: str
    source_hf_id: str

    @property
    def vllm_id(self) -> str:
        """Path string for `vllm serve <vllm_id>` to load this checkpoint."""
        return str(self.output_dir)


def safe_dirname(name: str) -> str:
    """Sanitize a model name for use as a filesystem directory."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def quantized_output_dir(model: ModelEntry, recipe: QuantRecipe) -> Path:
    """Resolve the on-disk location for a model+recipe quantization."""
    return recipe.output_dir / safe_dirname(model.name) / safe_dirname(recipe.name)


def build_export_metadata(model: ModelEntry, recipe: QuantRecipe) -> dict[str, Any]:
    """Sidecar `lmbench_quant_meta.json` describing the export."""
    return {
        "method": recipe.method,
        "source_hf_id": model.hf_id,
        "source_revision": model.revision,
        "model_name": model.name,
        "recipe_name": recipe.name,
        "calibration": {
            "dataset": recipe.calibration.dataset,
            "dataset_config": recipe.calibration.dataset_config,
            "split": recipe.calibration.split,
            "num_samples": recipe.calibration.num_samples,
            "max_seq_len": recipe.calibration.max_seq_len,
            "seed": recipe.calibration.seed,
        },
    }


def _import_torch_and_transformers() -> tuple[Any, Any, Any]:
    """Lazy imports — heavy and Linux-only via the [gpu] extra."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "modelopt_nvfp4 needs the [gpu] extra. "
            "Install with `uv sync --extra all` on a CUDA host."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _import_modelopt() -> tuple[Any, Any]:
    """Lazy modelopt import — Linux/CUDA only via the [quant] extra."""
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization import config as mtq_config
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "NVFP4 quantization needs `nvidia-modelopt[all]`. "
            "Install with `uv sync --extra all` on a CUDA host."
        ) from exc
    return mtq, mtq_config


def _select_modelopt_config(method: str, mtq_config: Any) -> Any:
    """Pick the modelopt config for a given recipe method."""
    if method == "nvfp4":
        cfg = getattr(mtq_config, "NVFP4_DEFAULT_CFG", None)
        if cfg is None:
            raise RuntimeError(
                "installed modelopt does not expose NVFP4_DEFAULT_CFG; "
                "upgrade `nvidia-modelopt`"
            )
        return cfg
    if method == "nvfp4_llmcompressor":
        raise NotImplementedError(
            "nvfp4_llmcompressor recipe is not yet wired; use 'nvfp4' for now"
        )
    raise ValueError(f"unknown quantization method: {method!r}")


def quantize_to_nvfp4(
    *,
    model_entry: ModelEntry,
    recipe: QuantRecipe,
    output_dir: Path | None = None,
) -> QuantizedCheckpoint:
    """Run NVFP4 PTQ end-to-end. Returns a handle to the saved checkpoint."""
    torch, AutoModelForCausalLM, AutoTokenizer = _import_torch_and_transformers()
    mtq, mtq_config = _import_modelopt()

    out_dir = output_dir or quantized_output_dir(model_entry, recipe)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_entry.hf_id,
        revision=model_entry.revision,
        trust_remote_code=model_entry.vllm.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_entry.hf_id,
        revision=model_entry.revision,
        trust_remote_code=model_entry.vllm.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    texts = sample_calibration_text(recipe.calibration)
    encoded = tokenize_for_calibration(
        texts, tokenizer, max_seq_len=recipe.calibration.max_seq_len
    )

    def forward_loop(m: Any) -> None:
        with torch.no_grad():
            for batch in encoded:
                inputs = {k: v.to(m.device) for k, v in batch.items()}
                m(**inputs)

    cfg = _select_modelopt_config(recipe.method, mtq_config)
    mtq.quantize(model, cfg, forward_loop=forward_loop)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta_path = out_dir / "lmbench_quant_meta.json"
    meta_path.write_text(
        json.dumps(build_export_metadata(model_entry, recipe), indent=2),
        encoding="utf-8",
    )
    return QuantizedCheckpoint(
        output_dir=out_dir, method=recipe.method, source_hf_id=model_entry.hf_id
    )
