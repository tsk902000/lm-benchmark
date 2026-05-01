"""Pydantic models for the lmbench configuration surface.

Schema is split into reusable components — `VLLMArgs`, `ModelEntry`,
`WorkloadSpec`, `EvalSuite`, `QuantRecipe`, `HardwareProfile` — composed by
`RunPlan`. All models are frozen and reject unknown fields so config typos
fail fast at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VLLMArgs(BaseModel):
    """vLLM serve flags. Maps onto `vllm serve` CLI options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    enforce_eager: bool = False
    trust_remote_code: bool = False
    quantization: str | None = None
    enable_prefix_caching: bool = True
    extra_args: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_args")
    @classmethod
    def _no_reserved(cls, v: dict[str, str]) -> dict[str, str]:
        reserved = {
            "model",
            "tensor-parallel-size",
            "tensor_parallel_size",
            "max-model-len",
            "max_model_len",
            "dtype",
        }
        clash = reserved & v.keys()
        if clash:
            raise ValueError(
                f"Use the dedicated VLLMArgs field instead of extra_args for: {sorted(clash)}"
            )
        return v


class ModelEntry(BaseModel):
    """A single entry in the model registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    hf_id: str = Field(min_length=1)
    revision: str | None = None
    served_model_name: str | None = None
    vllm: VLLMArgs = Field(default_factory=VLLMArgs)
    notes: str | None = None
    gated: bool = False
    expected_max_model_len: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> ModelEntry:
        if (
            self.expected_max_model_len is not None
            and self.vllm.max_model_len is not None
            and self.vllm.max_model_len > self.expected_max_model_len
        ):
            raise ValueError(
                f"vllm.max_model_len ({self.vllm.max_model_len}) exceeds "
                f"expected_max_model_len ({self.expected_max_model_len}) "
                f"for model {self.name!r}"
            )
        return self


class WorkloadSpec(BaseModel):
    """A performance benchmark workload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["sharegpt", "random", "longctx"]
    num_prompts: int = Field(ge=1)
    concurrency: tuple[int, ...] = Field(default=(1, 8, 32, 128))
    input_len: int | None = Field(default=None, ge=1)
    output_len: int | None = Field(default=None, ge=1)
    seed: int = 0
    warmup_prompts: int = Field(default=8, ge=0)

    @field_validator("concurrency")
    @classmethod
    def _normalize_concurrency(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("concurrency must contain at least one level")
        if any(c < 1 for c in v):
            raise ValueError("concurrency levels must be >= 1")
        if len(set(v)) != len(v):
            raise ValueError("concurrency levels must be unique")
        return tuple(sorted(v))

    @model_validator(mode="after")
    def _validate(self) -> WorkloadSpec:
        if self.kind in ("random", "longctx") and (
            self.input_len is None or self.output_len is None
        ):
            raise ValueError(
                f"workload kind={self.kind!r} requires input_len and output_len"
            )
        if self.warmup_prompts > self.num_prompts:
            raise ValueError(
                f"warmup_prompts ({self.warmup_prompts}) > num_prompts ({self.num_prompts})"
            )
        return self


class EvalSuite(BaseModel):
    """A quality benchmark suite (driven by `lm-eval`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    tasks: tuple[str, ...] = Field(min_length=1)
    num_fewshot: dict[str, int] = Field(default_factory=dict)
    limit: int | None = Field(default=None, ge=1)
    include_humaneval: bool = False
    long_context: tuple[str, ...] = Field(default=())

    @field_validator("tasks")
    @classmethod
    def _unique_tasks(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("tasks must be unique")
        return v

    @model_validator(mode="after")
    def _validate(self) -> EvalSuite:
        if "humaneval" in self.tasks and not self.include_humaneval:
            raise ValueError(
                "humaneval requires include_humaneval=True (executes generated code)"
            )
        unknown = set(self.num_fewshot) - set(self.tasks)
        if unknown:
            raise ValueError(
                f"num_fewshot keys must be a subset of tasks; unknown: {sorted(unknown)}"
            )
        return self


class CalibrationSpec(BaseModel):
    """Calibration data spec for post-training quantization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = "cnn_dailymail"
    dataset_config: str | None = "3.0.0"
    split: str = "train"
    num_samples: int = Field(default=512, ge=1)
    max_seq_len: int = Field(default=2048, ge=1)
    seed: int = 0


class QuantRecipe(BaseModel):
    """A quantization recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    method: Literal["nvfp4", "nvfp4_llmcompressor"] = "nvfp4"
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)
    output_dir: Path = Path("results/quantized")


class HardwareProfile(BaseModel):
    """Target hardware profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    blackwell: bool = False
    num_gpus: int = Field(default=1, ge=1)
    default_tp_size: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def _validate(self) -> HardwareProfile:
        if self.default_tp_size > self.num_gpus:
            raise ValueError(
                f"default_tp_size ({self.default_tp_size}) > num_gpus ({self.num_gpus})"
            )
        return self


class RunPlan(BaseModel):
    """Top-level configuration tying models, workloads, eval, quant, and hardware."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    models: tuple[ModelEntry, ...] = Field(min_length=1)
    workloads: tuple[WorkloadSpec, ...] = Field(min_length=1)
    eval_suite: EvalSuite
    quant_recipe: QuantRecipe | None = None
    hardware: HardwareProfile
    output_dir: Path = Path("results")

    @model_validator(mode="after")
    def _validate(self) -> RunPlan:
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        wl_names = [w.name for w in self.workloads]
        if len(set(wl_names)) != len(wl_names):
            raise ValueError("workload names must be unique")
        for m in self.models:
            if m.vllm.tensor_parallel_size > self.hardware.num_gpus:
                raise ValueError(
                    f"model {m.name!r} tensor_parallel_size "
                    f"({m.vllm.tensor_parallel_size}) > hardware.num_gpus "
                    f"({self.hardware.num_gpus})"
                )
        if (
            self.quant_recipe is not None
            and self.quant_recipe.method == "nvfp4"
            and not self.hardware.blackwell
        ):
            raise ValueError(
                "NVFP4 quantization only accelerates on Blackwell GPUs; "
                "set hardware.blackwell=True to acknowledge or use a different recipe"
            )
        return self
