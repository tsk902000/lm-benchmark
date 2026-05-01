"""Unit tests for `lmbench.config.schema`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lmbench.config import (
    CalibrationSpec,
    EvalSuite,
    HardwareProfile,
    ModelEntry,
    QuantRecipe,
    RunPlan,
    VLLMArgs,
    WorkloadSpec,
)

# ---- VLLMArgs ----------------------------------------------------------


def test_vllm_args_defaults() -> None:
    a = VLLMArgs()
    assert a.tensor_parallel_size == 1
    assert a.gpu_memory_utilization == 0.9
    assert a.dtype == "auto"
    assert a.enable_prefix_caching is True


def test_vllm_args_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        VLLMArgs.model_validate({"unknown": True})


def test_vllm_args_extra_args_rejects_reserved_keys() -> None:
    with pytest.raises(ValidationError, match="dedicated VLLMArgs field"):
        VLLMArgs.model_validate({"extra_args": {"dtype": "bfloat16"}})


def test_vllm_args_gpu_util_bounds() -> None:
    with pytest.raises(ValidationError):
        VLLMArgs(gpu_memory_utilization=0.0)
    with pytest.raises(ValidationError):
        VLLMArgs(gpu_memory_utilization=1.1)


def test_vllm_args_is_frozen() -> None:
    a = VLLMArgs()
    with pytest.raises(ValidationError):
        a.tensor_parallel_size = 4  # type: ignore[misc]


# ---- ModelEntry --------------------------------------------------------


def test_model_entry_minimal() -> None:
    m = ModelEntry(name="opt", hf_id="facebook/opt-125m")
    assert m.gated is False
    assert m.vllm.tensor_parallel_size == 1


def test_model_entry_max_len_exceeds_expected() -> None:
    with pytest.raises(ValidationError, match="exceeds expected_max_model_len"):
        ModelEntry(
            name="m",
            hf_id="x/y",
            expected_max_model_len=1024,
            vllm=VLLMArgs(max_model_len=2048),
        )


def test_model_entry_max_len_within_expected() -> None:
    m = ModelEntry(
        name="m",
        hf_id="x/y",
        expected_max_model_len=4096,
        vllm=VLLMArgs(max_model_len=2048),
    )
    assert m.vllm.max_model_len == 2048


# ---- WorkloadSpec ------------------------------------------------------


def test_workload_random_requires_lens() -> None:
    with pytest.raises(ValidationError, match="requires input_len"):
        WorkloadSpec(name="w", kind="random", num_prompts=10)


def test_workload_sharegpt_no_lens_required() -> None:
    w = WorkloadSpec(name="w", kind="sharegpt", num_prompts=10, warmup_prompts=2)
    assert w.input_len is None


def test_workload_sorts_dedupes_concurrency() -> None:
    w = WorkloadSpec(
        name="w",
        kind="random",
        num_prompts=10,
        input_len=128,
        output_len=32,
        concurrency=(8, 1, 32),
    )
    assert w.concurrency == (1, 8, 32)


def test_workload_rejects_duplicate_concurrency() -> None:
    with pytest.raises(ValidationError, match="unique"):
        WorkloadSpec(
            name="w",
            kind="random",
            num_prompts=10,
            input_len=128,
            output_len=32,
            concurrency=(1, 1, 8),
        )


def test_workload_rejects_zero_concurrency() -> None:
    with pytest.raises(ValidationError):
        WorkloadSpec(
            name="w",
            kind="random",
            num_prompts=10,
            input_len=128,
            output_len=32,
            concurrency=(0, 8),
        )


def test_workload_rejects_empty_concurrency() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        WorkloadSpec(
            name="w",
            kind="random",
            num_prompts=10,
            input_len=128,
            output_len=32,
            concurrency=(),
        )


def test_workload_warmup_capped_by_num_prompts() -> None:
    with pytest.raises(ValidationError, match="warmup_prompts"):
        WorkloadSpec(
            name="w",
            kind="random",
            num_prompts=10,
            input_len=128,
            output_len=32,
            warmup_prompts=20,
        )


# ---- EvalSuite ---------------------------------------------------------


def test_eval_suite_humaneval_requires_opt_in() -> None:
    with pytest.raises(ValidationError, match="include_humaneval"):
        EvalSuite(name="e", tasks=("mmlu", "humaneval"))


def test_eval_suite_humaneval_with_opt_in() -> None:
    e = EvalSuite(name="e", tasks=("humaneval",), include_humaneval=True)
    assert e.tasks == ("humaneval",)


def test_eval_suite_unknown_fewshot_key() -> None:
    with pytest.raises(ValidationError, match="num_fewshot keys"):
        EvalSuite(name="e", tasks=("mmlu",), num_fewshot={"gsm8k": 5})


def test_eval_suite_duplicate_tasks() -> None:
    with pytest.raises(ValidationError, match="unique"):
        EvalSuite(name="e", tasks=("mmlu", "mmlu"))


# ---- HardwareProfile ---------------------------------------------------


def test_hardware_tp_within_gpus() -> None:
    p = HardwareProfile(name="hp", gpu="B300", num_gpus=2, default_tp_size=2)
    assert p.default_tp_size == 2


def test_hardware_tp_exceeds_gpus() -> None:
    with pytest.raises(ValidationError, match="default_tp_size"):
        HardwareProfile(name="hp", gpu="B300", num_gpus=1, default_tp_size=2)


# ---- QuantRecipe / CalibrationSpec -------------------------------------


def test_calibration_defaults() -> None:
    c = CalibrationSpec()
    assert c.dataset == "cnn_dailymail"
    assert c.num_samples == 512


def test_quant_recipe_default_method() -> None:
    r = QuantRecipe(name="r")
    assert r.method == "nvfp4"
    assert r.calibration.dataset == "cnn_dailymail"


def test_quant_recipe_invalid_method() -> None:
    with pytest.raises(ValidationError):
        QuantRecipe.model_validate({"name": "r", "method": "int8"})


# ---- RunPlan -----------------------------------------------------------


def _basic_plan(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "p",
        "models": [
            {
                "name": "m1",
                "hf_id": "x/y",
                "vllm": {"tensor_parallel_size": 1},
            }
        ],
        "workloads": [
            {
                "name": "w1",
                "kind": "random",
                "num_prompts": 10,
                "concurrency": [1],
                "input_len": 128,
                "output_len": 32,
            }
        ],
        "eval_suite": {"name": "e", "tasks": ["mmlu"]},
        "hardware": {"name": "hp", "gpu": "B300", "num_gpus": 2, "default_tp_size": 2},
    }
    base.update(overrides)
    return base


def test_run_plan_minimal() -> None:
    plan = RunPlan.model_validate(_basic_plan())
    assert plan.name == "p"
    assert plan.models[0].name == "m1"


def test_run_plan_duplicate_model_names() -> None:
    bad = _basic_plan(
        models=[
            {"name": "m1", "hf_id": "x/y"},
            {"name": "m1", "hf_id": "a/b"},
        ]
    )
    with pytest.raises(ValidationError, match="model names must be unique"):
        RunPlan.model_validate(bad)


def test_run_plan_duplicate_workload_names() -> None:
    bad = _basic_plan(
        workloads=[
            {
                "name": "w1",
                "kind": "random",
                "num_prompts": 10,
                "concurrency": [1],
                "input_len": 128,
                "output_len": 32,
            },
            {
                "name": "w1",
                "kind": "random",
                "num_prompts": 5,
                "concurrency": [1],
                "input_len": 128,
                "output_len": 32,
                "warmup_prompts": 0,
            },
        ]
    )
    with pytest.raises(ValidationError, match="workload names must be unique"):
        RunPlan.model_validate(bad)


def test_run_plan_tp_exceeds_hardware() -> None:
    bad = _basic_plan(
        models=[{"name": "m1", "hf_id": "x/y", "vllm": {"tensor_parallel_size": 4}}]
    )
    with pytest.raises(ValidationError, match="tensor_parallel_size"):
        RunPlan.model_validate(bad)


def test_run_plan_nvfp4_requires_blackwell() -> None:
    bad = _basic_plan(
        quant_recipe={"name": "r", "method": "nvfp4"},
        hardware={
            "name": "hp",
            "gpu": "H100",
            "num_gpus": 2,
            "default_tp_size": 2,
            "blackwell": False,
        },
    )
    with pytest.raises(ValidationError, match="Blackwell"):
        RunPlan.model_validate(bad)


def test_run_plan_nvfp4_blackwell_ok() -> None:
    good = _basic_plan(
        quant_recipe={"name": "r", "method": "nvfp4"},
        hardware={
            "name": "hp",
            "gpu": "B300",
            "num_gpus": 2,
            "default_tp_size": 2,
            "blackwell": True,
        },
    )
    plan = RunPlan.model_validate(good)
    assert plan.quant_recipe is not None
    assert plan.hardware.blackwell is True
