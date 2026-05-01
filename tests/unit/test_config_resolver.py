"""Unit tests for `lmbench.config.resolver`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lmbench.config import (
    HardwareProfile,
    RunPlan,
    WorkloadSpec,
    apply_hardware_defaults,
    expand_concurrency,
    expand_plan_concurrency,
    select_models,
)


def _plan(**overrides: object) -> RunPlan:
    base: dict[str, object] = {
        "name": "p",
        "models": [
            {"name": "m1", "hf_id": "x/y"},
            {"name": "m2", "hf_id": "a/b", "vllm": {"tensor_parallel_size": 2}},
        ],
        "workloads": [
            {
                "name": "w1",
                "kind": "random",
                "num_prompts": 100,
                "concurrency": [1, 8, 32],
                "input_len": 128,
                "output_len": 32,
                "warmup_prompts": 4,
            }
        ],
        "eval_suite": {"name": "e", "tasks": ["mmlu"]},
        "hardware": {"name": "hp", "gpu": "B300", "num_gpus": 2, "default_tp_size": 1},
    }
    base.update(overrides)
    return RunPlan.model_validate(base)


# ---- expand_concurrency ------------------------------------------------


def test_expand_concurrency_one_per_level() -> None:
    w = WorkloadSpec(
        name="w",
        kind="random",
        num_prompts=10,
        input_len=128,
        output_len=32,
        concurrency=(1, 8, 32),
        warmup_prompts=2,
    )
    out = expand_concurrency(w)
    assert len(out) == 3
    assert [x.concurrency for x in out] == [(1,), (8,), (32,)]
    for x in out:
        assert x.name == "w"
        assert x.input_len == 128
        assert x.output_len == 32


def test_expand_plan_concurrency_flattens_each_workload() -> None:
    plan = _plan()
    expanded = expand_plan_concurrency(plan)
    assert len(expanded.workloads) == 3
    assert {w.concurrency for w in expanded.workloads} == {(1,), (8,), (32,)}
    assert len(plan.workloads) == 1  # original immutable


# ---- apply_hardware_defaults ------------------------------------------


def test_apply_hardware_defaults_inherits_tp() -> None:
    plan = _plan()
    profile = HardwareProfile(
        name="hp2", gpu="B300", num_gpus=4, default_tp_size=2, blackwell=True
    )
    new = apply_hardware_defaults(plan, profile)
    m1 = next(m for m in new.models if m.name == "m1")
    assert m1.vllm.tensor_parallel_size == 2
    m2 = next(m for m in new.models if m.name == "m2")
    assert m2.vllm.tensor_parallel_size == 2
    assert new.hardware.name == "hp2"
    assert plan.hardware.name == "hp"


def test_apply_hardware_defaults_revalidates_tp_overflow() -> None:
    plan = _plan()
    too_small = HardwareProfile(name="hp3", gpu="A100", num_gpus=1, default_tp_size=1)
    with pytest.raises(ValidationError, match="tensor_parallel_size"):
        apply_hardware_defaults(plan, too_small)


def test_apply_hardware_defaults_no_change_when_profile_default_is_one() -> None:
    plan = _plan()
    profile = HardwareProfile(name="hp4", gpu="B300", num_gpus=2, default_tp_size=1)
    new = apply_hardware_defaults(plan, profile)
    m1 = next(m for m in new.models if m.name == "m1")
    assert m1.vllm.tensor_parallel_size == 1


# ---- select_models -----------------------------------------------------


def test_select_models_preserves_order() -> None:
    plan = _plan()
    sub = select_models(plan, ("m2", "m1"))
    assert [m.name for m in sub.models] == ["m2", "m1"]


def test_select_models_unknown_raises() -> None:
    plan = _plan()
    with pytest.raises(KeyError, match="missing-model"):
        select_models(plan, ("missing-model",))
