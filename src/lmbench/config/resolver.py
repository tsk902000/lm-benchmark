"""Run-plan resolution: sweep expansion and hardware-profile merging.

These are pure functions over the frozen pydantic models defined in
`schema.py`. They never mutate; each transformation returns a new instance.
"""

from __future__ import annotations

from .schema import HardwareProfile, ModelEntry, RunPlan, VLLMArgs, WorkloadSpec


def expand_concurrency(workload: WorkloadSpec) -> tuple[WorkloadSpec, ...]:
    """Expand a workload's concurrency tuple into one workload per level.

    Useful for the runner, which iterates one concurrency at a time so each
    sweep point gets its own perf-result record.
    """
    return tuple(
        workload.model_copy(update={"concurrency": (level,)})
        for level in workload.concurrency
    )


def expand_plan_concurrency(plan: RunPlan) -> RunPlan:
    """Apply `expand_concurrency` to every workload in a plan.

    Workload names are duplicated across concurrency levels by design — the
    runner pairs each `(workload, concurrency)` with its own result file —
    so we use `model_copy` rather than re-validating the plan.
    """
    expanded: list[WorkloadSpec] = []
    for w in plan.workloads:
        expanded.extend(expand_concurrency(w))
    return plan.model_copy(update={"workloads": tuple(expanded)})


def apply_hardware_defaults(plan: RunPlan, profile: HardwareProfile) -> RunPlan:
    """Replace the plan's hardware profile and inherit `default_tp_size`.

    Models that left `vllm.tensor_parallel_size` at its default (1) inherit
    the profile's `default_tp_size`. Models that explicitly set tp_size keep
    their value. The result is re-validated so any conflict (e.g. tp_size >
    num_gpus, NVFP4 on non-Blackwell) surfaces immediately.
    """
    new_models: list[ModelEntry] = []
    for m in plan.models:
        if m.vllm.tensor_parallel_size == 1 and profile.default_tp_size != 1:
            new_vllm: VLLMArgs = m.vllm.model_copy(
                update={"tensor_parallel_size": profile.default_tp_size}
            )
            new_models.append(m.model_copy(update={"vllm": new_vllm}))
        else:
            new_models.append(m)
    payload = plan.model_dump()
    payload["models"] = [m.model_dump() for m in new_models]
    payload["hardware"] = profile.model_dump()
    return RunPlan.model_validate(payload)


def select_models(plan: RunPlan, names: tuple[str, ...]) -> RunPlan:
    """Return a new plan whose `models` contains only the named entries.

    Order is preserved from `names`. Unknown names raise.
    """
    by_name = {m.name: m for m in plan.models}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"models not found in plan: {missing}")
    selected = tuple(by_name[n] for n in names)
    return plan.model_copy(update={"models": selected})
