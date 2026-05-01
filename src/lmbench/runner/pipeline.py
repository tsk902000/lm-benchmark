"""End-to-end pipeline.

For each model in the plan: capture env -> serve baseline -> bench perf -> bench quality
-> teardown -> quantize -> serve nvfp4 -> bench perf -> bench quality -> teardown ->
compare -> report. Heavy operations are imported but only executed when the
runner is actually invoked, so this module imports cleanly on dev hosts
without GPU extras.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lmbench.bench import (
    PerfResult,
    QualityResult,
    generate,
    run_quality,
    run_workload,
)
from lmbench.compare import (
    ComparisonReport,
    PerfComparison,
    QualityComparison,
    diff_perf,
    diff_quality,
)
from lmbench.config import (
    ModelEntry,
    RunPlan,
    expand_concurrency,
    load_run_plan,
)
from lmbench.quantize import (
    QuantizedCheckpoint,
    quantize_to_nvfp4,
    verify_checkpoint,
)
from lmbench.report import write_html, write_markdown
from lmbench.serve import ServerHandle, serve_model

from .env import EnvCapture, capture_to_path


@dataclass(frozen=True)
class ModelRunResult:
    """Per-model outcome bundle."""

    model_name: str
    baseline_perf: tuple[PerfResult, ...]
    baseline_quality: QualityResult | None
    quantized_perf: tuple[PerfResult, ...]
    quantized_quality: QualityResult | None
    comparison: ComparisonReport
    quantized_checkpoint: QuantizedCheckpoint | None
    report_md: Path
    report_html: Path


@dataclass(frozen=True)
class PipelineResult:
    """Full run outcome."""

    plan_name: str
    output_dir: Path
    env: EnvCapture
    models: tuple[ModelRunResult, ...]


def _perf_summary_to_dict(result: PerfResult) -> dict[str, object]:
    s = result.summary
    return {
        "n_requests": s.n_requests,
        "n_success": s.n_success,
        "duration_s": s.duration_s,
        "ttft": asdict(s.ttft),
        "itl": asdict(s.itl),
        "tpot": asdict(s.tpot),
        "e2e": asdict(s.e2e),
        "output_tokens_total": s.output_tokens_total,
        "output_tokens_per_s": s.output_tokens_per_s,
        "request_rate_per_s": s.request_rate_per_s,
    }


def _save_perf_result(result: PerfResult, target_dir: Path) -> Path:
    """Persist a `PerfResult` summary as JSON next to its raw samples."""
    target_dir.mkdir(parents=True, exist_ok=True)
    name = f"{result.workload_name}_c{result.concurrency}.json"
    path = target_dir / name
    payload = {
        "workload_name": result.workload_name,
        "concurrency": result.concurrency,
        "summary": _perf_summary_to_dict(result),
        "n_samples": len(result.samples),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _bench_perf_for_model(
    *,
    plan: RunPlan,
    model: ModelEntry,
    handle: ServerHandle,
    output_dir: Path,
) -> tuple[PerfResult, ...]:
    """Run every (workload, concurrency) cell against a live server."""
    perf_dir = output_dir / "perf"
    results: list[PerfResult] = []
    for workload in plan.workloads:
        for expanded in expand_concurrency(workload):
            prompts = generate(expanded)
            concurrency = expanded.concurrency[0]
            result = run_workload(
                base_url=handle.base_url,
                served_model_name=handle.served_model_name,
                workload=expanded,
                concurrency=concurrency,
                prompts=prompts,
            )
            _save_perf_result(result, perf_dir)
            results.append(result)
    del model
    return tuple(results)


def _bench_quality_for_model(
    *,
    plan: RunPlan,
    model: ModelEntry,
    handle: ServerHandle,
    output_dir: Path,
) -> QualityResult:
    """Run the eval suite against a live server."""
    return run_quality(
        suite=plan.eval_suite,
        model=model,
        base_url=handle.base_url,
        served_model_name=handle.served_model_name,
        output_dir=output_dir / "quality",
    )


def _build_comparison(
    *,
    baseline_perf: tuple[PerfResult, ...],
    candidate_perf: tuple[PerfResult, ...],
    baseline_quality: QualityResult | None,
    candidate_quality: QualityResult | None,
) -> ComparisonReport:
    """Pair perf results by (workload, concurrency); pair quality by suite."""
    cand_index = {(r.workload_name, r.concurrency): r for r in candidate_perf}
    perf_cmps: list[PerfComparison] = []
    for base in baseline_perf:
        cand = cand_index.get((base.workload_name, base.concurrency))
        if cand is None:
            continue
        perf_cmps.append(
            diff_perf(
                workload_name=base.workload_name,
                concurrency=base.concurrency,
                baseline=base.summary,
                candidate=cand.summary,
            )
        )
    quality_cmps: list[QualityComparison] = []
    if baseline_quality is not None and candidate_quality is not None:
        quality_cmps.append(
            diff_quality(baseline=baseline_quality, candidate=candidate_quality)
        )
    return ComparisonReport(perf=tuple(perf_cmps), quality=tuple(quality_cmps))


def _run_one_model(
    *,
    plan: RunPlan,
    model: ModelEntry,
    output_dir: Path,
    skip_quality: bool = False,
    skip_quantize: bool = False,
    skip_baseline: bool = False,
) -> ModelRunResult:
    """Execute the full per-model pipeline.

    `skip_baseline=True` is for hosts that can't fit the bf16 baseline
    (e.g. 2x B300 vs a 310B-param model) — the run quantizes and measures
    only the candidate, and the comparison report is empty. Raises if
    `skip_baseline` and `skip_quantize` are both True (nothing would run).
    """
    if skip_baseline and skip_quantize:
        raise ValueError(
            "skip_baseline and skip_quantize cannot both be True; "
            "nothing would run for this model"
        )
    model_dir = output_dir / model.name
    baseline_dir = model_dir / "baseline"
    quant_dir = model_dir / "quantized"
    cmp_dir = model_dir / "comparison"

    baseline_perf: tuple[PerfResult, ...] = ()
    baseline_quality: QualityResult | None = None
    if not skip_baseline:
        with serve_model(model) as handle:
            baseline_perf = _bench_perf_for_model(
                plan=plan, model=model, handle=handle, output_dir=baseline_dir
            )
            baseline_quality = (
                None
                if skip_quality
                else _bench_quality_for_model(
                    plan=plan, model=model, handle=handle, output_dir=baseline_dir
                )
            )

    quantized_perf: tuple[PerfResult, ...] = ()
    quantized_quality: QualityResult | None = None
    quant_ckpt: QuantizedCheckpoint | None = None
    if not skip_quantize and plan.quant_recipe is not None:
        quant_ckpt = quantize_to_nvfp4(
            model_entry=model, recipe=plan.quant_recipe
        )
        verify = verify_checkpoint(checkpoint=quant_ckpt, base_entry=model)
        if not verify.ok:
            raise RuntimeError(
                f"NVFP4 verification failed for {model.name}: {verify.reason}; "
                f"completion={verify.completion!r}"
            )
        quant_entry = ModelEntry(
            name=f"{model.name}-quant",
            hf_id=quant_ckpt.vllm_id,
            served_model_name=f"{model.name}-quant",
            vllm=model.vllm.model_copy(update={"quantization": "modelopt_fp4"}),
        )
        with serve_model(quant_entry) as handle:
            quantized_perf = _bench_perf_for_model(
                plan=plan, model=quant_entry, handle=handle, output_dir=quant_dir
            )
            if not skip_quality:
                quantized_quality = _bench_quality_for_model(
                    plan=plan,
                    model=quant_entry,
                    handle=handle,
                    output_dir=quant_dir,
                )

    comparison = _build_comparison(
        baseline_perf=baseline_perf,
        candidate_perf=quantized_perf,
        baseline_quality=baseline_quality,
        candidate_quality=quantized_quality,
    )
    md_path = write_markdown(comparison, cmp_dir / "report.md", title=model.name)
    html_path = write_html(comparison, cmp_dir / "report.html", title=model.name)
    return ModelRunResult(
        model_name=model.name,
        baseline_perf=baseline_perf,
        baseline_quality=baseline_quality,
        quantized_perf=quantized_perf,
        quantized_quality=quantized_quality,
        comparison=comparison,
        quantized_checkpoint=quant_ckpt,
        report_md=md_path,
        report_html=html_path,
    )


def run_plan(
    plan: RunPlan,
    *,
    output_dir: Path | None = None,
    skip_quality: bool = False,
    skip_quantize: bool = False,
    skip_baseline: bool = False,
) -> PipelineResult:
    """Run a full `RunPlan` — every model gets its own subdir under output_dir."""
    out = output_dir or plan.output_dir
    out.mkdir(parents=True, exist_ok=True)
    env = capture_to_path(out / "env.json")
    model_results: list[ModelRunResult] = []
    for model in plan.models:
        result = _run_one_model(
            plan=plan,
            model=model,
            output_dir=out,
            skip_quality=skip_quality,
            skip_quantize=skip_quantize,
            skip_baseline=skip_baseline,
        )
        model_results.append(result)
    return PipelineResult(
        plan_name=plan.name,
        output_dir=out,
        env=env,
        models=tuple(model_results),
    )


def run_plan_from_file(
    path: Path,
    *,
    output_dir: Path | None = None,
    skip_quality: bool = False,
    skip_quantize: bool = False,
    skip_baseline: bool = False,
) -> PipelineResult:
    """Convenience: load a plan YAML and run it."""
    plan = load_run_plan(path)
    return run_plan(
        plan,
        output_dir=output_dir,
        skip_quality=skip_quality,
        skip_quantize=skip_quantize,
        skip_baseline=skip_baseline,
    )
