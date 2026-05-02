"""lmbench CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="lmbench",
    help="vLLM benchmark harness with NVFP4 quantization comparison.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to model registry YAML.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model entry name from the registry.")
    ],
    mode: Annotated[
        str, typer.Option("--mode", help="online (HTTP server) or offline (in-process).")
    ] = "online",
) -> None:
    """Start a vLLM server (or offline engine) for the named model."""
    console.print(f"[yellow]serve[/] (stub) config={config} model={model} mode={mode}")


@app.command()
def bench(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to benchmark config YAML.")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model entry name.")],
    suite: Annotated[str, typer.Option("--suite", help="perf | quality | both.")] = "perf",
) -> None:
    """Run benchmarks against a running vLLM endpoint."""
    console.print(f"[yellow]bench[/] (stub) config={config} model={model} suite={suite}")


@app.command()
def quantize(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to quantization config YAML.")
    ],
    model: Annotated[str, typer.Option("--model", "-m", help="Model entry name.")],
    recipe: Annotated[
        str, typer.Option("--recipe", help="nvfp4 | nvfp4-llmcompressor.")
    ] = "nvfp4",
) -> None:
    """Produce a quantized checkpoint (default: NVFP4 via nvidia-modelopt)."""
    console.print(f"[yellow]quantize[/] (stub) config={config} model={model} recipe={recipe}")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline run-results directory.")],
    candidate: Annotated[
        Path, typer.Option("--candidate", help="Candidate run-results directory.")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the report.")
    ] = Path("reports"),
) -> None:
    """Diff two run-results directories and emit a side-by-side report."""
    from lmbench.compare import compare_result_dirs
    from lmbench.report import write_html, write_markdown

    report = compare_result_dirs(baseline, candidate)
    md = write_markdown(report, output / "report.md")
    html = write_html(report, output / "report.html")
    flag = "[red]REGRESSION[/]" if report.any_regression else "[green]ok[/]"
    console.print(f"[green]compare[/] complete: {flag} md={md} html={html}")


@app.command()
def run(
    plan: Annotated[Path, typer.Option("--plan", "-p", help="Path to a run-plan YAML.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Override the plan's output directory."),
    ] = None,
    skip_quality: Annotated[
        bool, typer.Option("--skip-quality", help="Skip lm-eval quality benchmarks.")
    ] = False,
    skip_quantize: Annotated[
        bool, typer.Option("--skip-quantize", help="Skip the NVFP4 candidate stage.")
    ] = False,
    skip_baseline: Annotated[
        bool,
        typer.Option(
            "--skip-baseline",
            help=(
                "Skip the baseline serving phase (use when the host cannot "
                "fit the public checkpoint weights; e.g. a 310B model on "
                "2x B300). The run still quantizes and measures the "
                "candidate; comparison report will be empty since there is "
                "no baseline to diff against."
            ),
        ),
    ] = False,
) -> None:
    """Execute the full pipeline: serve, bench, quantize, re-serve, bench, compare."""
    from lmbench.runner import run_plan_from_file

    def progress(message: str) -> None:
        console.print(f"[cyan]run[/] {message}")

    result = run_plan_from_file(
        plan,
        output_dir=output,
        skip_quality=skip_quality,
        skip_quantize=skip_quantize,
        skip_baseline=skip_baseline,
        progress=progress,
    )
    console.print(
        f"[green]run[/] complete: plan={result.plan_name} "
        f"output_dir={result.output_dir} models={len(result.models)}"
    )
    for m in result.models:
        flag = (
            "[red]REGRESSION[/]"
            if m.comparison.any_regression
            else "[green]ok[/]"
        )
        console.print(
            f"  - {m.model_name}: {flag} "
            f"md={m.report_md} html={m.report_html}"
        )


if __name__ == "__main__":
    app()
