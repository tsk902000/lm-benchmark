"""lmbench CLI entry point.

Phase 1: subcommand stubs that validate argument wiring. Real implementations
land in subsequent phases (serve in Phase 3, bench in Phase 4/5, quantize in
Phase 6, compare in Phase 7, run in Phase 8).
"""

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
    console.print(
        f"[yellow]compare[/] (stub) baseline={baseline} candidate={candidate} output={output}"
    )


@app.command()
def run(
    plan: Annotated[Path, typer.Option("--plan", "-p", help="Path to a run-plan YAML.")],
) -> None:
    """Execute the full pipeline: serve → bench → quantize → serve → bench → compare."""
    console.print(f"[yellow]run[/] (stub) plan={plan}")


if __name__ == "__main__":
    app()
