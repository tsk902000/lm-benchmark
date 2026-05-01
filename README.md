# lmbench

Reproducible vLLM benchmark harness with NVFP4 quantization comparison.

For each configured model, `lmbench` will:

1. Serve the baseline checkpoint via vLLM (online HTTP mode by default).
2. Run performance benchmarks (TTFT, ITL, TPOT, throughput, GPU memory) and quality benchmarks (MMLU, GSM8K, ARC-C, HellaSwag, long-context suites).
3. Quantize the model to **NVFP4** using NVIDIA TensorRT Model Optimizer.
4. Re-serve the quantized checkpoint and re-run the same benchmarks.
5. Emit a side-by-side comparison report (Markdown + interactive HTML).

The target deployment hardware is the **NVIDIA B300 (Blackwell Ultra)** with **TP=2**. NVFP4 acceleration only materializes on Blackwell tensor cores; on other GPUs the quantization pipeline still runs but the perf delta will be misleading. The harness scaffolding (config, schema, serve wrappers, tests) works on any host — authoring and unit tests do not need a GPU.

## Quickstart from clone (no GPU required)

```bash
git clone https://github.com/<your-org>/lm-benchmark.git
cd lm-benchmark

# Install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Resolve dependencies and create a Python 3.11 venv
uv sync

# Verify the harness
uv run ruff check .
uv run mypy
uv run pytest
```

That's it. ~5 seconds with deps cached, ~30 seconds first time.

`uv sync` reads `pyproject.toml` + `uv.lock` and produces a reproducible `.venv/`. No `pip install` needed; no manual virtualenv activation. All commands run via `uv run …`.

## Quickstart on a GPU host (B300 or any CUDA dev box)

```bash
# After uv sync above — pull in vLLM, lm-eval, modelopt, transformers, etc.
uv sync --extra all

# Run the GPU integration smoke (loads facebook/opt-125m end-to-end)
uv run pytest --gpu

# Phase 8 will wire these up:
uv run lmbench run --plan configs/run_smoke.yaml
uv run lmbench run --plan configs/run_baseline_vs_nvfp4.yaml
```

> The `[gpu]` extra (vllm, lm-eval, transformers, pynvml) and `[quant]` extra (nvidia-modelopt) are Linux/CUDA-only. They are isolated in `pyproject.toml [project.optional-dependencies]` precisely so the dev path stays cross-platform.

## Common commands

| Command | What it does |
|---|---|
| `uv sync` | Install core deps (cross-platform, no GPU). |
| `uv sync --extra all` | Add `[gpu]` + `[quant]` extras (Linux/CUDA only). |
| `uv run pytest` | Unit tests (skips GPU smoke). |
| `uv run pytest --gpu` | Include the GPU smoke (`tests/integration/test_serve_smoke.py`). |
| `uv run pytest -q tests/unit/test_config_schema.py` | Run a single test file. |
| `uv run ruff check .` | Lint. |
| `uv run ruff check --fix .` | Lint and auto-fix. |
| `uv run mypy` | Strict type checking. |
| `uv run lmbench --help` | CLI help (subcommands are stubs through Phase 7). |

GPU tests can also be enabled with `LMBENCH_GPU=1 uv run pytest`. Blackwell-only tests use `--blackwell` or `LMBENCH_BLACKWELL=1`.

## Configuration

All run-time inputs are YAML under `configs/` and validated by the pydantic schema in `src/lmbench/config/schema.py` (frozen models, unknown fields rejected).

```text
configs/
├── models.yaml          # ModelEntry[] — HF ids, vLLM args, gating
├── benchmarks.yaml      # WorkloadSpec[] + EvalSuite (perf workloads + lm-eval tasks)
├── quantization.yaml    # QuantRecipe (NVFP4 default, cnn_dailymail × 512 calibration)
├── hardware.yaml        # HardwareProfile (B300 TP=2 by default)
└── run_smoke.yaml       # Single-document RunPlan (CPU-loadable opt-125m smoke)
```

`${VAR}` and `${VAR:-default}` are interpolated from the environment at load time. Required vars without a default raise loudly.

```yaml
# Example: a token in the env, with a default
hf_id: ${HF_MODEL:-facebook/opt-125m}
```

Set `HF_TOKEN` in `.env` (template at `.env.example`) for gated models like `meta-llama/Llama-3.1-8B-Instruct`.

## Layout

```
configs/        model registry, workload + eval definitions, hardware profiles
src/lmbench/    Python package
  cli.py        typer entrypoint
  config/       pydantic schema + YAML loader + run-plan resolver
  serve/        vLLM server lifecycle, OpenAI-API health probes, offline engine
  bench/        perf + quality benchmark drivers (incl. long-context)   [Phase 4-5]
  quantize/     NVFP4 (modelopt) + alternative llmcompressor recipes    [Phase 6]
  compare/      baseline-vs-candidate diffing + stats                   [Phase 7]
  report/       Markdown + Plotly HTML rendering                        [Phase 7]
  runner/       end-to-end pipeline orchestration                       [Phase 8]
  utils/        gpu telemetry, paths, logging
tests/
  unit/         offline tests (pure Python, no GPU)
  integration/  @pytest.mark.gpu / @pytest.mark.blackwell tests
docs/
  HANDOFF.md    rolling phase-by-phase status and roadmap
```

## Status

Phase 1 — project scaffolding ✅
Phase 2 — config schema + YAML loader + resolver ✅
Phase 3 — vLLM serving layer (subprocess wrapper, health probes, lifecycle) ✅

Up next: Phase 4 perf benchmarks. See `docs/HANDOFF.md` for the full phase roadmap, open questions, and risk register.
