# lmbench

Reproducible vLLM benchmark harness with NVFP4 quantization comparison.

For each configured model, `lmbench` will:

1. Serve the baseline checkpoint via vLLM (online HTTP, OpenAI-compatible).
2. Run **performance benchmarks** (TTFT, ITL, TPOT, throughput, GPU memory) at a sweep of concurrencies.
3. Run **quality benchmarks** (MMLU, GSM8K, ARC-C, HellaSwag, TruthfulQA + RULER, LongBench v2, LiveCodeBench) via `lm-eval --model local-completions`.
4. Quantize the model to **NVFP4** using NVIDIA TensorRT Model Optimizer (`mtq.quantize` + unified HF export + `cnn_dailymail` x 512 starter calibration).
5. Re-serve the quantized checkpoint and re-run the same benchmarks.
6. Emit a side-by-side comparison report (Markdown + interactive Plotly HTML) with regression flags.

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

# Pure-baseline smoke (CPU-loadable opt-125m, no NVFP4)
uv run lmbench run --plan configs/run_smoke.yaml --skip-quantize

# Public-FP8-checkpoint -> NVFP4 attempt against MiMo-V2.5 on 2x B300
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml
```

> The `[gpu]` extra (vllm, lm-eval, transformers, pynvml) and `[quant]` extra (nvidia-modelopt) are Linux/CUDA-only. They are isolated in `pyproject.toml [project.optional-dependencies]` precisely so the dev path stays cross-platform.

### Install vLLM

Use a Linux CUDA environment for vLLM. Upstream vLLM does not support native Windows; use a Linux host or WSL2.

The normal repo-managed path is:

```bash
uv sync --extra all
uv run python -c "import torch, vllm; print('cuda=', torch.version.cuda); print('vllm=', vllm.__version__)"
uv run vllm --help
```

If the resolved wheel does not match the B300 driver/CUDA stack, refresh vLLM inside the same `.venv` using vLLM's CUDA-aware installer:

```bash
uv pip install -U vllm --torch-backend=auto
uv run python -c "import torch, vllm; print(torch.cuda.get_device_name(0)); print(vllm.__version__)"
```

For MiMo-V2.5 specifically, the current vLLM recipe says stable vLLM does not yet support the model and recommends the MiMo-tagged image:

```bash
docker pull vllm/vllm-openai:mimov25-cu129
```

To use that image with this harness, run `lmbench` inside an equivalent container environment, or install the matching MiMo/nightly vLLM wheel in the `.venv` before running:

```bash
uv run pytest --gpu
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml
```

### 2x B300 + MiMo-V2.5

The public MiMo-V2.5 checkpoint is already FP8, so this is not a BF16 baseline-vs-NVFP4 run. On a 2x B300 host, try the public-FP8-checkpoint-vs-NVFP4 plan first:

```bash
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml
```

This TP=2 setup is memory-plausible but not validated in this repo yet; the current vLLM recipe uses a MiMo-specific/nightly image and TP=4 on H200-class memory. If the baseline fails to serve on your exact stack, rerun with `--skip-baseline` to collect NVFP4 candidate-only data and compare against a separately captured FP8 baseline or the published references in `docs/published_baselines/mimo_v2_5.md`.

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
| `uv run lmbench --help` | CLI help. |
| `uv run lmbench run --plan <plan.yaml>` | Run the full pipeline. |
| `uv run lmbench run --plan <plan.yaml> --no-stream-vllm-logs` | Run quietly without vLLM download/startup logs in the terminal. |
| `uv run lmbench run --plan <plan.yaml> --skip-quality` | Skip lm-eval; perf only. |
| `uv run lmbench run --plan <plan.yaml> --skip-quantize` | Baseline only; no NVFP4 stage. |
| `uv run lmbench run --plan <plan.yaml> --skip-baseline` | NVFP4 only; for hosts that can't fit the baseline checkpoint. |
| `uv run lmbench compare --baseline <dir> --candidate <dir> -o <reports>` | Diff two prior stage directories offline, such as `<model>/baseline` and `<model>/quantized`. |

GPU tests can also be enabled with `LMBENCH_GPU=1 uv run pytest`. Blackwell-only tests use `--blackwell` or `LMBENCH_BLACKWELL=1`.

## Configuration

All run-time inputs are YAML under `configs/` and validated by the pydantic schema in `src/lmbench/config/schema.py` (frozen models, unknown fields rejected).

```text
configs/
├── models.yaml                   # ModelEntry[] — HF ids, vLLM args, gating
├── benchmarks.yaml               # WorkloadSpec[] + EvalSuite (perf + lm-eval + long_context)
├── quantization.yaml             # QuantRecipe (NVFP4 default, cnn_dailymail x 512 calibration)
├── hardware.yaml                 # HardwareProfile (B300 TP=2 by default)
├── run_smoke.yaml                # CPU-loadable opt-125m smoke (no NVFP4)
└── run_mimo_v2_5_nvfp4.yaml      # MiMo-V2.5 public FP8 checkpoint vs NVFP4 attempt
```

`${VAR}` and `${VAR:-default}` are interpolated from the environment at load time. Required vars without a default raise loudly.

```yaml
# Example: a token in the env, with a default
hf_id: ${HF_MODEL:-facebook/opt-125m}
```

Set `HF_TOKEN` in `.env` (template at `.env.example`) if you later add gated Hugging Face models.

### Long-context + coding tasks

`EvalSuite.long_context` is merged into the `lm_eval --tasks` argv via `bench.quality.merged_task_list`. The default suite enables RULER, LongBench v2, and LiveCodeBench. Whether each runs depends on the installed `lm-eval` version supporting that task; lm-eval surfaces a clear error otherwise.

> Out of scope for this harness (would need separate Docker / shell / multi-turn drivers): SWE-bench Verified/Pro, Terminal-Bench, tau2-bench, Aider Polyglot. Run those upstream against the same vLLM endpoint your `lmbench` run uses.

## Layout

```
configs/        model registry, workload + eval definitions, hardware profiles
src/lmbench/    Python package
  cli.py        typer entrypoint (serve / bench / quantize / compare / run)
  config/       pydantic schema + YAML loader + run-plan resolver
  serve/        vLLM server lifecycle, OpenAI-API health probes, offline engine
  bench/        perf + quality benchmark drivers (incl. long-context)
  quantize/     NVFP4 (modelopt) + post-quant verifier
  compare/      baseline-vs-candidate diffing + bootstrap CI
  report/       Markdown + Plotly HTML rendering
  runner/       env capture + end-to-end pipeline orchestration
  utils/        gpu telemetry (pynvml sampler thread)
tests/
  unit/         offline tests (pure Python, no GPU)
  integration/  @pytest.mark.gpu / @pytest.mark.blackwell tests
docs/
  HANDOFF.md                    rolling phase-by-phase status
  PRD.md                        product requirements
  architecture.md               module map + lifecycle invariants
  nvfp4_workflow.md             quantize step internals + tuning
  b300_setup.md                 driver / uv / HF token / smoke recipe
  troubleshooting.md            known failure modes
  published_baselines/
    mimo_v2_5.md                published MiMo-V2.5 scores for offline comparison
summary.md                      one-page session summary at the repo root
```

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project scaffolding | ✅ |
| 2 | Config schema + YAML loader + resolver | ✅ |
| 3 | vLLM serving layer (subprocess wrapper, health probes, lifecycle) | ✅ |
| 4 | Performance benchmarks (async streaming driver, GPU sampler) | ✅ |
| 5 | Quality benchmarks (`lm-eval` local-completions wrapper) | ✅ |
| 6 | NVFP4 quantization (modelopt + post-quant verifier) | ✅ |
| 7 | Compare + report (deltas, bootstrap CI, Markdown / Plotly HTML) | ✅ |
| 8 | Runner + CLI wiring (`--skip-quality`, `--skip-quantize`, `--skip-baseline`) | ✅ |
| 9 | Documentation | ✅ |

**Tests:** 216 passed + 1 GPU integration smoke skipped because `vllm` is not on this Windows host. 90% coverage. ruff and mypy clean. No B300/GPU benchmark has been run in this workspace.

**Up next:** first real B300 run. Once that lands, pin the `(vllm, nvidia-modelopt, transformers, driver)` quartet in `pyproject.toml` and `docs/b300_setup.md`.

See `docs/HANDOFF.md` for the rolling status, open questions, and risk register, or `summary.md` for a single-page session recap.
