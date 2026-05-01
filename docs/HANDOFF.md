# lmbench — Session Handoff

A self-contained brief so any future session (Windows, macOS, or WSL2/Linux) can pick this up without re-deriving context.

## Project goal

Reproducible Python harness that, for each configured LLM:

1. Serves it via **vLLM** (online HTTP mode, default).
2. Runs **performance benchmarks** (TTFT, ITL, TPOT, throughput, GPU memory) and **quality benchmarks** (MMLU, GSM8K, ARC-C, HellaSwag, TruthfulQA, **plus long-context** suites).
3. Quantizes the model to **NVFP4** via `nvidia-modelopt` (TensorRT Model Optimizer).
4. Re-serves and re-benchmarks the NVFP4 variant on the same hardware.
5. Emits a **side-by-side comparison report** (Markdown + interactive Plotly HTML).

## Target hardware and locked defaults

| Setting | Value |
|---|---|
| GPU | NVIDIA **B300** (Blackwell Ultra) |
| Tensor parallel | **TP = 2** |
| Serve mode default | **online HTTP** (`vllm serve`) |
| Long-context | **included** (RULER / LongBench — TBD) |
| Quantization tool | `nvidia-modelopt` (primary). `llmcompressor` deferred. |
| Calibration | `cnn_dailymail` × 512 samples (default) |
| Concurrency sweep | `[1, 8, 32, 128]` |
| Dashboard | Plotly HTML |
| HumanEval | opt-in (executes code; off by default) |
| Package manager | `uv` |
| Python | 3.11 |

## Status — Phases 1-9 complete (harness done, awaiting first B300 run)

Phase 1 landed (project scaffolding, typer CLI stubs, ruff/mypy/pytest harness, 13 tests, 100% coverage).

Phase 2 landed:

- `src/lmbench/config/schema.py` — frozen pydantic models with `extra="forbid"`:
  `VLLMArgs`, `ModelEntry`, `WorkloadSpec`, `EvalSuite`, `CalibrationSpec`,
  `QuantRecipe`, `HardwareProfile`, `RunPlan`. Cross-cutting validators
  (tp ≤ num_gpus, NVFP4 ⇒ Blackwell, unique model/workload names, humaneval
  opt-in, warmup ≤ num_prompts).
- `src/lmbench/config/loader.py` — YAML loader with `${VAR}` and
  `${VAR:-default}` env-var interpolation. Missing required vars raise
  loudly. Loaders for each section plus `load_run_plan` for single-document
  plans.
- `src/lmbench/config/resolver.py` — `expand_concurrency`,
  `expand_plan_concurrency`, `apply_hardware_defaults` (re-validates after
  substitution), `select_models`.
- `src/lmbench/config/__init__.py` — public surface re-exported.
- `configs/models.yaml`, `configs/benchmarks.yaml`, `configs/quantization.yaml`,
  `configs/hardware.yaml`, `configs/run_smoke.yaml`. Seed registry includes
  `facebook/opt-125m` (CPU-loadable smoke), `meta-llama/Llama-3.1-8B-Instruct`
  (gated), and `Qwen/Qwen2.5-7B-Instruct`.
- `tests/unit/test_config_schema.py` (30 tests), `test_config_loader.py`
  (15 tests), `test_config_resolver.py` (7 tests).

Phase 3 landed:

- `src/lmbench/serve/vllm_server.py` — `ServerHandle` dataclass; pure
  `build_serve_args(entry)` argv builder; `start_vllm_server` spawning
  `vllm serve` in a fresh process group; `is_healthy` / `lists_model` /
  `wait_for_ready` (tenacity exponential backoff over `/health` and
  `/v1/models`); `stop_vllm_server` with SIGTERM→SIGKILL escalation.
- `src/lmbench/serve/vllm_offline.py` — `build_llm_kwargs(entry)` plus
  lazy-import `load_offline_engine(entry)` for in-process generation.
- `src/lmbench/serve/lifecycle.py` — `serve_model(entry)` context manager.
- `src/lmbench/serve/__init__.py` — public surface re-exported.
- `tests/unit/test_serve_args.py` (15), `test_serve_health.py` (12),
  `test_serve_offline.py` (2), `test_serve_lifecycle.py` (10).
- `tests/integration/test_serve_smoke.py` — `@pytest.mark.gpu`, opt-125m
  round-trip via real `vllm serve` subprocess.
- `tests/conftest.py` — `--gpu` / `--blackwell` flags + `LMBENCH_GPU` /
  `LMBENCH_BLACKWELL` env vars gate GPU/Blackwell tests.

Phase 4 landed: `bench/metrics.py` (TTFT/ITL/TPOT + percentile + bootstrap_ci),
`bench/workloads.py` (random / longctx / sharegpt deterministic generators),
`bench/perf.py` (async streaming OpenAI-compatible driver), `utils/gpu.py`
(pynvml sampler thread).

Phase 5 landed: `bench/quality.py` wraps `lm_eval --model local-completions`,
parses `results_*.json` into `QualityResult`, picks a primary metric per task.

Phase 6 landed: `quantize/calibration.py` (cnn_dailymail loader),
`quantize/modelopt_nvfp4.py` (mtq.quantize + save_pretrained + sidecar
metadata), `quantize/verify.py` (sanity probe).

Phase 7 landed: `compare/differ.py` + `compare/stats.py` for deltas with
regression flags + bootstrap CI; `report/markdown.py` + `report/html.py`
for Markdown and Plotly HTML output (with plain-HTML fallback).

Phase 8 landed: `runner/env.py` (env.json snapshot), `runner/pipeline.py`
(full per-model orchestration). `lmbench run --plan ...` is wired through.

Phase 9 landed: `docs/PRD.md`, `docs/architecture.md`,
`docs/nvfp4_workflow.md`, `docs/b300_setup.md`, `docs/troubleshooting.md`.

Test status: **203 unit + 1 GPU integration smoke (skipped without --gpu)**,
**90% coverage**, ruff clean, mypy 0 issues (verified on Linux WSL2 with
`uv 0.7.3` + Python 3.11.11).

Known limitations:
- vLLM internally spawns EngineCore subprocesses in a fresh session;
  `os.killpg` on the spawn pgid does not always reap them. The runner
  should be extended with a `psutil`-based child-walk sweep on teardown.
- `nvfp4_llmcompressor` recipe is reserved in the schema but not wired.
- `configs/run_baseline_vs_nvfp4.yaml` is referenced in README but not
  shipped (write it once the model registry stabilizes).
- First real B300 run still needs to land. Once it does, pin the
  `(vllm, nvidia-modelopt, transformers, driver)` quartet in
  `pyproject.toml` and `docs/b300_setup.md`.

## Resuming in WSL2

```bash
# In WSL2, after `wsl` from PowerShell:
cd /mnt/d/Work/lm-benchmark   # or clone to a Linux-native path for better I/O perf

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Re-create the venv (uv resolves Linux wheels for the Linux platform)
uv sync

# Sanity check
uv run ruff check .
uv run mypy
uv run pytest
```

When you have a GPU host (B300 or any CUDA dev box), install the full set:

```bash
uv sync --extra all   # pulls vllm, lm-eval, modelopt, etc.
```

> Note: `vllm` and `nvidia-modelopt` have **no Windows wheels**. That is why they live in optional extras — `uv sync` on Windows works with just the harness scaffolding.

> Performance tip: developing inside `/mnt/d/...` is slow due to the 9P filesystem layer. Cloning the repo to `~/lm-benchmark` (Linux-native ext4) is materially faster for `uv sync` and pytest.

## Open questions to resolve before Phase 2 wraps

These don't block scaffolding work but they shape the seed configs. Answer in `configs/models.yaml` and `configs/benchmarks.yaml` once Phase 2 lands.

1. **Models for the initial registry.** Concrete HF ids please — examples: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`. Pick 1–3 to start.
2. **Long-context benchmark choice.** RULER (synthetic, controlled) vs LongBench (realistic, slower) vs both. Up to what max sequence length?
3. **HumanEval.** Include or skip? (Executes generated code in a sandbox; default off.)
4. **HF_TOKEN.** Will gated models (Llama family) be used? If yes, plan to set `HF_TOKEN` in `.env`.
5. **Concurrency sweep grid.** Default `[1, 8, 32, 128]` — adjust?
6. **Result retention.** Auto-prune `results/` to last N runs?

## Phase roadmap (remaining)

Each phase is independently mergeable. Tiny-model (`facebook/opt-125m`) smoke runs at the end of each phase from Phase 3 onwards.

### Phase 2 — Config registry and schema (~6h)
- `src/lmbench/config/schema.py` — pydantic models: `ModelEntry`, `VLLMArgs`, `WorkloadSpec`, `EvalSuite`, `QuantRecipe`, `HardwareProfile`, `RunPlan`. Validation (e.g. `tp_size` divides head count when known, `max_model_len <= model_max`).
- `src/lmbench/config/loader.py` — YAML loader with env-var interpolation.
- `src/lmbench/config/resolver.py` — hardware profile merge, sweep expansion.
- `configs/models.yaml`, `configs/benchmarks.yaml`, `configs/quantization.yaml`, `configs/hardware.yaml`. Seed with `facebook/opt-125m` for smoke + 1–3 production models.
- Tests: `tests/unit/test_config_schema.py`.

### Phase 3 — vLLM serving layer (~10h)
- `src/lmbench/serve/vllm_server.py` — subprocess wrapper for `vllm serve`. Health-probe `/health` and `/v1/models` with `tenacity` backoff. `ServerHandle` dataclass.
- `src/lmbench/serve/vllm_offline.py` — wraps `vllm.LLM(...)`.
- `src/lmbench/serve/lifecycle.py` — context manager for clean startup/teardown, kills orphaned processes on failure.
- `tests/integration/test_serve_smoke.py` — `@pytest.mark.gpu`, opt-125m round-trip.

### Phase 4 — Performance benchmarks (~10h)
- `src/lmbench/bench/workloads.py` — ShareGPT loader, synthetic random prompts (controlled in/out lengths), long-context generator (variable up to `max_model_len`). Deterministic seeded sampling.
- `src/lmbench/bench/perf.py` — wraps `vllm bench serve` (or `benchmark_serving.py`). Sweeps concurrency. Captures warmup separately.
- `src/lmbench/bench/metrics.py` — TTFT, ITL, TPOT, e2e latency, percentile math. Pure functions.
- `src/lmbench/utils/gpu.py` — `pynvml` sampler thread for peak/steady memory, SM util, power.
- Tests: `tests/unit/test_metrics.py`, `tests/unit/test_workloads.py`.

### Phase 5 — Quality benchmarks (~8h)
- `src/lmbench/bench/quality.py` — drives `lm-eval` via `local-completions` backend pointed at the running vLLM server. MMLU, GSM8K, HumanEval (gated), ARC-C, HellaSwag, TruthfulQA. Persists raw + summary JSON.
- Long-context option (RULER or LongBench subset, gated by `max_model_len`).
- Tests: `tests/unit/test_quality_parser.py`.

### Phase 6 — NVFP4 quantization (~12h, B300 needed for full validation)
- `src/lmbench/quantize/calibration.py` — `cnn_dailymail` 512 samples default. Pluggable. Configurable seq-len, sample count, seed.
- `src/lmbench/quantize/modelopt_nvfp4.py` — load HF model -> `mtq.quantize` with NVFP4 config -> calibrate -> export checkpoint in vLLM-consumable format. Handle TP-aware export.
- `src/lmbench/quantize/llmcompressor_nvfp4.py` — alternative path (deferred unless v1 needs it).
- `src/lmbench/quantize/verify.py` — load quantized checkpoint into vLLM, run a fixed prompt, fail fast on degenerate output.
- Tests: `tests/unit/test_quantize_config.py`, `tests/integration/test_quantize_tiny.py` (`@pytest.mark.blackwell`).
- **Pin tested `(vllm, modelopt)` version pair** in `pyproject.toml` and add a runtime preflight check.

### Phase 7 — Compare and report (~6h)
- `src/lmbench/compare/differ.py` — pure `(baseline, candidate) -> ComparisonReport`. Absolute + relative deltas. Threshold-based regression flags.
- `src/lmbench/compare/stats.py` — bootstrap CI for quality deltas (combine with lm-eval stderr).
- `src/lmbench/report/markdown.py` — side-by-side tables (perf per concurrency, quality per task, memory, env summary).
- `src/lmbench/report/html.py` — Plotly figures (latency CDF, throughput vs concurrency, quality bar chart, memory time series).
- Tests: `tests/unit/test_differ.py`, `tests/unit/test_report.py`.

### Phase 8 — Pipeline orchestration (~6h)
- `src/lmbench/runner/pipeline.py` — for each model in plan: capture env -> serve baseline -> bench perf -> bench quality -> teardown -> quantize -> serve nvfp4 -> bench perf -> bench quality -> teardown -> compare -> report. Mid-pipeline checkpoint/resume.
- `src/lmbench/runner/env.py` — capture `nvidia-smi`, driver, CUDA, vllm/modelopt/transformers versions, GPU SKU + clocks, OS, git SHA -> `env.json`.
- `scripts/run_pipeline.sh`, `scripts/setup_b300.sh` (preflight: validates CUDA / driver / vLLM versions), `scripts/ci_smoke.sh`.

### Phase 9 — Documentation (~4h)
- `docs/PRD.md`, `docs/architecture.md`, `docs/nvfp4_workflow.md`, `docs/b300_setup.md`, `docs/troubleshooting.md`.

### Total remaining: ~62h

## Risk register (top items)

| Risk | Mitigation |
|---|---|
| `nvidia-modelopt` <-> vLLM version drift breaks the export-import handshake | Pin tested pair in `pyproject.toml`. Runtime preflight asserts versions. Document known-good combos in `docs/nvfp4_workflow.md`. |
| NVFP4 only accelerates on Blackwell | Hardware profile gates `quantization=modelopt_fp4` runs to Blackwell. Harness refuses to publish a "comparison" report from non-Blackwell hardware. |
| Calibration dataset choice biases NVFP4 quality | Default `cnn_dailymail` 512 samples; expose in `quantization.yaml`; record calibration metadata in run artifacts. |
| Long-context perf workloads OOM or run multi-hour | Token-budget guard on workload generator. |
| Reproducibility drift (warmup, JIT, autotuner) | Mandatory warmup phase reported separately; fixed seeds; capture vLLM autotune cache fingerprint. |
| HumanEval executes model output as code | Disabled by default; explicit opt-in flag. |

## Pointers

- Plan source: `docs/HANDOFF.md` (this file).
- Project root: `D:\Work\lm-benchmark` on Windows; `/mnt/d/Work/lm-benchmark` from WSL2 (or clone to `~/lm-benchmark` for ext4 perf).
- Repo style: terse PEP 8, `from __future__ import annotations`, `Annotated[...]` typer pattern.
- Verify before each commit: `uv run ruff check . && uv run mypy && uv run pytest`.
