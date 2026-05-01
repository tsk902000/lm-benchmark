# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Phase 1: project scaffolding (uv project, ruff/mypy/pytest config, typer CLI skeleton).
- Phase 2: configuration surface — pydantic schema (`VLLMArgs`, `ModelEntry`,
  `WorkloadSpec`, `EvalSuite`, `CalibrationSpec`, `QuantRecipe`,
  `HardwareProfile`, `RunPlan`), YAML loader with `${VAR}` / `${VAR:-default}`
  env interpolation, and resolver utilities (`expand_concurrency`,
  `expand_plan_concurrency`, `apply_hardware_defaults`, `select_models`).
- Seed configs under `configs/`: `models.yaml`, `benchmarks.yaml`,
  `quantization.yaml`, `hardware.yaml`, `run_smoke.yaml` (smoke plan with
  `facebook/opt-125m`).
- Phase 2 tests: 52 new unit tests across schema/loader/resolver. Full suite
  at 65 tests, 100% coverage, ruff/mypy clean.
- Phase 3: vLLM serving layer — `src/lmbench/serve/vllm_server.py`
  (`build_serve_args`, `start_vllm_server`, `wait_for_ready` with tenacity
  backoff, `stop_vllm_server` with SIGTERM→SIGKILL escalation),
  `vllm_offline.py` (lazy-imported `vllm.LLM` wrapper),
  `lifecycle.py` (`serve_model` context manager).
- Phase 3 tests: 39 new unit tests (argv builder, health probes, offline
  kwargs, lifecycle subprocess management) plus a GPU integration smoke
  (`tests/integration/test_serve_smoke.py`). Full suite at 105 tests,
  97.5% coverage. Conftest gates `@pytest.mark.gpu` / `@pytest.mark.blackwell`
  behind `--gpu` / `--blackwell` flags or `LMBENCH_GPU=1` / `LMBENCH_BLACKWELL=1`
  env vars.
- README: from-clone quickstart, GPU host quickstart, common-commands table.
- Phase 4: perf benchmarks. `bench/metrics.py` (RequestSample, PerfSummary,
  percentile, latency_stats, summarize, bootstrap_ci), `bench/workloads.py`
  (random / longctx / sharegpt prompt generators with deterministic seeded
  sampling), `bench/perf.py` (async streaming OpenAI-compatible
  /v1/completions driver with concurrency semaphore, warmup split, GPU
  sampler integration), `utils/gpu.py` (pynvml sampler thread with no-op
  fallback). 39 new unit tests.
- Phase 5: quality benchmarks. `bench/quality.py` wraps `lm_eval --model
  local-completions`, parses `results_*.json` into a normalized
  `QualityResult`, picks a primary metric per task (`acc_norm` > `acc` >
  `exact_match` > `f1` > ...). 11 new unit tests.
- Phase 6: NVFP4 quantization. `quantize/calibration.py` (cnn_dailymail
  loader with lazy `datasets` import), `quantize/modelopt_nvfp4.py`
  (`mtq.quantize` + `save_pretrained` + sidecar `lmbench_quant_meta.json`),
  `quantize/verify.py` (post-quantization sanity probe rejecting empty /
  degenerate completions). 21 new unit tests.
- Phase 7: compare + report. `compare/differ.py` (MetricDelta, diff_perf,
  diff_quality, ComparisonReport with regression flags),
  `compare/stats.py` (delta bootstrap CI), `report/markdown.py` (side-by-
  side tables with regression banner), `report/html.py` (Plotly tables
  with plain-HTML fallback). 18 new unit tests.
- Phase 8: runner + CLI wiring. `runner/env.py` (env.json snapshot:
  nvidia-smi, git SHA, package versions), `runner/pipeline.py`
  (capture env -> serve baseline -> bench -> quantize -> verify -> serve
  candidate -> bench -> compare -> report). `lmbench run --plan ...` is
  now wired through to the real pipeline. 8 new unit tests.
- Phase 9: documentation. `docs/PRD.md`, `docs/architecture.md`,
  `docs/nvfp4_workflow.md`, `docs/b300_setup.md`, `docs/troubleshooting.md`.

Test status: **203 unit tests + 1 GPU integration smoke (skipped without
`--gpu`)**, **90% coverage**, ruff clean, mypy 0 issues.

### Phase 5+ — long-context + coding tasks via lm-eval

- `bench/quality.py` now exposes `merged_task_list(suite)` and merges
  `EvalSuite.long_context` into the lm-eval `--tasks` argv. Previously the
  field was schema-only and silently ignored at runtime.
- `configs/benchmarks.yaml` enables RULER, LongBench v2, and LiveCodeBench
  in the default suite's `long_context` field. These run through the same
  `lm_eval --model local-completions` driver as the standard tasks; whether
  each one runs depends on the installed lm-eval version supporting that
  task (otherwise lm-eval surfaces a clear error).
- 4 new unit tests covering merge dedup, argv inclusion, and seed-config
  wiring. Test suite at 207 unit tests, 90% coverage.

Out of scope (would need separate harness modules; deferred): SWE-bench
Verified / Pro, Terminal-Bench, tau2-bench, Aider Polyglot - these require
Docker patch evaluation, shell session drivers, or simulated-user
multi-turn dialogue and don't fit the lm-eval `local-completions` model.
