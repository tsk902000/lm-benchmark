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
