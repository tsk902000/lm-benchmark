# lmbench - Architecture

## Module map

```
src/lmbench/
  cli.py              # typer entrypoint - serve / bench / quantize / compare / run
  config/             # YAML config: pydantic schema + loader + resolver
    schema.py         # frozen models: VLLMArgs, ModelEntry, WorkloadSpec,
                      # EvalSuite, CalibrationSpec, QuantRecipe,
                      # HardwareProfile, RunPlan
    loader.py         # YAML reader with ${VAR} / ${VAR:-default} interpolation
    resolver.py       # expand_concurrency, apply_hardware_defaults, select_models
  serve/              # vLLM lifecycle wrappers
    vllm_server.py    # build_serve_args, ServerHandle, start/stop, health probes
    vllm_offline.py   # in-process vllm.LLM(...) wrapper (lazy import)
    lifecycle.py      # serve_model context manager
  bench/              # perf + quality drivers
    metrics.py        # RequestSample, PerfSummary, percentile / bootstrap_ci
    workloads.py      # random / longctx / sharegpt prompt generators
    perf.py           # async streaming OpenAI-compatible /v1/completions driver
    quality.py        # lm-eval local-completions wrapper + results parser
  quantize/           # NVFP4 PTQ
    calibration.py    # cnn_dailymail loader (lazy datasets import)
    modelopt_nvfp4.py # mtq.quantize wrapper + checkpoint export
    verify.py         # post-quantization sanity probe
  compare/            # baseline-vs-candidate diffing
    differ.py         # MetricDelta, diff_perf, diff_quality, ComparisonReport
    stats.py          # bootstrap CI on (baseline, candidate) per-sample deltas
  report/             # Markdown + Plotly HTML rendering
    markdown.py       # render_markdown / write_markdown
    html.py           # render_html / write_html (Plotly with plain-HTML fallback)
  runner/             # end-to-end orchestration
    env.py            # capture nvidia-smi / git SHA / package versions -> env.json
    pipeline.py       # serve -> bench -> quantize -> verify -> serve -> bench -> compare
  utils/
    gpu.py            # pynvml sampler thread (peak / steady memory, SM util, power)
```

## Pipeline flow (per model)

```
+-- env.json (env capture)
|
+-- baseline/
|     +-- perf/<workload>_c<concurrency>.json
|     +-- quality/quality_summary.json
|     +-- quality/results_*.json (raw lm-eval)
|
+-- quantized/
|     +-- (only if quant_recipe set)
|     +-- perf/<workload>_c<concurrency>.json
|     +-- quality/quality_summary.json
|
+-- comparison/
      +-- report.md
      +-- report.html
```

The runner is purely sequential per-model: serve baseline -> measure -> tear down -> quantize -> verify -> serve quantized -> measure -> tear down -> compare -> report. No async at the orchestration layer; the inner perf driver does its own asyncio for concurrent HTTP requests.

## Lifecycle invariants

- **Server isolation.** Every `vllm serve` lives in its own process group (`start_new_session=True`). Teardown sends SIGTERM to the pgid, then SIGKILL after `shutdown_timeout_s`. The lifecycle context manager teardown is in a `finally` so even crashed body code reaps the server.
- **Frozen config.** All pydantic models are `frozen=True` and `extra="forbid"`. Typos in config files fail at load, not at runtime.
- **Determinism.** Workload generation uses `random.Random(seed)` exclusively. No `random.random()` global state.
- **No silent failures in quantize.** `verify_checkpoint` runs immediately after `quantize_to_nvfp4`; the runner aborts the entire model run if the quantized checkpoint produces empty / degenerate output.

## What is _not_ in this repo

- No CI configuration. The `scripts/` directory is intentionally minimal; CI lives wherever the platform engineer hosts it.
- No experiment tracker integration. `lmbench` writes flat files; integrating with W&B / MLflow is left to the caller.
- No model serving in production. `lmbench` is a benchmark harness, not an inference deployment.
