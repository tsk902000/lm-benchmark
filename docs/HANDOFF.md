# lmbench - Session Handoff

This file is the current pickup point for future work. It replaces the older
phase-roadmap notes, which had drifted from the actual code and MiMo release
state.

## Project Goal

`lmbench` is a reproducible vLLM benchmark harness for:

1. Serving a baseline checkpoint with `vllm serve`.
2. Running performance workloads with TTFT, ITL, TPOT, throughput, and GPU
   telemetry.
3. Running quality workloads through `lm-eval --model local-completions`.
4. Quantizing with NVIDIA TensorRT Model Optimizer NVFP4.
5. Re-serving the candidate and writing Markdown/HTML comparison reports.

## Current Verification Status

Validated locally on this Windows host, without B300 hardware:

```bash
uv run pytest -q      # 216 passed, 1 skipped
uv run ruff check .   # clean
uv run mypy           # clean
```

The skipped test is the GPU integration smoke because `vllm` is not installed
on this host. No real B300 benchmark or ModelOpt NVFP4 export has been run in
this workspace.

## Important Corrections

- Public `XiaomiMiMo/MiMo-V2.5` is an FP8 checkpoint, not a BF16 checkpoint.
  Any accuracy-loss statement from the public artifact is public-FP8-checkpoint
  -> NVFP4, not BF16 -> NVFP4.
- A 2x B300 host has 576 GB HBM if each GPU is 288 GB. That makes a public FP8
  MiMo-V2.5 baseline memory-plausible at TP=2, but this repo has not proven it.
  The current vLLM recipe uses a MiMo-specific/nightly image and TP=4 on H200.
- The ModelOpt export path must use unified HF export:
  `modelopt.torch.export.export_hf_checkpoint(...)`, and the output should
  contain `hf_quant_config.json`.
- `lm-eval` only accepts one global `--num_fewshot` per invocation, so mixed
  few-shot suites must be split into groups.

## Shipped Plans

`configs/run_mimo_v2_5_nvfp4.yaml`

- Runs `XiaomiMiMo/MiMo-V2.5`.
- Intended as a public-FP8-checkpoint -> NVFP4 attempt on 2x B300 with TP=2.
- Run without `--skip-baseline` first on the real host.
- If the FP8 baseline stage fails on the exact vLLM/driver stack, rerun with
  `--skip-baseline` to collect candidate-only data.

## Notable Implementation Notes

- `bench.quality.run_quality()` now splits mixed few-shot suites and merges the
  parsed task scores into one `quality_summary.json`.
- `runner.pipeline` passes a real `GPUSampler` into perf workloads and writes
  `gpu_summary` into perf JSON artifacts.
- `lmbench compare` is now functional for saved stage directories such as
  `<model>/baseline` and `<model>/quantized`.
- Windows lifecycle tests patch POSIX-only signal/process-group behavior so the
  offline unit suite is cross-platform.

## Commands For A B300 Host

```bash
uv sync --extra all
uv run pytest --gpu

# Attempt MiMo-V2.5 public FP8 vs NVFP4.
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml
```

If MiMo-V2.5 baseline serving fails:

```bash
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml --skip-baseline
```

Offline comparison of two saved stages:

```bash
uv run lmbench compare \
  --baseline results/<run>/<model>/baseline \
  --candidate results/<run>/<model>/quantized \
  --output reports/<model>
```

## Remaining Risks

- First real B300 validation is still outstanding.
- MiMo-V2.5 support is tied to fast-moving vLLM MiMo-specific/nightly builds.
- ModelOpt NVFP4 may need custom ignore rules for routers/gates, attention
  sink paths, multimodal encoders, embeddings, or the lm head.
- The default `cnn_dailymail` calibration is only a starter. For publishable
  low-loss numbers, use representative MiMo chat, code, math, tool, and
  long-context calibration samples.
- `nvfp4_llmcompressor` is still only reserved in the schema; it is not wired.

## Key Files

| Path | Purpose |
|---|---|
| `summary.md` | Root-level correctness summary and current verdict. |
| `configs/run_mimo_v2_5_nvfp4.yaml` | V2.5 public FP8 -> NVFP4 2x B300 attempt. |
| `docs/nvfp4_workflow.md` | ModelOpt export path and tuning checklist. |
| `docs/b300_setup.md` | Host setup and run sequence. |
| `docs/published_baselines/mimo_v2_5.md` | Published MiMo reference scores. |
| `src/lmbench/runner/pipeline.py` | End-to-end runner orchestration. |
| `src/lmbench/quantize/modelopt_nvfp4.py` | ModelOpt NVFP4 quantization/export. |
| `src/lmbench/bench/quality.py` | lm-eval wrapper and few-shot grouping. |
| `src/lmbench/compare/offline.py` | Offline comparison artifact loader. |
