# lmbench - correctness summary

## Verdict

The original idea is useful, but the earlier write-up overstated what the
repo could prove. After verification, the safe statement is:

> For public MiMo checkpoints, this project can benchmark public FP8 checkpoint
> -> NVFP4 loss and speed, not BF16 -> NVFP4 loss, unless you provide a true
> BF16 checkpoint and a matching baseline run.

The code now reflects that. The only XiaomiMiMo production target in this repo
is `XiaomiMiMo/MiMo-V2.5`. ModelOpt export uses the current unified-HF path,
mixed few-shot quality tasks are handled correctly, GPU telemetry is written
into perf artifacts, and offline comparison is wired.

## What Was Verified

Local verification on this Windows machine:

```bash
uv run pytest -q      # 216 passed, 1 skipped
uv run ruff check .   # clean
uv run mypy           # clean
```

The skipped test is the real `vllm serve` GPU smoke because this machine does
not have the vLLM/GPU stack. No 2x B300 benchmark has been run here.

## Key Corrections

- `XiaomiMiMo/MiMo-V2.5` is published with FP8 quantization metadata
  (`quant_method: fp8`, `store_dtype: fp8`).
- A 2x B300 host has about 576 GB HBM total if each B300 is 288 GB. A public
  FP8 MiMo-V2.5 baseline is memory-plausible at TP=2, but still unvalidated.
  The current vLLM recipe uses a MiMo-specific/nightly image and TP=4 on H200.
- `--skip-baseline` should be a fallback or candidate-only mode, not the default
  assumption for public FP8 MiMo-V2.5.
- Public-FP8 -> NVFP4 numbers should not be described as BF16 -> NVFP4 accuracy
  loss.

## How To Run

Attempt MiMo-V2.5 public FP8 vs NVFP4 on the 2x B300 host:

```bash
uv sync --extra all
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml
```

If the FP8 baseline stage fails on the exact vLLM/driver stack:

```bash
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml --skip-baseline
```

Offline comparison of saved stages is now real:

```bash
uv run lmbench compare \
  --baseline results/<run>/<model>/baseline \
  --candidate results/<run>/<model>/quantized \
  --output reports/<model>
```

## Low-Loss NVFP4 Strategy For MiMo

1. Start with the conservative 32K-context MiMo-V2.5 plan before increasing
   context length or concurrency.
2. Keep calibration representative: MiMo chat templates, code, math, tool-use,
   and long-context samples. `cnn_dailymail` x 512 is only a starter default.
3. Increase calibration samples to 1024+ and raise calibration sequence length
   if loss appears on long-context tasks.
4. Start with sensitive modules excluded from quantization: routers/gates,
   embeddings, lm head, attention-sink/bias paths, and multimodal encoders.
5. Compare the same tasks, same prompt limits, same few-shot settings, same
   vLLM image, and same driver stack. Otherwise the accuracy delta is polluted.

## 2x B300 Settings

The current 2x B300 plan is a reasonable first attempt, not a proven recipe:

- `tensor_parallel_size: 2`
- `dtype: auto`
- `kv_cache_dtype: fp8`
- `max_model_len: 32768` for the first validation run
- `gpu_memory_utilization: 0.9`
- `enable_prefix_caching: false` for consistent benchmark measurements
- `generation-config: vllm`
- `max-num-batched-tokens: 32768`

For full 1M context, memory pressure will be KV-cache dominated. Validate the
32K plan first, then scale context and concurrency separately.

## Benchmark Correctness

Fixed issues:

- Mixed few-shot quality suites are split into multiple `lm_eval` invocations,
  instead of applying the largest few-shot count to every task.
- Perf JSON artifacts include `gpu_summary`.
- `lmbench compare` reads saved artifacts and writes Markdown/HTML reports.
- The project now has a single XiaomiMiMo run plan, so stage skips only affect
  MiMo-V2.5.

Remaining limitations:

- SWE-bench Pro and Terminal-Bench 2 are not covered by this harness; they need
  separate official benchmark harnesses.
- RULER, LongBench, and LiveCodeBench depend on the installed `lm-eval` version.
- First real B300 run still needs to pin the known-good `(vllm, modelopt,
  transformers, driver)` quartet.

## Main Files

| Path | Purpose |
|---|---|
| `configs/run_mimo_v2_5_nvfp4.yaml` | V2.5 public FP8 -> NVFP4 2x B300 attempt. |
| `src/lmbench/quantize/modelopt_nvfp4.py` | ModelOpt NVFP4 quantize/export. |
| `src/lmbench/bench/quality.py` | lm-eval wrapper and few-shot grouping. |
| `src/lmbench/runner/pipeline.py` | Runner and GPU telemetry wiring. |
| `src/lmbench/compare/offline.py` | Offline artifact comparison. |

## Sources Checked

- https://huggingface.co/XiaomiMiMo/MiMo-V2.5/blob/main/config.json
- https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5
- https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html
- https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html
