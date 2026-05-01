# lmbench - Product Requirements Document

## Problem

Picking quantization recipes for serving large language models on Blackwell-class hardware is currently guesswork. Vendors publish microbenchmarks; tooling around `vllm` and `nvidia-modelopt` is moving fast; and the existing community harnesses do not produce a clean *baseline vs candidate* delta with both performance and quality side by side.

`lmbench` is a small, opinionated, reproducible Python harness that for each configured model:

1. Serves the **baseline** checkpoint via `vllm serve`.
2. Runs **performance benchmarks** (TTFT, ITL, TPOT, throughput, GPU memory) at a sweep of concurrencies.
3. Runs **quality benchmarks** (MMLU, GSM8K, ARC-C, HellaSwag, TruthfulQA, optional long-context) via `lm-eval --model local-completions`.
4. Quantizes the model to **NVFP4** via `nvidia-modelopt`.
5. Re-serves the quantized checkpoint and re-runs the same benchmarks.
6. Emits a **side-by-side comparison report** (Markdown + interactive Plotly HTML) with regression flags.

The goal is to make a `baseline -> NVFP4` decision a one-command action - not a multi-day investigation.

## Non-goals

- Replacing `lm-eval` or `vllm bench serve` - `lmbench` orchestrates them, it does not reimplement them.
- Training, fine-tuning, or any kind of weight modification beyond post-training quantization.
- Cross-vendor portability. The target is **NVIDIA Blackwell (B100 / B200 / B300)** with optional fall-throughs to older GPUs for sanity, but NVFP4 deltas only mean anything on Blackwell tensor cores.
- A full UI. Reports are static Markdown + HTML.

## Personas

- **Inference platform engineer** - owns vLLM serving config; needs to validate that NVFP4 doesn't crater quality for their model.
- **MLE / model owner** - wants a quick "is FP4 safe for my model?" answer with a defensible quality table.
- **Performance engineer** - wants reproducible TTFT / ITL numbers under controlled concurrency, not noise from a partner-marketing slide.

## Success criteria

- A fresh checkout on a B300 host can run `uv sync --extra all && uv run lmbench run --plan configs/run_smoke.yaml` and produce both reports without intervention.
- Dev-loop tests pass on any host (`uv sync && uv run pytest`) with **no GPU** required.
- Coverage stays at 80%+ on the harness layer.
- The harness is *small enough to read in one sitting* - single-author, no microservices, no orchestration framework.

## Locked defaults (from HANDOFF.md)

| Setting              | Value                                |
|----------------------|--------------------------------------|
| GPU                  | NVIDIA B300 (Blackwell Ultra)        |
| Tensor parallel      | TP = 2                               |
| Serve mode           | online HTTP (`vllm serve`)           |
| Long-context         | included (RULER / LongBench TBD)     |
| Quantization tool    | `nvidia-modelopt` (primary)          |
| Calibration          | `cnn_dailymail` x 512 samples        |
| Concurrency sweep    | `[1, 8, 32, 128]`                    |
| Dashboard            | Plotly HTML                          |
| HumanEval            | opt-in (executes generated code)     |
| Package manager      | `uv`                                 |
| Python               | 3.11                                 |
