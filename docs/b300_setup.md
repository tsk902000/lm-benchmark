# B300 host setup

This is the recipe for bringing up an NVIDIA B300 (Blackwell Ultra) host so it can run the full `lmbench` pipeline end-to-end.

## Prerequisites

- A Linux distribution that NVIDIA still ships drivers for (Ubuntu 22.04 LTS or newer is the safest choice).
- Root or sudo for the driver install.
- Network egress to HuggingFace (gated models need an `HF_TOKEN`) and to the public CDN that hosts vLLM / modelopt wheels.

## Step 1 - Driver and CUDA

Install a recent enough NVIDIA driver to recognize Blackwell / Blackwell Ultra.

```bash
nvidia-smi
# expect a modern data-center driver with CUDA 12.9+ visible; CUDA 13 is the
# preferred target for B300/GB300 vLLM builds.
```

If `nvidia-smi` does not list a `B300` (or whichever Blackwell SKU you have), the driver is the problem. Do not attempt to skip ahead.

## Step 2 - uv

`lmbench` standardizes on `uv` for dependency resolution. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 3 - Repo + dependencies

```bash
git clone https://github.com/<your-org>/lm-benchmark.git
cd lm-benchmark
uv sync --extra all
```

The `[gpu]` extra pulls vllm / lm-eval / transformers / pynvml. The `[quant]` extra pulls `nvidia-modelopt[all]`. `--extra all` gets both.

## Step 4 - HF token (for gated models)

```bash
cp .env.example .env
echo "HF_TOKEN=<your token>" >> .env
```

`HF_TOKEN` is read by the HF client libraries automatically; nothing in `lmbench` parses it directly.

## Step 5 - Smoke test

```bash
uv run pytest --gpu
```

This boots `facebook/opt-125m` via the real `vllm serve` subprocess and round-trips one completion. If it passes, the GPU stack is healthy.

## Step 6 - Run a real plan

```bash
uv run lmbench run --plan configs/run_smoke.yaml --skip-quantize
```

The smoke plan runs `facebook/opt-125m` only. Once that produces a baseline report:

```bash
# Attempt the V2.5 public-FP8 baseline vs NVFP4 path on 2x B300.
uv run lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml
```

MiMo-V2.5 currently depends on fast-moving vLLM support; use the MiMo-specific or nightly vLLM image recommended by the current vLLM recipe rather than assuming a generic old wheel will work. If the baseline stage fails on your exact stack, rerun the same plan with `--skip-baseline` and compare the candidate against a separately captured public-FP8 baseline.

## Known-good versions

Once we have a green run, document the exact `(vllm, nvidia-modelopt, transformers, driver)` quartet here. Until then, the harness will use whatever `uv sync --extra all` resolves.

## Things that go wrong

- **`vllm serve` exits immediately with `CUDA out of memory`.** Lower `gpu_memory_utilization` (default 0.9) in the model entry.
- **`vllm` hangs at startup for several minutes.** First run compiles CUDA graphs; subsequent runs cache the autotune. Set `enforce_eager: true` in the model entry to disable graph capture if you're debugging.
- **modelopt complains about missing kernels.** You're probably on a CUDA driver that's older than the modelopt prebuilt requires. Upgrade the driver.
