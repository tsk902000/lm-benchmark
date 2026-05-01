# lmbench - session summary

## What this repo is

`lmbench` is a small, opinionated, reproducible Python harness that for each configured LLM:

1. Serves the **baseline** checkpoint via `vllm serve` (online HTTP, OpenAI-compatible).
2. Runs **performance benchmarks** (TTFT, ITL, TPOT, throughput, GPU memory) at a sweep of concurrencies.
3. Runs **quality benchmarks** (MMLU, GSM8K, ARC-C, HellaSwag, TruthfulQA + RULER, LongBench v2, LiveCodeBench) via `lm-eval --model local-completions`.
4. Quantizes the model to **NVFP4** via `nvidia-modelopt` (cnn_dailymail x 512 calibration).
5. Re-serves the quantized checkpoint and re-runs the same benchmarks.
6. Emits a **side-by-side comparison report** (Markdown + interactive Plotly HTML) with regression flags.

The goal is to make a `baseline -> NVFP4` decision a one-command action, not a multi-day investigation.

## Status

All 9 phases of `docs/HANDOFF.md` are shipped:

| Phase | What it ships |
|---|---|
| 1 | uv project, ruff/mypy/pytest harness, typer CLI skeleton |
| 2 | frozen pydantic schema (`VLLMArgs`, `ModelEntry`, `WorkloadSpec`, `EvalSuite`, `CalibrationSpec`, `QuantRecipe`, `HardwareProfile`, `RunPlan`), YAML loader with `${VAR}` / `${VAR:-default}` interpolation, resolver utilities |
| 3 | vLLM lifecycle wrappers (subprocess in fresh process group, health probes via httpx + tenacity, `serve_model` context manager, offline `vllm.LLM` wrapper) |
| 4 | perf benchmarks (async streaming OpenAI driver, deterministic workload generators, pynvml GPU sampler, percentile / bootstrap_ci pure functions) |
| 5 | quality benchmarks (`lm-eval` wrapper, results parser picking primary metric per task) |
| 6 | NVFP4 PTQ (cnn_dailymail loader, `mtq.quantize` + `save_pretrained` + `lmbench_quant_meta.json` sidecar, post-quant verifier rejecting empty / degenerate output) |
| 7 | compare + report (MetricDelta with regression flags, bootstrap-CI on per-sample deltas, Markdown side-by-side tables, Plotly HTML with plain-HTML fallback) |
| 8 | runner + CLI wiring (`runner/env.py` env.json snapshot, `runner/pipeline.py` full orchestration, CLI flags: `--skip-quality`, `--skip-quantize`, `--skip-baseline`) |
| 9 | docs (PRD, architecture, nvfp4_workflow, b300_setup, troubleshooting, published baselines for MiMo-V2.5) |

**Test status:** 210 unit tests + 1 GPU integration smoke (skipped without `--gpu` / `LMBENCH_GPU=1`). 90% coverage. ruff and mypy strict clean. Verified on Linux WSL2 with `uv 0.7.3` + Python 3.11.11.

## Pipeline flow (per model)

```
+-- env.json (env capture: nvidia-smi, git SHA, package versions)
|
+-- baseline/    (skipped if --skip-baseline)
|     +-- perf/<workload>_c<concurrency>.json
|     +-- quality/quality_summary.json + raw lm-eval results_*.json
|
+-- quantized/   (only if quant_recipe set, --skip-quantize off)
|     +-- perf/<workload>_c<concurrency>.json
|     +-- quality/quality_summary.json
|
+-- comparison/
      +-- report.md
      +-- report.html
```

Sequence: capture env -> serve baseline -> bench -> quantize -> verify -> serve candidate -> bench -> compare -> report. Async only inside the perf driver; orchestration is purely sequential per-model.

## Hardware: 2x B300

User has **2x NVIDIA B300 Blackwell Ultra**, 288 GB HBM each, 576 GB total. This shapes which models fit:

| Model bf16 weight footprint | Fits TP=2 on 2x B300? |
|---|---|
| up to ~250B (~500 GB) | yes (with KV cache + activation headroom) |
| 310B (e.g. MiMo-V2.5, ~620 GB) | **no** -- requires `--skip-baseline` |

NVFP4 cuts weights by ~4x. Even MiMo-V2.5 NVFP4 (~155 GB) fits TP=2 comfortably. FP8 KV cache adds extra long-context / concurrency headroom. **Important:** FP8 KV cache reduces KV memory only; it does NOT shrink model weights, so it cannot make a too-large bf16 baseline fit.

## Target model: XiaomiMiMo/MiMo-V2.5

- 310B / 15B-active sparse MoE
- Hybrid attention: 5:1 SWA:GA with 128 sliding window + learnable attention-sink bias (claims ~6x KV-cache reduction)
- Native multimodal: 729M ViT + audio encoder
- 1M context (262K reported, 1M positioned)
- Custom modeling code -> needs `trust_remote_code: true`
- MIT license

**NVFP4 viability is uncertain.** Risks:
- Custom hybrid SWA+GA block with attention sinks -- modelopt's `NVFP4_DEFAULT_CFG` pattern-matches known module shapes; bespoke blocks may need a custom `quant_cfg`.
- Multimodal ViT + audio encoders -- `NVFP4_DEFAULT_CFG` is text-LLM oriented; calibration in this harness is text-only via cnn_dailymail; encoders likely stay bf16.
- vLLM must support both bf16 and `modelopt_fp4` forms; custom remote_code may not load cleanly.
- NVIDIA has shipped `nvidia/*-NVFP4` for Llama 3.x/4, DeepSeek-R1/V3, Qwen3.5 (incl. MoE), Gemma 4, Phi-4-multimodal -- **not MiMo**.

**Derisking strategy** (encoded in `configs/run_baseline_vs_nvfp4.yaml`):
1. Run **MiMo-V2-Flash** (smaller text trunk, same architectural family) through the full baseline-vs-NVFP4 cycle. If this NVFP4's cleanly, the MiMo block family is likely supported.
2. Then run **MiMo-V2.5** via `--skip-baseline` (the only way it fits 2x B300). Compare offline against `docs/published_baselines/mimo_v2_5.md`.

## How to run on 2x B300

**Full V2-Flash baseline-vs-NVFP4 (works locally on 2x B300):**

```bash
uv sync --extra all
uv run lmbench run --plan configs/run_baseline_vs_nvfp4.yaml
```

V2-Flash gets the full report (`results/baseline-vs-nvfp4/mimo-v2-flash/comparison/report.md` and `.html`).

**V2.5 NVFP4-only on 2x B300 (skip baseline):**

```bash
uv run lmbench run --plan configs/run_baseline_vs_nvfp4.yaml --skip-baseline
```

V2.5's comparison report will be empty (no baseline to diff against). Compare the produced `quality_summary.json` against the published numbers in `docs/published_baselines/mimo_v2_5.md`.

**For full V2.5 baseline-vs-NVFP4 with rigorous A/B:** rent a 4+ B300 / 8x H200 host once for the bf16 baseline run, save the JSONs, then return to 2x B300 for NVFP4. Run `lmbench compare --baseline <dir> --candidate <dir>` offline.

## Published baselines for MiMo-V2.5

Pulled from the official Hugging Face model card and `mimo.xiaomi.com` landing page; full table at `docs/published_baselines/mimo_v2_5.md`:

| Benchmark | MiMo-V2.5 score | Source |
|---|---:|---|
| SWE-bench Pro | 56.1 | huggingface.co/XiaomiMiMo/MiMo-V2.5 |
| Terminal-Bench 2 | 65.8 | huggingface.co/XiaomiMiMo/MiMo-V2.5 |
| Claw-Eval (general) | 62.3 | mimo.xiaomi.com/mimo-v2-5 |

Most other benchmark numbers on the model card are published as comparison **charts** (images), not text -- so they could not be machine-extracted. Re-fetch before citing externally.

Sibling-model bracketing (V2.5-Pro and V2-Flash scores) is in the same doc; use only as sanity bounds, not as the V2.5 baseline.

## Benchmark coverage

Three perf workload kinds (`WorkloadSpec.kind`):
- `random` -- synthetic prompts with controlled input/output length (default `random-short`).
- `longctx` -- long-context with token-budget guard (default `random-long` at 16K).
- `sharegpt` -- real human turns from a ShareGPT JSON file.

Quality tasks (run via `lm-eval --model local-completions` against the live vLLM server):
- Standard: `mmlu`, `gsm8k`, `arc_challenge`, `hellaswag`, `truthfulqa_mc2`. Optional: `humaneval` (gated, executes generated code).
- Long-context + coding (in `EvalSuite.long_context`, merged into `--tasks` via `bench.quality.merged_task_list`): `ruler`, `longbench`, `livecodebench`.

## Out of scope (deferred -- don't fit lm-eval `local-completions`)

These need separate harness modules; flagged in `CHANGELOG.md`:

- **SWE-bench Verified / Pro** -- needs Docker patch-evaluation harness.
- **Terminal-Bench 2** -- needs tmux/shell session driver.
- **tau2-bench** -- needs simulated-user multi-turn dialogue.
- **Aider Polyglot** -- needs the Aider tool harness wired up.

For SWE-bench Pro and Terminal-Bench 2, run the official upstream harnesses against the same vLLM endpoint your NVFP4 run uses if you need direct comparison to MiMo's published numbers.

## Commit history (local; push blocked by missing credentials in WSL2)

```
273194f feat(runner): --skip-baseline mode + V2.5 NVFP4-only entry + published baselines doc
7eac0ae fix(configs): right-size run_baseline_vs_nvfp4.yaml for a 2x B300 host
ad96a56 feat(configs): run_baseline_vs_nvfp4.yaml targeting XiaomiMiMo/MiMo-V2.5
28623b3 feat(bench): wire EvalSuite.long_context into lm-eval; enable RULER / LongBench / LiveCodeBench
3b84ee1 feat: lmbench harness phases 4-9 (perf, quality, NVFP4, compare/report, runner, docs)
e094c49 feat: lmbench harness scaffolding (phases 1-3)
```

`git push origin main` requires the user to run it manually (no cached credentials in this WSL2 environment). Or `! gh auth login` then `! git push origin main` from inside the Claude Code session.

## Known limitations

- **Orphan EngineCore subprocesses on shutdown.** vLLM internally spawns its own session, so `os.killpg` on the spawn pgid doesn't always reap them. Phase 8 runner should be extended with a `psutil`-based child-walk sweep.
- **`nvfp4_llmcompressor` recipe** is reserved in the schema but not wired (placeholder for fallback if modelopt rejects MiMo's block).
- **No `--baseline-only` mode.** If you need just the bf16 numbers without quantize, use `--skip-quantize`.
- **First real B300 run still needs to land.** Once it does, pin the `(vllm, nvidia-modelopt, transformers, driver)` quartet in `pyproject.toml` and `docs/b300_setup.md`.

## Key files

| Path | Purpose |
|---|---|
| `configs/run_smoke.yaml` | CPU-loadable opt-125m smoke; no NVFP4 |
| `configs/run_baseline_vs_nvfp4.yaml` | MiMo V2-Flash + V2.5 plan, 2x B300 |
| `configs/benchmarks.yaml` | Default workloads + eval suite (incl. long_context) |
| `docs/HANDOFF.md` | Phase-by-phase status (canonical progress source) |
| `docs/published_baselines/mimo_v2_5.md` | Confirmed published scores for MiMo-V2.5 |
| `docs/architecture.md` | Module map, pipeline diagram, lifecycle invariants |
| `docs/nvfp4_workflow.md` | Quantize step internals + tuning checklist |
| `docs/b300_setup.md` | Driver / uv / HF token / smoke recipe |
| `docs/troubleshooting.md` | Known failure modes |
| `src/lmbench/runner/pipeline.py` | The orchestrator (`run_plan`, `_run_one_model`) |
| `src/lmbench/cli.py` | Typer CLI; `lmbench run --plan ...` is the main entrypoint |
