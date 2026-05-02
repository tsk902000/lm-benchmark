# Troubleshooting

A grab bag of failure modes we've already hit and how to recover from them.

## Setup / install

### `uv sync` fails with "no Python found"

You are probably installing `uv` for the first time on this host and `~/.local/bin` is not yet on `PATH`. Reload the shell, or `export PATH="$HOME/.local/bin:$PATH"`.

### `uv sync --extra all` fails on Windows / macOS

The `[gpu]` and `[quant]` extras are Linux/CUDA only. Drop them: `uv sync` (no extras) is the dev-only path. The harness module imports cleanly without them; only Phase 3 onwards needs them at runtime.

### `git push` asks for credentials

This is a fresh repo with no cached credentials. Either run `gh auth login` (after `apt install gh`), or set up an SSH key and switch the remote to `git@github.com:...`. See README "Common commands".

## Test failures

### `pytest` hangs running the integration smoke

The integration smoke (`tests/integration/test_serve_smoke.py`) actually loads `facebook/opt-125m`. It only runs when `--gpu` is passed (or `LMBENCH_GPU=1`). Without that flag, it should auto-skip.

If it does not skip and you are not on a GPU, you probably modified `tests/conftest.py` and broke the gate. The gate registers `--gpu` and `--blackwell` and adds skip markers in `pytest_collection_modifyitems`.

### `mypy` complains about missing imports for `vllm`, `modelopt`, `torch`

Those are gated behind the `[gpu]` / `[quant]` extras. mypy is configured in `pyproject.toml [[tool.mypy.overrides]]` to `ignore_missing_imports` for them. If you added a new GPU-only dep, extend that list.

## Runtime

### `lmbench run` appears to hang with no output

MiMo-V2.5 can spend a long time downloading weights, loading remote code,
building CUDA kernels, or starting vLLM before the first benchmark artifact is
written. The CLI now prints progress messages for plan load, env capture,
server startup, perf cells, quality runs, quantization, verification, and report
writing. If it is still silent, check whether the process is alive with `nvidia-smi`
and inspect `results/<run>/env.json` or the vLLM process logs.

### `vllm serve` orphans EngineCore subprocesses on shutdown

Known limitation. vLLM internally spawns its own subprocesses that may end up in their own session, so `os.killpg` on the script's pgid does not always reap them. The teardown will report success but `ps -ef | grep vllm` may show leftover `VLLM::EngineCore` rows. A future runner hardening pass should use `psutil` to walk the child tree explicitly; for now, manual `kill -9` is the fallback.

### "no results_*.json under <output_dir>; lm-eval may have failed"

`lm-eval` exited 0 but produced no results - usually because the task list was wrong or the local-completions endpoint was unreachable. Check `<output_dir>/quality/<lm-eval logs>` if redirected, or re-run with the same args manually.

### "Regression detected" but the candidate is actually faster

Default thresholds in `compare/differ.py` are 5% for both latency and throughput. If the noise floor in your environment is higher, raise `latency_threshold_pct` / `throughput_threshold_pct` when calling `diff_perf` from a custom script. The runner currently does not expose these as CLI flags - that's a near-term polish item.

### NVFP4 run aborts with "verification failed"

By design. `verify_checkpoint` runs immediately after quantization and rejects degenerate output (empty completions or repeating-character output). Inspect `<output_dir>/quantized/.../lmbench_quant_meta.json` and the verifier's `completion` / `reason` to debug. Common causes: calibration sample size too low (try 1024), calibration max_seq_len shorter than the prompt, modelopt version mismatch with vllm's quantization loader.

## Performance / measurement

### TTFT spikes on the first concurrency level

That's warmup. The driver runs `warmup_prompts` (default 8) before the measured phase, but the very first prompt of the measured phase still pays for kv-cache allocation if the workload's input length differs from warmup. Increase warmup or use the same input_len throughout.

### GPU memory `peak` and `steady` are identical

You ran a workload too short for the sampler to capture more than one tick. The sampler defaults to 250ms; lower it via `GPUSampler(interval_s=0.05)` for short workloads.

## Reporting

### Plotly tables look broken in the HTML report

Plotly is loaded from CDN by default (`include_plotlyjs="cdn"`). On an air-gapped host either replace it with a vendored asset, or fall back to the plain-HTML variant by uninstalling the `plotly` package - the HTML report layer detects that and degrades gracefully.
