# NVFP4 quantization workflow

`lmbench` quantizes via the `nvidia-modelopt` post-training (PTQ) flow:

```
HF baseline checkpoint
        |
        v
1. Load with transformers (bfloat16, device_map="auto")
        |
        v
2. Sample calibration text (cnn_dailymail x 512 by default)
        |
        v
3. Tokenize with the model's tokenizer (truncate to max_seq_len)
        |
        v
4. modelopt.torch.quantization.quantize(model, NVFP4_DEFAULT_CFG,
                                        forward_loop=cb)
        |
        v
5. model.save_pretrained(output_dir)
   tokenizer.save_pretrained(output_dir)
   write lmbench_quant_meta.json sidecar
        |
        v
6. verify_checkpoint(): boot it in vllm, send "The capital of France is",
   reject empty / degenerate completions
```

## Source files

- `src/lmbench/quantize/calibration.py` - cnn_dailymail / wikitext loaders + tokenization helpers.
- `src/lmbench/quantize/modelopt_nvfp4.py` - the `quantize_to_nvfp4(model_entry, recipe)` end-to-end wrapper.
- `src/lmbench/quantize/verify.py` - the post-quantization sanity probe.

## Pinned versions

vLLM and `nvidia-modelopt` move quickly; the export-import handshake breaks across some minor releases. Pin a tested pair in `pyproject.toml [project.optional-dependencies]` and document the known-good combo in `docs/b300_setup.md` once we have a green run on real B300 hardware.

## Calibration knobs

| Field             | Default            | Notes                                                                 |
|-------------------|--------------------|-----------------------------------------------------------------------|
| `dataset`         | `cnn_dailymail`    | Override to `wikitext` for math-heavy bias; pluggable.                |
| `dataset_config`  | `"3.0.0"`          | Required for cnn_dailymail.                                           |
| `split`           | `train`            | `validation` is fine for spot-checks; the harness does not memorize.  |
| `num_samples`     | 512                | More samples = more stable scaling factors. 256 is OK in a pinch.     |
| `max_seq_len`     | 2048               | Should match or undershoot `vllm.max_model_len`.                      |
| `seed`            | 0                  | Reproducible sampling.                                                |

## Failure modes already guarded

- **Quantized output is empty / repeating one character.** Caught by `verify_checkpoint`. Run aborts.
- **NVFP4 on non-Blackwell hardware.** Caught at config load - `RunPlan` validator requires `hardware.blackwell=True` when `quant_recipe.method=="nvfp4"`.
- **Calibration dataset returned 0 rows.** Caught by `sample_calibration_text` - raises before we touch the model.

## Tuning checklist when quality drops

1. Increase `num_samples` to 1024.
2. Try a different `dataset` (wikitext vs cnn_dailymail).
3. Bump `max_seq_len` so calibration covers the full context window the model is going to serve.
4. If still bad, fall back to `nvfp4_llmcompressor` (deferred recipe in the schema; not yet wired).
