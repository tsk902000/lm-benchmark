# MiMo-V2.5 - published reference notes

Run `lmbench run --plan configs/run_mimo_v2_5_nvfp4.yaml` on a 2x B300 host
to attempt a same-harness public-FP8 checkpoint vs NVFP4 comparison.

This is not BF16->NVFP4 loss: the public MiMo-V2.5 checkpoint is already FP8.
If the FP8 baseline stage cannot serve on your exact vLLM / driver stack,
rerun with `--skip-baseline` and use this file only as loose external context.

## What This Harness Covers

Directly covered through `lm-eval`:

- `mmlu`
- `gsm8k`
- `arc_challenge`
- `hellaswag`
- `truthfulqa_mc2`
- optionally `ruler`, `longbench`, `livecodebench` if supported by the
  installed `lm-eval` version

Not covered here:

- SWE-bench Pro
- Terminal-Bench 2
- tau-bench / tau2-style multi-turn agent tasks
- Aider Polyglot

Run those upstream harnesses against the same vLLM endpoint if you need direct
comparison to Xiaomi's published agentic/coding numbers.

## External References

At verification time, public pages indicated:

- MiMo-V2.5 is an FP8 custom-code checkpoint on Hugging Face.
- The vLLM recipe describes MiMo-V2.5 as 310B total / 15B active, 1,048,576
  context, native omnimodal, and native FP8.
- Xiaomi/third-party model pages report ClawEval-style public numbers, but many
  benchmark values are chart images or evaluation-result metadata rather than
  stable text tables.

Re-fetch before citing exact published benchmark numbers externally.

## Sources

- https://huggingface.co/XiaomiMiMo/MiMo-V2.5
- https://huggingface.co/XiaomiMiMo/MiMo-V2.5/blob/main/config.json
- https://huggingface.co/XiaomiMiMo/MiMo-V2.5/discussions/1
- https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5
- https://mimo.mi.com/
