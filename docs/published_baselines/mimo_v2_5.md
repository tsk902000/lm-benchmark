# MiMo-V2.5 - published baseline scores

When running `lmbench run --plan configs/run_baseline_vs_nvfp4.yaml --skip-baseline`
on a 2x B300 host, the comparison report for MiMo-V2.5 will be empty
(there is no on-host bf16 baseline to diff against - 620 GB does not fit
576 GB HBM). Use the numbers below as the published reference and
compare your NVFP4 results against them by hand.

These were extracted from the MiMo-V2.5 model card and the official
mimo.xiaomi.com landing page.

> Caveat: the model card mostly publishes scores as comparison charts
> (images), not as text tables. The numbers below are the explicit
> textual scores that appeared on the model card / landing page at the
> time the harness was authored. Re-fetch before using as a citation
> in any external report.

## Confirmed scores

| Benchmark             | Score | Source                       |
|-----------------------|------:|------------------------------|
| SWE-bench Pro         |  56.1 | model card (huggingface)     |
| Terminal-Bench 2      |  65.8 | model card (huggingface)     |
| Claw-Eval (general)   |  62.3 | mimo.xiaomi.com landing page |

## Qualitative claims (from the model card / landing page)

- Video understanding: "matching Gemini 3 Pro"
- Multimodal agentic work: "matching Claude Sonnet 4.6"
- Image and document understanding: "staying competitive"
- Long context: chart only, no published number; uses Graphwalks-style
  evaluation per the model card

## Sibling-model scores (for sanity bracketing only)

These are NOT the V2.5 baseline. Use them only to sanity-check your
NVFP4-V2.5 numbers (V2.5 should be at least as strong as V2-Flash on the
same task, and within striking distance of V2.5-Pro).

### MiMo-V2.5-Pro

| Benchmark | Score | Notes                |
|-----------|------:|----------------------|
| GSM8K     |  66.7 | per llm-stats summary |
| HLE       |  99.6 | per llm-stats summary; verify before citing - HLE rarely reaches 99 |
| MMLU-Pro  |  48.0 | per llm-stats summary |

### MiMo-V2-Flash

| Benchmark   | Score | Notes               |
|-------------|------:|---------------------|
| MMLU-Pro    |  73.2 | per llm-stats summary |
| GSM8K (8-shot) |  92.3 | per llm-stats summary |
| HumanEval+ (1-shot) | 70.7 | per llm-stats summary |

## How to compare

After your NVFP4 run completes, the candidate side of the report
includes parsed `quality_summary.json` files at
`results/baseline-vs-nvfp4/mimo-v2.5/quantized/quality/quality_summary.json`.

The relevant tasks for direct comparison against MiMo's published
baseline are the lm-eval entries that overlap:
- `mmlu` (compare against MMLU-Pro qualitatively if direct MMLU is missing)
- `gsm8k`
- `livecodebench` (newest; published HumanEval+ is the loose proxy)

For SWE-bench Pro and Terminal-Bench 2 we don't currently run them in
this harness (they need separate Docker / shell drivers - see the
"Out of scope" note in `CHANGELOG.md`). If you need a direct match for
those two benchmarks, run them with their official harnesses against
the same vLLM endpoint your NVFP4 run uses.

## Sources

- https://huggingface.co/XiaomiMiMo/MiMo-V2.5
- https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
- https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash
- https://mimo.xiaomi.com/mimo-v2-5
- https://llm-stats.com/benchmarks
