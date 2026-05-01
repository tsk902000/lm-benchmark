"""Perf benchmark driver.

Streams OpenAI-compatible `/v1/completions` requests at a configurable
concurrency, capturing per-token timing. Returns `(samples, summary)`
ready for the compare/report layers.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass

import httpx

from lmbench.config import WorkloadSpec
from lmbench.utils.gpu import DeviceSummary, GPUSampler

from .metrics import PerfSummary, RequestSample, summarize
from .workloads import Prompt


@dataclass(frozen=True)
class PerfResult:
    """Bundle of samples + summary + GPU telemetry for one perf run."""

    samples: tuple[RequestSample, ...]
    summary: PerfSummary
    gpu_summary: dict[int, DeviceSummary]
    concurrency: int
    workload_name: str


async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: Prompt,
) -> RequestSample:
    """Issue one streaming completion request; return a RequestSample.

    The OpenAI streaming protocol emits `data: {...}\\n\\n` chunks
    terminated by `data: [DONE]\\n\\n`. We measure TTFT (time to first
    chunk with non-empty `text`) and ITL (gap between successive token
    deliveries).
    """
    body = {
        "model": model,
        "prompt": prompt.text,
        "max_tokens": prompt.expected_output_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    url = f"{base_url}/v1/completions"

    t_start = time.monotonic()
    last_t = t_start
    ttft: float | None = None
    itls: list[float] = []
    output_tokens = 0
    success = True

    try:
        async with client.stream("POST", url, json=body, timeout=300.0) as resp:
            if resp.status_code != 200:
                return RequestSample(
                    ttft_s=0.0,
                    itl_s=(),
                    e2e_s=time.monotonic() - t_start,
                    output_tokens=0,
                    success=False,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                first = choices[0]
                text = first.get("text") if isinstance(first, dict) else None
                if not isinstance(text, str) or not text:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t_start
                else:
                    itls.append(now - last_t)
                last_t = now
                output_tokens += 1
    except (httpx.RequestError, httpx.HTTPError):
        success = False

    e2e = time.monotonic() - t_start
    return RequestSample(
        ttft_s=ttft if ttft is not None else 0.0,
        itl_s=tuple(itls),
        e2e_s=e2e,
        output_tokens=output_tokens,
        success=success and ttft is not None,
    )


async def _run_async(
    base_url: str,
    model: str,
    prompts: Iterable[Prompt],
    concurrency: int,
) -> tuple[tuple[RequestSample, ...], float]:
    """Run all prompts at fixed concurrency; return (samples, duration_s)."""
    semaphore = asyncio.Semaphore(concurrency)
    prompt_list = list(prompts)

    async def _bounded(client: httpx.AsyncClient, p: Prompt) -> RequestSample:
        async with semaphore:
            return await _stream_one(client, base_url, model, p)

    t_start = time.monotonic()
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_bounded(client, p)) for p in prompt_list]
        samples = await asyncio.gather(*tasks)
    duration = time.monotonic() - t_start
    return tuple(samples), duration


def run_workload(
    *,
    base_url: str,
    served_model_name: str,
    workload: WorkloadSpec,
    concurrency: int,
    prompts: Iterable[Prompt],
    gpu_sampler: GPUSampler | None = None,
) -> PerfResult:
    """Run one workload at one concurrency level.

    Splits the prompt sequence into a warmup phase (counted but excluded
    from the main metrics) and a measured phase. The GPU sampler, if
    provided, is started before the measured phase and stopped after.
    """
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("run_workload requires a non-empty prompt sequence")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    warmup_n = min(workload.warmup_prompts, len(prompt_list))
    warmup_prompts = prompt_list[:warmup_n]
    measured_prompts = prompt_list[warmup_n:]
    if not measured_prompts:
        raise ValueError(
            f"workload has {len(prompt_list)} prompts but warmup_prompts={warmup_n} "
            "consumes them all; raise num_prompts or lower warmup_prompts"
        )

    if warmup_prompts:
        asyncio.run(
            _run_async(base_url, served_model_name, warmup_prompts, concurrency)
        )

    if gpu_sampler is not None:
        gpu_sampler.start()
    try:
        samples, duration = asyncio.run(
            _run_async(base_url, served_model_name, measured_prompts, concurrency)
        )
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()

    summary = summarize(list(samples), duration)
    gpu_summary = gpu_sampler.summarize() if gpu_sampler is not None else {}
    return PerfResult(
        samples=samples,
        summary=summary,
        gpu_summary=gpu_summary,
        concurrency=concurrency,
        workload_name=workload.name,
    )
