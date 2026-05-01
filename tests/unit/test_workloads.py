"""Unit tests for `lmbench.bench.workloads`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmbench.bench import gen_longctx, gen_random, gen_sharegpt, generate
from lmbench.config import WorkloadSpec


def _spec(**overrides: object) -> WorkloadSpec:
    base: dict[str, object] = {
        "name": "w",
        "kind": "random",
        "num_prompts": 5,
        "input_len": 32,
        "output_len": 16,
        "seed": 0,
        "warmup_prompts": 0,
        "concurrency": (1,),
    }
    base.update(overrides)
    return WorkloadSpec.model_validate(base)


def test_gen_random_count_and_lengths_meta() -> None:
    spec = _spec(num_prompts=7, output_len=64)
    prompts = gen_random(spec)
    assert len(prompts) == 7
    for p in prompts:
        assert p.expected_output_tokens == 64
        assert p.text


def test_gen_random_deterministic_with_seed() -> None:
    a = gen_random(_spec(seed=42))
    b = gen_random(_spec(seed=42))
    assert a == b


def test_gen_random_diverges_with_different_seed() -> None:
    a = gen_random(_spec(seed=1))
    b = gen_random(_spec(seed=2))
    assert a != b


def test_gen_random_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="random"):
        gen_random(_spec(kind="longctx"))


def test_gen_longctx_basic() -> None:
    spec = _spec(kind="longctx", input_len=2048, output_len=512)
    prompts = gen_longctx(spec)
    assert len(prompts) == 5
    assert prompts[0].expected_output_tokens == 512


def test_gen_longctx_token_budget_guard() -> None:
    spec = _spec(kind="longctx", input_len=4096, output_len=4096)
    with pytest.raises(ValueError, match="exceeds max_total_tokens"):
        gen_longctx(spec, max_total_tokens=4096)


def test_gen_longctx_token_budget_pass() -> None:
    spec = _spec(kind="longctx", input_len=2048, output_len=1024)
    prompts = gen_longctx(spec, max_total_tokens=4096)
    assert len(prompts) == 5


def test_gen_sharegpt_filters_and_samples(tmp_path: Path) -> None:
    payload = [
        {
            "conversations": [
                {"from": "human", "value": "Tell me about RAG."},
                {"from": "gpt", "value": "Sure ..."},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Another prompt."},
            ]
        },
        {"conversations": [{"from": "system", "value": "ignored"}]},
        {"not": "valid"},
    ]
    p = tmp_path / "sharegpt.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    spec = _spec(kind="sharegpt", num_prompts=10)
    prompts = gen_sharegpt(spec, dataset_path=p)
    assert len(prompts) == 10
    texts = {pr.text for pr in prompts}
    assert texts <= {"Tell me about RAG.", "Another prompt."}


def test_gen_sharegpt_rejects_non_list(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"oops": "not-a-list"}), encoding="utf-8")
    spec = _spec(kind="sharegpt")
    with pytest.raises(ValueError, match="JSON list"):
        gen_sharegpt(spec, dataset_path=p)


def test_gen_sharegpt_rejects_no_usable(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")
    spec = _spec(kind="sharegpt")
    with pytest.raises(ValueError, match="no usable conversations"):
        gen_sharegpt(spec, dataset_path=p)


def test_generate_dispatches() -> None:
    assert len(generate(_spec(kind="random"))) == 5
    assert len(generate(_spec(kind="longctx", input_len=128, output_len=32))) == 5


def test_generate_sharegpt_requires_path() -> None:
    spec = _spec(kind="sharegpt")
    with pytest.raises(ValueError, match="sharegpt_path"):
        generate(spec)
