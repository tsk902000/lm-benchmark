"""Unit tests for `lmbench.config.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmbench.config import (
    load_eval_suite,
    load_hardware,
    load_models,
    load_quant_recipe,
    load_run_plan,
    load_workloads,
    load_yaml,
)


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


def test_load_yaml_returns_empty_dict_for_empty_file(tmp_path: Path) -> None:
    p = _write(tmp_path / "empty.yaml", "")
    assert load_yaml(p) == {}


def test_load_yaml_env_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMBENCH_TEST_VAL", "hello")
    p = _write(tmp_path / "x.yaml", "greet: ${LMBENCH_TEST_VAL}\n")
    assert load_yaml(p) == {"greet": "hello"}


def test_load_yaml_env_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMBENCH_TEST_MISSING", raising=False)
    p = _write(tmp_path / "x.yaml", "greet: ${LMBENCH_TEST_MISSING:-fallback}\n")
    assert load_yaml(p) == {"greet": "fallback"}


def test_load_yaml_missing_var_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMBENCH_TEST_REQUIRED", raising=False)
    p = _write(tmp_path / "x.yaml", "greet: ${LMBENCH_TEST_REQUIRED}\n")
    with pytest.raises(KeyError, match="LMBENCH_TEST_REQUIRED"):
        load_yaml(p)


def test_load_yaml_substitutes_inside_lists_and_nested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LMBENCH_TOK", "abc")
    p = _write(
        tmp_path / "x.yaml",
        """
items:
  - name: a
    token: ${LMBENCH_TOK}
  - name: b
    nested:
      key: prefix-${LMBENCH_TOK}-suffix
""",
    )
    data = load_yaml(p)
    assert data["items"][0]["token"] == "abc"
    assert data["items"][1]["nested"]["key"] == "prefix-abc-suffix"


def test_load_models_seed_yaml() -> None:
    models = load_models(Path("configs/models.yaml"))
    names = {m.name for m in models}
    assert names == {"opt-125m-smoke", "mimo-v2.5"}
    mimo = next(m for m in models if m.name == "mimo-v2.5")
    assert mimo.hf_id == "XiaomiMiMo/MiMo-V2.5"
    assert mimo.vllm.trust_remote_code is True


def test_load_workloads_seed_yaml() -> None:
    workloads = load_workloads(Path("configs/benchmarks.yaml"))
    names = {w.name for w in workloads}
    assert "random-short" in names


def test_load_eval_suite_seed_yaml() -> None:
    suite = load_eval_suite(Path("configs/benchmarks.yaml"))
    assert "mmlu" in suite.tasks


def test_load_quant_recipe_seed_yaml() -> None:
    recipe = load_quant_recipe(Path("configs/quantization.yaml"))
    assert recipe.method == "nvfp4"
    assert recipe.calibration.num_samples == 512


def test_load_hardware_seed_yaml() -> None:
    hw = load_hardware(Path("configs/hardware.yaml"))
    assert hw.gpu == "B300"
    assert hw.blackwell is True


def test_load_run_plan_seed_yaml() -> None:
    plan = load_run_plan(Path("configs/run_smoke.yaml"))
    assert plan.name == "smoke"
    assert plan.models[0].hf_id == "facebook/opt-125m"


def test_load_run_plan_mimo_v2_5_fp8_vs_nvfp4() -> None:
    """The V2.5 plan attempts public FP8 baseline vs NVFP4 on 2x B300."""
    plan = load_run_plan(Path("configs/run_mimo_v2_5_nvfp4.yaml"))
    assert plan.name == "mimo-v2-5-fp8-vs-nvfp4"
    assert {m.name for m in plan.models} == {"mimo-v2.5"}
    assert plan.hardware.num_gpus == 2
    assert plan.hardware.blackwell is True
    assert plan.quant_recipe is not None
    assert plan.quant_recipe.method == "nvfp4"
    v25 = plan.models[0]
    assert v25.vllm.tensor_parallel_size == 2
    assert v25.vllm.kv_cache_dtype == "fp8"
    assert v25.vllm.trust_remote_code is True


def test_load_models_rejects_non_list(tmp_path: Path) -> None:
    p = _write(tmp_path / "models.yaml", "models:\n  not_a_list: 1\n")
    with pytest.raises(ValueError, match="expected a list of models"):
        load_models(p)


def test_load_workloads_rejects_non_list(tmp_path: Path) -> None:
    p = _write(tmp_path / "wl.yaml", "workloads: 5\n")
    with pytest.raises(ValueError, match="expected a list of workloads"):
        load_workloads(p)


def test_load_models_accepts_bare_list(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "models.yaml",
        "- name: m\n  hf_id: x/y\n",
    )
    models = load_models(p)
    assert models[0].name == "m"


def test_load_hardware_bare_mapping(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "hw.yaml",
        "name: hp\ngpu: H100\nnum_gpus: 1\ndefault_tp_size: 1\n",
    )
    hw = load_hardware(p)
    assert hw.gpu == "H100"
