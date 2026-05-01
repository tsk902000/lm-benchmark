"""Unit tests for `lmbench.serve.vllm_server.build_serve_args`."""

from __future__ import annotations

from typing import Any

from lmbench.config import ModelEntry, VLLMArgs
from lmbench.serve import build_serve_args, served_model_name


def _entry(**vllm_overrides: Any) -> ModelEntry:
    return ModelEntry(
        name="opt",
        hf_id="facebook/opt-125m",
        vllm=VLLMArgs(**vllm_overrides),
    )


def test_build_serve_args_minimal() -> None:
    argv = build_serve_args(_entry())
    assert argv[:3] == ["vllm", "serve", "facebook/opt-125m"]
    assert argv[argv.index("--served-model-name") + 1] == "opt"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert argv[argv.index("--dtype") + 1] == "auto"


def test_build_serve_args_max_model_len_set() -> None:
    argv = build_serve_args(_entry(max_model_len=4096))
    assert argv[argv.index("--max-model-len") + 1] == "4096"


def test_build_serve_args_max_model_len_unset() -> None:
    argv = build_serve_args(_entry())
    assert "--max-model-len" not in argv


def test_build_serve_args_kv_cache_dtype_omitted_when_auto() -> None:
    argv = build_serve_args(_entry())
    assert "--kv-cache-dtype" not in argv


def test_build_serve_args_kv_cache_dtype_set() -> None:
    argv = build_serve_args(_entry(kv_cache_dtype="fp8"))
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"


def test_build_serve_args_quantization_set() -> None:
    argv = build_serve_args(_entry(quantization="modelopt_fp4"))
    assert argv[argv.index("--quantization") + 1] == "modelopt_fp4"


def test_build_serve_args_enforce_eager_flag() -> None:
    argv_no = build_serve_args(_entry())
    argv_yes = build_serve_args(_entry(enforce_eager=True))
    assert "--enforce-eager" not in argv_no
    assert "--enforce-eager" in argv_yes


def test_build_serve_args_no_enable_prefix_caching_when_disabled() -> None:
    argv = build_serve_args(_entry(enable_prefix_caching=False))
    assert "--no-enable-prefix-caching" in argv


def test_build_serve_args_extra_args_normalized() -> None:
    argv = build_serve_args(_entry(extra_args={"swap-space": "4", "--seed": "42"}))
    assert argv[argv.index("--swap-space") + 1] == "4"
    assert argv[argv.index("--seed") + 1] == "42"


def test_build_serve_args_revision_passed_through() -> None:
    entry = ModelEntry(name="opt", hf_id="facebook/opt-125m", revision="main")
    argv = build_serve_args(entry)
    assert argv[argv.index("--revision") + 1] == "main"


def test_build_serve_args_pipeline_parallel() -> None:
    argv_default = build_serve_args(_entry())
    assert "--pipeline-parallel-size" not in argv_default
    argv_set = build_serve_args(_entry(pipeline_parallel_size=2))
    assert argv_set[argv_set.index("--pipeline-parallel-size") + 1] == "2"


def test_build_serve_args_trust_remote_code() -> None:
    argv = build_serve_args(_entry(trust_remote_code=True))
    assert "--trust-remote-code" in argv


def test_served_model_name_default_uses_entry_name() -> None:
    entry = ModelEntry(name="opt-shorthand", hf_id="facebook/opt-125m")
    assert served_model_name(entry) == "opt-shorthand"


def test_served_model_name_override() -> None:
    entry = ModelEntry(
        name="opt",
        hf_id="facebook/opt-125m",
        served_model_name="custom-id",
    )
    assert served_model_name(entry) == "custom-id"


def test_build_serve_args_custom_executable_and_port() -> None:
    argv = build_serve_args(_entry(), host="0.0.0.0", port=9000, executable="/opt/vllm")
    assert argv[0] == "/opt/vllm"
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "9000"
