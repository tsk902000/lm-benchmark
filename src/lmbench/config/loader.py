"""YAML loaders with environment-variable interpolation.

Each loader returns a fully-validated pydantic model. Strings of the form
`${VAR}` or `${VAR:-default}` are substituted from the process environment
before validation. Missing variables without a default raise — silent
empty-string substitution masks misconfiguration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    EvalSuite,
    HardwareProfile,
    ModelEntry,
    QuantRecipe,
    RunPlan,
    WorkloadSpec,
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _substitute(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        var, default = m.group(1), m.group(2)
        if var in os.environ:
            return os.environ[var]
        if default is not None:
            return default
        raise KeyError(
            f"Environment variable {var!r} is not set and no default was provided "
            f"in placeholder {m.group(0)!r}"
        )

    return _ENV_PATTERN.sub(repl, s)


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        return _substitute(value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_yaml(path: Path) -> Any:
    """Read a YAML file from disk and apply env-var interpolation to all strings."""
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    return _interpolate(data)


def _section(data: Any, key: str) -> Any:
    """Return `data[key]` if present, else `data` (allows bare or wrapped docs)."""
    if isinstance(data, dict) and key in data:
        return data[key]
    return data


def load_models(path: Path) -> tuple[ModelEntry, ...]:
    """Load a model registry. Expects either `{models: [...]}` or a bare list."""
    data = load_yaml(path)
    items = _section(data, "models")
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of models")
    return tuple(ModelEntry.model_validate(m) for m in items)


def load_workloads(path: Path) -> tuple[WorkloadSpec, ...]:
    """Load workload specs. Expects either `{workloads: [...]}` or a bare list."""
    data = load_yaml(path)
    items = _section(data, "workloads")
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of workloads")
    return tuple(WorkloadSpec.model_validate(w) for w in items)


def load_eval_suite(path: Path) -> EvalSuite:
    """Load an eval suite. Expects either `{eval_suite: {...}}` or a bare mapping."""
    data = load_yaml(path)
    return EvalSuite.model_validate(_section(data, "eval_suite"))


def load_quant_recipe(path: Path) -> QuantRecipe:
    """Load a quant recipe. Expects either `{quant_recipe: {...}}` or a bare mapping."""
    data = load_yaml(path)
    return QuantRecipe.model_validate(_section(data, "quant_recipe"))


def load_hardware(path: Path) -> HardwareProfile:
    """Load a hardware profile. Expects either `{hardware: {...}}` or a bare mapping."""
    data = load_yaml(path)
    return HardwareProfile.model_validate(_section(data, "hardware"))


def load_run_plan(path: Path) -> RunPlan:
    """Load a complete run plan from a single YAML document."""
    return RunPlan.model_validate(load_yaml(path))
