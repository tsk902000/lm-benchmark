"""Unit tests for `lmbench.utils.gpu`."""

from __future__ import annotations

import time
from typing import Any

import pytest

from lmbench.utils import gpu as gpu_mod
from lmbench.utils.gpu import GPUSampler, sample_gpu


class _FakeMem:
    def __init__(self, used_mb: float) -> None:
        self.used = int(used_mb * 1024 * 1024)


class _FakeUtil:
    def __init__(self, gpu_pct: float) -> None:
        self.gpu = int(gpu_pct)


class _FakePynvml:
    def __init__(self, count: int = 2) -> None:
        self.count = count
        self.tick = 0

    def nvmlInit(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return self.count

    def nvmlDeviceGetHandleByIndex(self, idx: int) -> Any:
        return ("handle", idx)

    def nvmlDeviceGetMemoryInfo(self, handle: Any) -> _FakeMem:
        idx = handle[1]
        # Memory grows linearly with tick to exercise peak vs steady.
        self.tick += 1
        return _FakeMem(100 + self.tick * 10 + idx * 1000)

    def nvmlDeviceGetUtilizationRates(self, handle: Any) -> _FakeUtil:
        return _FakeUtil(50 + handle[1] * 10)

    def nvmlDeviceGetPowerUsage(self, handle: Any) -> int:
        return 200_000 + handle[1] * 50_000


def test_sampler_no_pynvml_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_mod, "_try_import_pynvml", lambda: None)
    sampler = GPUSampler(interval_s=0.05)
    assert sampler.available is False
    sampler.start()
    sampler.stop()
    assert sampler.summarize() == {}


def test_sampler_collects_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePynvml(count=2)
    monkeypatch.setattr(gpu_mod, "_try_import_pynvml", lambda: fake)
    sampler = GPUSampler(interval_s=0.01)
    assert sampler.available is True
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    summary = sampler.summarize()
    assert set(summary) == {0, 1}
    assert summary[0].memory_peak_mb >= summary[0].memory_steady_mb
    assert summary[0].sm_util_peak_pct >= summary[0].sm_util_mean_pct
    assert summary[0].power_peak_w >= summary[0].power_mean_w
    assert summary[0].samples


def test_sampler_validates_interval() -> None:
    with pytest.raises(ValueError, match="interval_s"):
        GPUSampler(interval_s=0)


def test_sampler_double_start_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_mod, "_try_import_pynvml", lambda: _FakePynvml())
    sampler = GPUSampler(interval_s=0.05)
    sampler.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            sampler.start()
    finally:
        sampler.stop()


def test_sample_gpu_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePynvml(count=1)
    monkeypatch.setattr(gpu_mod, "_try_import_pynvml", lambda: fake)
    with sample_gpu(interval_s=0.01) as sampler:
        time.sleep(0.05)
    summary = sampler.summarize()
    assert 0 in summary


def test_sampler_init_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenInit:
        def nvmlInit(self) -> None:
            raise RuntimeError("driver missing")

    monkeypatch.setattr(gpu_mod, "_try_import_pynvml", lambda: _BrokenInit())
    sampler = GPUSampler(interval_s=0.05)
    assert sampler.available is False


def test_sampler_validates_device_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePynvml(count=2)
    monkeypatch.setattr(gpu_mod, "_try_import_pynvml", lambda: fake)
    with pytest.raises(ValueError, match="out of range"):
        GPUSampler(interval_s=0.05, device_indices=(5,))
