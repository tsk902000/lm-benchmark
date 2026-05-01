"""GPU telemetry sampler.

Wraps `pynvml` in a background thread that polls each visible device on
a fixed interval. The lazy import lets the harness run on hosts without
NVIDIA drivers (where `pynvml.nvmlInit` would otherwise raise).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class DeviceSample:
    """One snapshot of a single GPU."""

    timestamp_s: float
    """Seconds since sampler start."""

    device_index: int
    memory_used_mb: float
    sm_utilization_pct: float
    power_w: float


@dataclass
class DeviceSummary:
    """Aggregate stats for a single device over the sampling window."""

    device_index: int
    memory_peak_mb: float = 0.0
    memory_steady_mb: float = 0.0
    sm_util_mean_pct: float = 0.0
    sm_util_peak_pct: float = 0.0
    power_mean_w: float = 0.0
    power_peak_w: float = 0.0
    samples: list[DeviceSample] = field(default_factory=list)


def _try_import_pynvml() -> ModuleType | None:
    try:
        import pynvml as _pynvml
    except ImportError:
        return None
    return _pynvml  # type: ignore[no-any-return]


class GPUSampler:
    """Background-thread GPU sampler.

    Constructed lazily — if `pynvml` is missing or `nvmlInit` fails the
    sampler enters a degraded "no-op" mode where `start`/`stop` succeed
    but no samples are produced. This keeps the test suite green on
    GPU-less hosts and avoids hard failures in mixed environments.
    """

    def __init__(
        self,
        *,
        interval_s: float = 0.25,
        device_indices: tuple[int, ...] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        self.interval_s = interval_s
        self._pynvml = _try_import_pynvml()
        self._handles: list[Any] = []
        self._device_indices: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[DeviceSample] = []
        self._start_time: float = 0.0

        if self._pynvml is None:
            return
        try:
            self._pynvml.nvmlInit()
        except Exception:
            self._pynvml = None
            return
        count = self._pynvml.nvmlDeviceGetCount()
        if device_indices is None:
            indices = tuple(range(count))
        else:
            for idx in device_indices:
                if idx < 0 or idx >= count:
                    raise ValueError(f"device index {idx} out of range [0, {count})")
            indices = device_indices
        self._device_indices = list(indices)
        self._handles = [
            self._pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices
        ]

    @property
    def available(self) -> bool:
        """True if pynvml + at least one device are usable."""
        return bool(self._handles)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPUSampler already started")
        self._stop.clear()
        self._samples = []
        self._start_time = time.monotonic()
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def _sample_once(self) -> None:
        if self._pynvml is None:
            return
        ts = time.monotonic() - self._start_time
        for idx, h in zip(self._device_indices, self._handles, strict=True):
            try:
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(h)
                power_mw = self._pynvml.nvmlDeviceGetPowerUsage(h)
            except Exception:
                continue
            self._samples.append(
                DeviceSample(
                    timestamp_s=ts,
                    device_index=idx,
                    memory_used_mb=mem.used / (1024 * 1024),
                    sm_utilization_pct=float(util.gpu),
                    power_w=power_mw / 1000.0,
                )
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def summarize(self) -> dict[int, DeviceSummary]:
        """Reduce raw samples into per-device peak/mean stats.

        "Steady" memory is the median across samples in the second half
        of the window — the warmup-tolerant baseline used in HANDOFF.md.
        """
        out: dict[int, DeviceSummary] = {
            idx: DeviceSummary(idx) for idx in self._device_indices
        }
        for s in self._samples:
            out.setdefault(s.device_index, DeviceSummary(s.device_index))
            out[s.device_index].samples.append(s)
        for idx, summary in out.items():
            samples = summary.samples
            if not samples:
                continue
            mems = [s.memory_used_mb for s in samples]
            utils = [s.sm_utilization_pct for s in samples]
            powers = [s.power_w for s in samples]
            half = len(mems) // 2
            steady_window = mems[half:] if half > 0 else mems
            sorted_steady = sorted(steady_window)
            mid = len(sorted_steady) // 2
            steady_mb = (
                sorted_steady[mid]
                if len(sorted_steady) % 2 == 1
                else (sorted_steady[mid - 1] + sorted_steady[mid]) / 2
            )
            out[idx] = DeviceSummary(
                device_index=idx,
                memory_peak_mb=max(mems),
                memory_steady_mb=steady_mb,
                sm_util_mean_pct=sum(utils) / len(utils),
                sm_util_peak_pct=max(utils),
                power_mean_w=sum(powers) / len(powers),
                power_peak_w=max(powers),
                samples=samples,
            )
        return out


@contextmanager
def sample_gpu(
    *,
    interval_s: float = 0.25,
    device_indices: tuple[int, ...] | None = None,
) -> Iterator[GPUSampler]:
    """Context manager that runs a `GPUSampler` for the duration of the body."""
    sampler = GPUSampler(interval_s=interval_s, device_indices=device_indices)
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
