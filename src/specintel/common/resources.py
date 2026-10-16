"""Timing / memory / hardware utilities shared by all workstreams.

Pure standard library so the module works on a laptop, a GPU box and inside a
container without extra dependencies.  ``gpu_memory_mb`` returns ``None`` when
no CUDA/NVIDIA tooling is present, which is what the logs should say.
"""

from __future__ import annotations

import functools
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def rss_mb() -> float:
    """Resident set size of this process in MiB (0.0 when unavailable)."""
    try:  # Linux
        with open("/proc/self/statm", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * (os_pagesize() / (1024 * 1024))
    except Exception:  # pragma: no cover - non-Linux fallback
        try:
            import resource  # noqa: PLC0415

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0


def os_pagesize() -> int:
    try:
        return __import__("os").sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return 4096


def gpu_memory_mb() -> float | None:
    """Peak GPU memory in MiB via ``nvidia-smi``; ``None`` when no GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        first = out.stdout.strip().splitlines()[0]
        return float(first)
    except Exception:
        return None


def hardware_profile() -> dict[str, Any]:
    """Describe the machine for logs and the evaluation report."""
    profile: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(),
        "ram_gb": None,
        "gpu": None,
        "gpu_memory_mb": gpu_memory_mb(),
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    profile["ram_gb"] = round(int(line.split()[1]) / (1024 * 1024), 1)
                    break
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0 and out.stdout.strip():
            profile["gpu"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return profile


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    """Accumulates count / duration / memory observations for one stage."""

    name: str
    count: int = 0
    duration_s: float = 0.0
    peak_rss_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)
    samples: list[float] = field(default_factory=list)

    def observe(self, duration_s: float, n: int = 1) -> None:
        self.count += n
        self.duration_s += duration_s
        self.samples.append(duration_s)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb())
        gpu = gpu_memory_mb()
        if gpu:
            self.peak_gpu_mb = max(self.peak_gpu_mb, gpu)

    # -- derived ----------------------------------------------------------- #
    @property
    def mean_s(self) -> float:
        return self.duration_s / self.count if self.count else 0.0

    def percentile_s(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "count": self.count,
            "duration_s": round(self.duration_s, 4),
            "mean_s": round(self.mean_s, 4),
            "p50_s": round(self.percentile_s(0.5), 4),
            "p95_s": round(self.percentile_s(0.95), 4),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_gpu_mb": round(self.peak_gpu_mb, 1),
            "notes": self.notes,
        }


@contextmanager
def stopwatch(stats: Stats | None = None, n: int = 1) -> Iterator[Callable[[], float]]:
    """Time a block; optionally feed the observation into ``stats``."""
    start = time.perf_counter()
    box: dict[str, float] = {}

    def elapsed() -> float:
        box["t"] = time.perf_counter() - start
        return box["t"]

    try:
        yield elapsed
    finally:
        dt = elapsed()
        if stats is not None:
            stats.observe(dt, n=n)


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator returning ``(result, seconds)`` is *not* applied here; the
    decorated function gains a ``.stats`` counter instead."""
    stats = Stats(name=func.__qualname__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            stats.observe(time.perf_counter() - start)

    wrapper.stats = stats  # type: ignore[attr-defined]
    return wrapper
