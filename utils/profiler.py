"""
Shared GPU profiling utilities for all experiment phases.
Uses torch.cuda.Event for precise GPU timing.
"""

import torch
import time
import json
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from contextlib import contextmanager


@dataclass
class TimingResult:
    name: str
    gpu_ms: float
    cpu_ms: float = 0.0
    count: int = 1

    @property
    def gpu_avg_ms(self):
        return self.gpu_ms / self.count

    @property
    def cpu_avg_ms(self):
        return self.cpu_ms / self.count


class CUDAProfiler:
    """Profile GPU operations using CUDA events."""

    def __init__(self, num_warmup: int = 5, num_iterations: int = 20, sync: bool = True):
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.sync = sync
        self.results: dict[str, TimingResult] = {}
        self._active_timers: dict[str, tuple] = {}

    def start(self, name: str):
        """Start a named timer."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if self.sync:
            torch.cuda.synchronize()
        start_event.record()
        self._active_timers[name] = (start_event, end_event, time.perf_counter())

    def stop(self, name: str) -> float:
        """Stop a named timer and return GPU time in ms."""
        if name not in self._active_timers:
            raise ValueError(f"Timer '{name}' was not started")
        start_event, end_event, cpu_start = self._active_timers.pop(name)
        end_event.record()
        if self.sync:
            torch.cuda.synchronize()
        gpu_ms = start_event.elapsed_time(end_event)
        cpu_ms = (time.perf_counter() - cpu_start) * 1000

        if name in self.results:
            self.results[name].gpu_ms += gpu_ms
            self.results[name].cpu_ms += cpu_ms
            self.results[name].count += 1
        else:
            self.results[name] = TimingResult(name=name, gpu_ms=gpu_ms, cpu_ms=cpu_ms, count=1)
        return gpu_ms

    @contextmanager
    def measure(self, name: str):
        """Context manager for timing a block."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def reset(self):
        self.results.clear()
        self._active_timers.clear()

    def summary(self) -> str:
        """Return a formatted summary table."""
        lines = []
        lines.append(f"{'Component':<40} {'GPU ms':>10} {'Count':>6} {'Avg ms':>10}")
        lines.append("-" * 70)
        for r in self.results.values():
            lines.append(f"{r.name:<40} {r.gpu_ms:>10.2f} {r.count:>6d} {r.gpu_avg_ms:>10.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            name: {
                "total_gpu_ms": r.gpu_ms,
                "total_cpu_ms": r.cpu_ms,
                "count": r.count,
                "avg_gpu_ms": r.gpu_avg_ms,
                "avg_cpu_ms": r.cpu_avg_ms,
            }
            for name, r in self.results.items()
        }

    def save(self, path: str, fmt: str = "json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict()
        if fmt == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif fmt == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["component", "total_gpu_ms", "count", "avg_gpu_ms"])
                for name, d in data.items():
                    writer.writerow([name, d["total_gpu_ms"], d["count"], d["avg_gpu_ms"]])


class FPSTracker:
    """Track frames-per-second over a streaming run."""

    def __init__(self, warmup_frames: int = 5):
        self.warmup_frames = warmup_frames
        self.frame_times: list[float] = []
        self.total_frames = 0
        self._last_time: Optional[float] = None

    def tick(self, num_frames: int = 1):
        """Call after processing num_frames."""
        now = time.perf_counter()
        self.total_frames += num_frames
        if self._last_time is not None and self.total_frames > self.warmup_frames:
            self.frame_times.append((now - self._last_time) / num_frames)
        self._last_time = now

    @property
    def avg_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    @property
    def avg_ms_per_frame(self) -> float:
        if not self.frame_times:
            return 0.0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000

    @property
    def min_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return 1.0 / max(self.frame_times)

    @property
    def max_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return 1.0 / min(self.frame_times)

    @property
    def jitter_ms(self) -> float:
        """Standard deviation of frame times in ms."""
        if len(self.frame_times) < 2:
            return 0.0
        import statistics
        return statistics.stdev(self.frame_times) * 1000

    def summary(self) -> dict:
        return {
            "avg_fps": round(self.avg_fps, 2),
            "avg_ms_per_frame": round(self.avg_ms_per_frame, 2),
            "min_fps": round(self.min_fps, 2),
            "max_fps": round(self.max_fps, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "total_frames": self.total_frames,
            "measured_frames": len(self.frame_times),
        }


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1),
    }


def estimate_kv_cache_memory_mb(
    num_blocks: int, cache_length: int, num_heads: int, head_dim: int = 128, dtype_bytes: int = 2
) -> float:
    """Estimate KV cache memory in MB. dtype_bytes=2 for BF16."""
    # Each block has k and v, each of shape [B, cache_length, num_heads, head_dim]
    per_block_bytes = 2 * cache_length * num_heads * head_dim * dtype_bytes
    total_bytes = num_blocks * per_block_bytes
    return total_bytes / 1024**2
