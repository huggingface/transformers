"""
Timing analyzer for tracking execution time per layer.
"""

import time
from typing import Any

import torch
import torch.nn as nn

from ..report import TimingStats
from .base import BaseAnalyzer


class TimingAnalyzer(BaseAnalyzer):
    """
    Analyzer for tracking execution time during forward passes.

    Uses time.perf_counter for high-resolution timing. For CUDA devices,
    optionally synchronizes before measurements for accurate GPU timing.

    Note: CUDA synchronization adds overhead but is required for accurate
    GPU timing measurements.
    """

    def __init__(self, sync_cuda: bool = True, use_cuda_events: bool = False):
        """
        Args:
            sync_cuda: Whether to synchronize CUDA before timing measurements.
                Required for accurate GPU timing but adds overhead.
            use_cuda_events: Whether to use CUDA events for timing (more accurate
                for GPU operations but more complex).
        """
        super().__init__()
        self.sync_cuda = sync_cuda
        self.use_cuda_events = use_cuda_events
        self._cuda_available = torch.cuda.is_available()
        self._stats_by_layer: dict[str, TimingStats] = {}
        self._total_time: float = 0.0

    def before_forward(self, module: nn.Module, layer_name: str, inputs: tuple) -> dict[str, Any]:
        """Record start time before forward pass."""
        if not self._enabled:
            return {}

        state = {"is_cuda": False}

        is_cuda = self._cuda_available and self.is_cuda(module)
        state["is_cuda"] = is_cuda

        if is_cuda and self.sync_cuda:
            device = self.get_device(module)
            torch.cuda.synchronize(device)
            state["device"] = device

            if self.use_cuda_events:
                state["start_event"] = torch.cuda.Event(enable_timing=True)
                state["start_event"].record()

        state["start_time"] = time.perf_counter()
        return state

    def after_forward(
        self, module: nn.Module, layer_name: str, inputs: tuple, outputs: Any, state: dict[str, Any]
    ) -> TimingStats:
        """Record end time and compute elapsed time."""
        if not self._enabled or not state:
            return TimingStats()

        if state.get("is_cuda") and self.sync_cuda:
            device = state.get("device")
            torch.cuda.synchronize(device)

            if self.use_cuda_events and "start_event" in state:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize(device)
                elapsed = state["start_event"].elapsed_time(end_event) / 1000  # Convert ms to s
            else:
                elapsed = time.perf_counter() - state["start_time"]
        else:
            elapsed = time.perf_counter() - state["start_time"]

        # Update or create stats for this layer
        if layer_name not in self._stats_by_layer:
            self._stats_by_layer[layer_name] = TimingStats()

        self._stats_by_layer[layer_name].update(elapsed)
        self._total_time += elapsed

        return self._stats_by_layer[layer_name]

    def get_stats(self, layer_name: str) -> TimingStats | None:
        """Get accumulated stats for a specific layer."""
        return self._stats_by_layer.get(layer_name)

    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get all accumulated layer stats."""
        return self._stats_by_layer.copy()

    def get_total_time(self) -> float:
        """Get total time across all layers."""
        return self._total_time

    def get_percentage_by_layer(self) -> dict[str, float]:
        """Get time percentage for each layer."""
        if self._total_time == 0:
            return {}
        return {name: (stats.total_time / self._total_time) * 100 for name, stats in self._stats_by_layer.items()}

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._stats_by_layer.clear()
        self._total_time = 0.0


class CudaEventTimer:
    """
    Context manager for timing CUDA operations with CUDA events.

    More accurate than wall-clock timing for GPU operations.

    Example:
        >>> with CudaEventTimer() as timer:
        ...     result = model(inputs)
        >>> print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None
        self._elapsed: float | None = None

    def __enter__(self) -> "CudaEventTimer":
        if not torch.cuda.is_available():
            self._start_cpu = time.perf_counter()
            return self

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        if self.device:
            torch.cuda.synchronize(self.device)
        else:
            torch.cuda.synchronize()

        self.start_event.record()
        return self

    def __exit__(self, *args) -> None:
        if not torch.cuda.is_available():
            self._elapsed = (time.perf_counter() - self._start_cpu) * 1000
            return

        self.end_event.record()

        if self.device:
            torch.cuda.synchronize(self.device)
        else:
            torch.cuda.synchronize()

        self._elapsed = self.start_event.elapsed_time(self.end_event)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._elapsed is None:
            raise RuntimeError("Timer has not been stopped yet")
        return self._elapsed

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ms / 1000
