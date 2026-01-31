"""
Memory analyzer for tracking GPU/CPU memory usage per layer.
"""

from typing import Any

import torch
import torch.nn as nn

from ..report import MemoryStats
from .base import BaseAnalyzer


class MemoryAnalyzer(BaseAnalyzer):
    """
    Analyzer for tracking memory allocation during forward passes.

    For CUDA devices, uses torch.cuda.memory_stats() for accurate tracking.
    For CPU, provides estimates based on tensor sizes.

    Note: CUDA memory tracking requires synchronization which adds overhead.
    """

    def __init__(self, track_allocations: bool = True):
        """
        Args:
            track_allocations: Whether to track number of allocations (more overhead)
        """
        super().__init__()
        self.track_allocations = track_allocations
        self._cuda_available = torch.cuda.is_available()
        self._stats_by_layer: dict[str, MemoryStats] = {}

    def before_forward(self, module: nn.Module, layer_name: str, inputs: tuple) -> dict[str, Any]:
        """Capture memory state before forward pass."""
        if not self._enabled:
            return {}

        state = {"is_cuda": False}

        if self._cuda_available and self.is_cuda(module):
            device = self.get_device(module)
            device_idx = device.index if device.index is not None else 0

            # Synchronize to ensure accurate measurement
            torch.cuda.synchronize(device)

            state["is_cuda"] = True
            state["device_idx"] = device_idx
            state["allocated_before"] = torch.cuda.memory_allocated(device)
            state["reserved_before"] = torch.cuda.memory_reserved(device)

            if self.track_allocations:
                stats = torch.cuda.memory_stats(device)
                state["allocs_before"] = stats.get("num_alloc_retries", 0)
        else:
            # CPU memory - we can only estimate based on output tensor sizes
            state["is_cuda"] = False

        return state

    def after_forward(
        self, module: nn.Module, layer_name: str, inputs: tuple, outputs: Any, state: dict[str, Any]
    ) -> MemoryStats:
        """Capture memory state after forward pass and compute delta."""
        if not self._enabled or not state:
            return MemoryStats()

        if state.get("is_cuda"):
            device_idx = state["device_idx"]
            device = torch.device("cuda", device_idx)

            # Synchronize for accurate measurement
            torch.cuda.synchronize(device)

            allocated_after = torch.cuda.memory_allocated(device)
            reserved_after = torch.cuda.memory_reserved(device)

            # Peak memory since last reset
            peak_allocated = torch.cuda.max_memory_allocated(device)

            stats = MemoryStats(
                allocated_delta=allocated_after - state["allocated_before"],
                peak_delta=peak_allocated - state["allocated_before"],
                reserved_delta=reserved_after - state["reserved_before"],
                is_cuda=True,
                device_index=device_idx,
            )

            if self.track_allocations:
                cuda_stats = torch.cuda.memory_stats(device)
                stats.num_allocs = cuda_stats.get("num_alloc_retries", 0) - state.get("allocs_before", 0)
        else:
            # CPU: Estimate memory from output tensor sizes
            output_bytes = self._estimate_tensor_bytes(outputs)
            stats = MemoryStats(
                allocated_delta=output_bytes,
                peak_delta=output_bytes,
                is_cuda=False,
            )

        # Accumulate stats
        if layer_name in self._stats_by_layer:
            existing = self._stats_by_layer[layer_name]
            stats.allocated_delta += existing.allocated_delta
            stats.peak_delta = max(stats.peak_delta, existing.peak_delta)
            stats.reserved_delta += existing.reserved_delta
            stats.num_allocs += existing.num_allocs

        self._stats_by_layer[layer_name] = stats
        return stats

    def _estimate_tensor_bytes(self, obj: Any) -> int:
        """Recursively estimate memory usage of tensors in an object."""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        elif isinstance(obj, (tuple, list)):
            return sum(self._estimate_tensor_bytes(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_tensor_bytes(v) for v in obj.values())
        elif hasattr(obj, "__dict__"):
            # Handle objects with tensor attributes
            return sum(self._estimate_tensor_bytes(v) for v in obj.__dict__.values() if isinstance(v, torch.Tensor))
        return 0

    def get_stats(self, layer_name: str) -> MemoryStats | None:
        """Get accumulated stats for a specific layer."""
        return self._stats_by_layer.get(layer_name)

    def get_all_stats(self) -> dict[str, MemoryStats]:
        """Get all accumulated layer stats."""
        return self._stats_by_layer.copy()

    def get_total_stats(self) -> MemoryStats:
        """Get aggregated stats across all layers."""
        total = MemoryStats()
        for stats in self._stats_by_layer.values():
            total.allocated_delta += stats.allocated_delta
            total.peak_delta = max(total.peak_delta, stats.peak_delta)
            total.reserved_delta += stats.reserved_delta
            total.num_allocs += stats.num_allocs
            if stats.is_cuda:
                total.is_cuda = True
                total.device_index = stats.device_index
        return total

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._stats_by_layer.clear()
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats()

    def reset_cuda_stats(self) -> None:
        """Reset CUDA memory statistics for fresh measurement."""
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
