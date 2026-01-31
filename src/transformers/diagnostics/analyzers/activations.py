"""
Activation analyzer for collecting statistics about layer outputs.
"""

import math
from typing import Any

import torch
import torch.nn as nn

from ..report import ActivationStats
from .base import BaseAnalyzer


class ActivationAnalyzer(BaseAnalyzer):
    """
    Analyzer for collecting statistics about layer activations (outputs).

    Computes configurable statistics (mean, std, min, max, norm, etc.)
    on layer outputs without storing the full tensors.

    Note: Statistics are computed on-the-fly to minimize memory overhead.
    For large tensors, sampling can be used to reduce computation time.
    """

    # Available statistics and their computation methods
    STAT_FUNCTIONS = {
        "mean": lambda t: t.float().mean().item(),
        "std": lambda t: t.float().std().item(),
        "min": lambda t: t.float().min().item(),
        "max": lambda t: t.float().max().item(),
        "norm": lambda t: t.float().norm().item(),
        "abs_mean": lambda t: t.float().abs().mean().item(),
        "sparsity": lambda t: (t == 0).float().mean().item(),
        "numel": lambda t: t.numel(),
        "shape": lambda t: tuple(t.shape),
        "dtype": lambda t: str(t.dtype),
    }

    def __init__(
        self,
        stats: set[str] | None = None,
        max_elements: int = 1_000_000,
        include_inputs: bool = False,
    ):
        """
        Args:
            stats: Set of statistics to compute. If None, computes all basic stats.
            max_elements: Maximum elements to sample for stats computation.
                Use -1 for all elements.
            include_inputs: Whether to also compute stats on layer inputs.
        """
        super().__init__()
        self.stats = stats or {"mean", "std", "min", "max", "norm"}
        self.max_elements = max_elements
        self.include_inputs = include_inputs
        self._stats_by_layer: dict[str, ActivationStats] = {}
        self._input_stats_by_layer: dict[str, ActivationStats] = {}

        # Validate stats
        invalid = self.stats - set(self.STAT_FUNCTIONS.keys())
        if invalid:
            raise ValueError(f"Unknown statistics: {invalid}")

    def before_forward(self, module: nn.Module, layer_name: str, inputs: tuple) -> dict[str, Any]:
        """Optionally compute input statistics."""
        if not self._enabled:
            return {}

        state = {}

        if self.include_inputs and inputs:
            input_tensor = self._get_main_tensor(inputs)
            if input_tensor is not None:
                state["input_stats"] = self._compute_stats(input_tensor)

        return state

    def after_forward(
        self, module: nn.Module, layer_name: str, inputs: tuple, outputs: Any, state: dict[str, Any]
    ) -> ActivationStats:
        """Compute output statistics."""
        if not self._enabled:
            return ActivationStats()

        # Get the main output tensor
        output_tensor = self._get_main_tensor(outputs)

        if output_tensor is None:
            return ActivationStats()

        # Compute stats
        activation_stats = self._compute_stats(output_tensor)

        # Handle aggregation
        if layer_name in self._stats_by_layer:
            existing = self._stats_by_layer[layer_name]
            activation_stats = self._merge_stats(existing, activation_stats)

        self._stats_by_layer[layer_name] = activation_stats

        # Store input stats if computed
        if "input_stats" in state:
            if layer_name in self._input_stats_by_layer:
                existing = self._input_stats_by_layer[layer_name]
                state["input_stats"] = self._merge_stats(existing, state["input_stats"])
            self._input_stats_by_layer[layer_name] = state["input_stats"]

        return activation_stats

    def _get_main_tensor(self, obj: Any) -> torch.Tensor | None:
        """Extract the main tensor from various output formats."""
        if isinstance(obj, torch.Tensor):
            return obj
        elif isinstance(obj, (tuple, list)) and len(obj) > 0:
            # Return first tensor found
            for item in obj:
                if isinstance(item, torch.Tensor):
                    return item
                elif isinstance(item, (tuple, list)):
                    result = self._get_main_tensor(item)
                    if result is not None:
                        return result
        elif hasattr(obj, "last_hidden_state"):
            # Handle transformers model outputs
            return obj.last_hidden_state
        elif hasattr(obj, "logits"):
            return obj.logits
        return None

    def _compute_stats(self, tensor: torch.Tensor) -> ActivationStats:
        """Compute statistics on a tensor."""
        # Sample if tensor is too large
        if self.max_elements > 0 and tensor.numel() > self.max_elements:
            tensor = self._sample_tensor(tensor, self.max_elements)

        stats = ActivationStats(num_samples=1)

        # Compute each requested statistic
        try:
            if "mean" in self.stats:
                stats.mean = self.STAT_FUNCTIONS["mean"](tensor)
            if "std" in self.stats:
                stats.std = self.STAT_FUNCTIONS["std"](tensor)
            if "min" in self.stats:
                stats.min_val = self.STAT_FUNCTIONS["min"](tensor)
            if "max" in self.stats:
                stats.max_val = self.STAT_FUNCTIONS["max"](tensor)
            if "norm" in self.stats:
                stats.norm = self.STAT_FUNCTIONS["norm"](tensor)
            if "abs_mean" in self.stats:
                stats.abs_mean = self.STAT_FUNCTIONS["abs_mean"](tensor)
            if "sparsity" in self.stats:
                stats.sparsity = self.STAT_FUNCTIONS["sparsity"](tensor)
            if "shape" in self.stats:
                stats.shape = tuple(tensor.shape)
            if "numel" in self.stats:
                stats.numel = tensor.numel()
            if "dtype" in self.stats:
                stats.dtype = str(tensor.dtype)
        except Exception:
            # Handle any tensor operation errors gracefully
            pass

        return stats

    def _sample_tensor(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Randomly sample n elements from a tensor."""
        flat = tensor.flatten()
        if flat.numel() <= n:
            return flat
        indices = torch.randperm(flat.numel(), device=tensor.device)[:n]
        return flat[indices]

    def _merge_stats(self, existing: ActivationStats, new: ActivationStats) -> ActivationStats:
        """Merge two ActivationStats using running mean/variance."""
        n1 = existing.num_samples
        n2 = new.num_samples
        n = n1 + n2

        merged = ActivationStats(num_samples=n)

        # Running mean
        if existing.mean is not None and new.mean is not None:
            merged.mean = (n1 * existing.mean + n2 * new.mean) / n

        # Running variance (Welford's parallel algorithm)
        if existing.std is not None and new.std is not None and existing.mean is not None and new.mean is not None:
            var1 = existing.std**2
            var2 = new.std**2
            delta = new.mean - existing.mean
            merged_var = (n1 * var1 + n2 * var2 + delta**2 * n1 * n2 / n) / n
            merged.std = math.sqrt(merged_var)

        # Min/max
        if existing.min_val is not None and new.min_val is not None:
            merged.min_val = min(existing.min_val, new.min_val)
        if existing.max_val is not None and new.max_val is not None:
            merged.max_val = max(existing.max_val, new.max_val)

        # Keep latest for non-aggregatable stats
        merged.shape = new.shape or existing.shape
        merged.dtype = new.dtype or existing.dtype
        merged.numel = new.numel or existing.numel
        merged.norm = new.norm  # Just keep latest for norm
        merged.abs_mean = new.abs_mean
        merged.sparsity = new.sparsity

        return merged

    def get_stats(self, layer_name: str) -> ActivationStats | None:
        """Get accumulated stats for a specific layer."""
        return self._stats_by_layer.get(layer_name)

    def get_input_stats(self, layer_name: str) -> ActivationStats | None:
        """Get accumulated input stats for a specific layer."""
        return self._input_stats_by_layer.get(layer_name)

    def get_all_stats(self) -> dict[str, ActivationStats]:
        """Get all accumulated layer stats."""
        return self._stats_by_layer.copy()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._stats_by_layer.clear()
        self._input_stats_by_layer.clear()


def compute_activation_stats(
    tensor: torch.Tensor,
    stats: set[str] | None = None,
) -> dict[str, Any]:
    """
    Standalone function to compute activation statistics on a tensor.

    Args:
        tensor: The tensor to analyze
        stats: Set of statistics to compute. Default: mean, std, min, max, norm

    Returns:
        Dictionary of computed statistics
    """
    stats = stats or {"mean", "std", "min", "max", "norm"}
    result = {}

    for stat_name in stats:
        if stat_name in ActivationAnalyzer.STAT_FUNCTIONS:
            try:
                result[stat_name] = ActivationAnalyzer.STAT_FUNCTIONS[stat_name](tensor)
            except Exception:
                result[stat_name] = None

    return result
