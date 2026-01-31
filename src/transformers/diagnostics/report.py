"""
Diagnostics report structures for model introspection.

This module provides dataclasses for storing and presenting diagnostic results.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryStats:
    """Memory statistics for a single layer or the full model."""

    # Memory allocated during forward pass (bytes)
    allocated_delta: int = 0
    # Peak memory during forward pass (bytes)
    peak_delta: int = 0
    # Memory reserved by allocator (bytes)
    reserved_delta: int = 0
    # Number of allocations
    num_allocs: int = 0
    # Whether this is GPU memory
    is_cuda: bool = False
    # Device index (for multi-GPU)
    device_index: int = 0

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in megabytes."""
        return self.allocated_delta / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak_delta / (1024 * 1024)

    def __repr__(self) -> str:
        device = f"cuda:{self.device_index}" if self.is_cuda else "cpu"
        return f"MemoryStats({device}: alloc={self.allocated_mb:.2f}MB, peak={self.peak_mb:.2f}MB)"


@dataclass
class TimingStats:
    """Timing statistics for a single layer or the full model."""

    # Total time in seconds
    total_time: float = 0.0
    # Number of calls
    num_calls: int = 0
    # Min time per call
    min_time: float = float("inf")
    # Max time per call
    max_time: float = 0.0
    # For computing running variance
    _m2: float = field(default=0.0, repr=False)
    _mean: float = field(default=0.0, repr=False)

    @property
    def mean_time(self) -> float:
        """Average time per call in seconds."""
        return self.total_time / max(self.num_calls, 1)

    @property
    def mean_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        return self.mean_time * 1000

    @property
    def total_time_ms(self) -> float:
        """Total time in milliseconds."""
        return self.total_time * 1000

    @property
    def std_time(self) -> float:
        """Standard deviation of time per call."""
        if self.num_calls < 2:
            return 0.0
        return (self._m2 / self.num_calls) ** 0.5

    def update(self, elapsed: float) -> None:
        """Update statistics with a new timing measurement (Welford's algorithm)."""
        self.num_calls += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

        # Welford's online algorithm for variance
        delta = elapsed - self._mean
        self._mean += delta / self.num_calls
        delta2 = elapsed - self._mean
        self._m2 += delta * delta2

    def __repr__(self) -> str:
        return f"TimingStats(total={self.total_time_ms:.2f}ms, mean={self.mean_time_ms:.3f}ms, calls={self.num_calls})"


@dataclass
class ActivationStats:
    """Statistics about layer activations (outputs)."""

    # Basic stats
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None

    # Additional stats
    norm: float | None = None  # L2 norm
    abs_mean: float | None = None  # Mean of absolute values
    sparsity: float | None = None  # Fraction of zeros

    # Shape info
    shape: tuple | None = None
    numel: int | None = None
    dtype: str | None = None

    # Aggregation info
    num_samples: int = 0

    def __repr__(self) -> str:
        parts = []
        if self.mean is not None:
            parts.append(f"mean={self.mean:.4f}")
        if self.std is not None:
            parts.append(f"std={self.std:.4f}")
        if self.shape is not None:
            parts.append(f"shape={self.shape}")
        return f"ActivationStats({', '.join(parts)})"


@dataclass
class LayerDiagnostics:
    """Combined diagnostics for a single layer."""

    name: str
    memory: MemoryStats | None = None
    timing: TimingStats | None = None
    activations: ActivationStats | None = None
    input_activations: ActivationStats | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"name": self.name}
        if self.memory:
            result["memory"] = asdict(self.memory)
        if self.timing:
            result["timing"] = {
                "total_time_ms": self.timing.total_time_ms,
                "mean_time_ms": self.timing.mean_time_ms,
                "num_calls": self.timing.num_calls,
                "min_time": self.timing.min_time,
                "max_time": self.timing.max_time,
            }
        if self.activations:
            result["activations"] = asdict(self.activations)
        if self.input_activations:
            result["input_activations"] = asdict(self.input_activations)
        return result


@dataclass
class DiagnosticsReport:
    """
    Complete diagnostics report for a model forward pass.

    This class aggregates all collected metrics and provides various
    export and visualization methods.

    Attributes:
        layers: Dictionary mapping layer names to their diagnostics
        total_time: Total forward pass time in seconds
        total_memory: Aggregated memory statistics
        model_name: Name of the model being profiled
        num_forward_passes: Number of forward passes aggregated
        metadata: Additional metadata (device, dtype, etc.)
    """

    layers: dict[str, LayerDiagnostics] = field(default_factory=dict)
    total_time: float = 0.0
    total_memory: MemoryStats | None = None
    model_name: str = ""
    num_forward_passes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_names(self) -> list[str]:
        """Get ordered list of layer names."""
        return list(self.layers.keys())

    @property
    def memory_by_layer(self) -> dict[str, MemoryStats]:
        """Get memory stats indexed by layer name."""
        return {name: diag.memory for name, diag in self.layers.items() if diag.memory is not None}

    @property
    def timing_by_layer(self) -> dict[str, TimingStats]:
        """Get timing stats indexed by layer name."""
        return {name: diag.timing for name, diag in self.layers.items() if diag.timing is not None}

    @property
    def activations_by_layer(self) -> dict[str, ActivationStats]:
        """Get activation stats indexed by layer name."""
        return {name: diag.activations for name, diag in self.layers.items() if diag.activations is not None}

    def get_slowest_layers(self, n: int = 10) -> list[tuple]:
        """Get the n slowest layers by total time."""
        timing = self.timing_by_layer
        sorted_layers = sorted(timing.items(), key=lambda x: x[1].total_time, reverse=True)
        return sorted_layers[:n]

    def get_memory_hotspots(self, n: int = 10) -> list[tuple]:
        """Get the n layers with highest memory allocation."""
        memory = self.memory_by_layer
        sorted_layers = sorted(memory.items(), key=lambda x: x[1].allocated_delta, reverse=True)
        return sorted_layers[:n]

    def summary(self, top_n: int = 5) -> str:
        """Generate a human-readable summary of the diagnostics."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"DIAGNOSTICS REPORT: {self.model_name or 'Model'}")
        lines.append("=" * 60)
        lines.append(f"Forward passes: {self.num_forward_passes}")
        lines.append(f"Total layers instrumented: {len(self.layers)}")
        lines.append(f"Total time: {self.total_time * 1000:.2f}ms")

        if self.total_memory:
            lines.append(f"Total memory allocated: {self.total_memory.allocated_mb:.2f}MB")
            lines.append(f"Peak memory: {self.total_memory.peak_mb:.2f}MB")

        lines.append("")

        # Timing breakdown
        timing = self.timing_by_layer
        if timing:
            lines.append(f"TOP {top_n} SLOWEST LAYERS:")
            lines.append("-" * 40)
            for name, stats in self.get_slowest_layers(top_n):
                pct = (stats.total_time / max(self.total_time, 1e-9)) * 100
                lines.append(f"  {name[:40]:<40} {stats.total_time_ms:>8.2f}ms ({pct:>5.1f}%)")
            lines.append("")

        # Memory breakdown
        memory = self.memory_by_layer
        if memory:
            lines.append(f"TOP {top_n} MEMORY CONSUMERS:")
            lines.append("-" * 40)
            for name, stats in self.get_memory_hotspots(top_n):
                lines.append(f"  {name[:40]:<40} {stats.allocated_mb:>8.2f}MB")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "num_forward_passes": self.num_forward_passes,
            "total_time_ms": self.total_time * 1000,
            "total_memory": asdict(self.total_memory) if self.total_memory else None,
            "layers": {name: diag.to_dict() for name, diag in self.layers.items()},
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_dataframe(self):
        """
        Convert report to pandas DataFrame.

        Returns:
            pandas.DataFrame with one row per layer and columns for each metric.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        rows = []
        for name, diag in self.layers.items():
            row = {"layer_name": name}

            if diag.timing:
                row["time_ms"] = diag.timing.total_time_ms
                row["time_pct"] = (diag.timing.total_time / max(self.total_time, 1e-9)) * 100
                row["calls"] = diag.timing.num_calls
                row["mean_time_ms"] = diag.timing.mean_time_ms

            if diag.memory:
                row["memory_mb"] = diag.memory.allocated_mb
                row["peak_mb"] = diag.memory.peak_mb

            if diag.activations:
                row["act_mean"] = diag.activations.mean
                row["act_std"] = diag.activations.std
                row["act_norm"] = diag.activations.norm
                row["act_shape"] = str(diag.activations.shape)

            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "DiagnosticsReport":
        """Load report from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct the report
        report = cls(
            model_name=data.get("model_name", ""),
            num_forward_passes=data.get("num_forward_passes", 0),
            total_time=data.get("total_time_ms", 0) / 1000,
            metadata=data.get("metadata", {}),
        )

        if data.get("total_memory"):
            report.total_memory = MemoryStats(**data["total_memory"])

        for name, layer_data in data.get("layers", {}).items():
            diag = LayerDiagnostics(name=name)
            if "memory" in layer_data:
                diag.memory = MemoryStats(**layer_data["memory"])
            if "timing" in layer_data:
                t = layer_data["timing"]
                diag.timing = TimingStats(
                    total_time=t.get("total_time_ms", 0) / 1000,
                    num_calls=t.get("num_calls", 0),
                    min_time=t.get("min_time", float("inf")),
                    max_time=t.get("max_time", 0),
                )
            if "activations" in layer_data:
                a = layer_data["activations"]
                diag.activations = ActivationStats(
                    mean=a.get("mean"),
                    std=a.get("std"),
                    min_val=a.get("min_val"),
                    max_val=a.get("max_val"),
                    norm=a.get("norm"),
                    abs_mean=a.get("abs_mean"),
                    sparsity=a.get("sparsity"),
                    shape=tuple(a["shape"]) if a.get("shape") else None,
                    numel=a.get("numel"),
                    dtype=a.get("dtype"),
                    num_samples=a.get("num_samples", 0),
                )
            report.layers[name] = diag

        return report

    def __repr__(self) -> str:
        return (
            f"DiagnosticsReport(model={self.model_name!r}, "
            f"layers={len(self.layers)}, "
            f"time={self.total_time * 1000:.2f}ms, "
            f"passes={self.num_forward_passes})"
        )
