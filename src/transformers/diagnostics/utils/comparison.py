"""
Utilities for comparing diagnostics reports between models or runs.
"""

from dataclasses import dataclass, field

from ..report import DiagnosticsReport


@dataclass
class LayerComparison:
    """Comparison data for a single layer."""

    name: str

    # Timing comparison
    time_a: float | None = None
    time_b: float | None = None
    time_diff: float | None = None
    time_ratio: float | None = None

    # Memory comparison
    memory_a: float | None = None
    memory_b: float | None = None
    memory_diff: float | None = None
    memory_ratio: float | None = None

    @property
    def time_pct_change(self) -> float | None:
        """Percentage change in timing."""
        if self.time_a and self.time_b:
            return ((self.time_b - self.time_a) / self.time_a) * 100
        return None

    @property
    def memory_pct_change(self) -> float | None:
        """Percentage change in memory."""
        if self.memory_a and self.memory_b:
            return ((self.memory_b - self.memory_a) / self.memory_a) * 100
        return None


@dataclass
class ReportComparison:
    """
    Comparison between two diagnostics reports.

    Useful for comparing:
    - Same model with different configurations
    - Different model versions
    - Before/after optimization
    """

    report_a: DiagnosticsReport
    report_b: DiagnosticsReport
    name_a: str = "A"
    name_b: str = "B"

    layer_comparisons: dict[str, LayerComparison] = field(default_factory=dict)

    # Aggregate comparisons
    total_time_a: float = 0.0
    total_time_b: float = 0.0
    total_memory_a: float = 0.0
    total_memory_b: float = 0.0

    def __post_init__(self):
        """Compute comparisons on initialization."""
        self._compute_comparisons()

    def _compute_comparisons(self) -> None:
        """Compute layer-by-layer comparisons."""
        # Get all layer names from both reports
        all_layers = set(self.report_a.layer_names) | set(self.report_b.layer_names)

        for layer_name in all_layers:
            comparison = LayerComparison(name=layer_name)

            # Timing comparison
            timing_a = self.report_a.timing_by_layer.get(layer_name)
            timing_b = self.report_b.timing_by_layer.get(layer_name)

            if timing_a:
                comparison.time_a = timing_a.total_time_ms
                self.total_time_a += timing_a.total_time_ms
            if timing_b:
                comparison.time_b = timing_b.total_time_ms
                self.total_time_b += timing_b.total_time_ms
            if timing_a and timing_b:
                comparison.time_diff = timing_b.total_time_ms - timing_a.total_time_ms
                comparison.time_ratio = timing_b.total_time_ms / max(timing_a.total_time_ms, 1e-9)

            # Memory comparison
            memory_a = self.report_a.memory_by_layer.get(layer_name)
            memory_b = self.report_b.memory_by_layer.get(layer_name)

            if memory_a:
                comparison.memory_a = memory_a.allocated_mb
                self.total_memory_a += memory_a.allocated_mb
            if memory_b:
                comparison.memory_b = memory_b.allocated_mb
                self.total_memory_b += memory_b.allocated_mb
            if memory_a and memory_b:
                comparison.memory_diff = memory_b.allocated_mb - memory_a.allocated_mb
                comparison.memory_ratio = memory_b.allocated_mb / max(memory_a.allocated_mb, 1e-9)

            self.layer_comparisons[layer_name] = comparison

    @property
    def total_time_diff(self) -> float:
        """Total time difference in ms (B - A)."""
        return self.total_time_b - self.total_time_a

    @property
    def total_time_ratio(self) -> float:
        """Total time ratio (B / A)."""
        return self.total_time_b / max(self.total_time_a, 1e-9)

    @property
    def total_time_pct_change(self) -> float:
        """Total time percentage change."""
        return ((self.total_time_b - self.total_time_a) / max(self.total_time_a, 1e-9)) * 100

    @property
    def total_memory_diff(self) -> float:
        """Total memory difference in MB (B - A)."""
        return self.total_memory_b - self.total_memory_a

    @property
    def total_memory_ratio(self) -> float:
        """Total memory ratio (B / A)."""
        return self.total_memory_b / max(self.total_memory_a, 1e-9)

    def get_speedups(self, min_speedup: float = 1.1) -> list[tuple[str, float]]:
        """Get layers where B is faster than A (speedup > min_speedup)."""
        speedups = []
        for name, comp in self.layer_comparisons.items():
            if comp.time_ratio and comp.time_ratio < (1 / min_speedup):
                speedups.append((name, 1 / comp.time_ratio))
        return sorted(speedups, key=lambda x: x[1], reverse=True)

    def get_slowdowns(self, min_slowdown: float = 1.1) -> list[tuple[str, float]]:
        """Get layers where B is slower than A (slowdown > min_slowdown)."""
        slowdowns = []
        for name, comp in self.layer_comparisons.items():
            if comp.time_ratio and comp.time_ratio > min_slowdown:
                slowdowns.append((name, comp.time_ratio))
        return sorted(slowdowns, key=lambda x: x[1], reverse=True)

    def get_biggest_time_changes(self, n: int = 10) -> list[tuple[str, float]]:
        """Get layers with the biggest absolute time changes."""
        changes = [
            (name, abs(comp.time_diff)) for name, comp in self.layer_comparisons.items() if comp.time_diff is not None
        ]
        return sorted(changes, key=lambda x: x[1], reverse=True)[:n]

    def summary(self) -> str:
        """Generate a human-readable comparison summary."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"COMPARISON: {self.name_a} vs {self.name_b}")
        lines.append("=" * 70)
        lines.append("")

        # Overall metrics
        lines.append("OVERALL METRICS:")
        lines.append("-" * 50)

        time_sign = "+" if self.total_time_diff >= 0 else ""
        lines.append(
            f"  Total Time:   {self.name_a}={self.total_time_a:.2f}ms  "
            f"{self.name_b}={self.total_time_b:.2f}ms  "
            f"({time_sign}{self.total_time_pct_change:.1f}%)"
        )

        mem_sign = "+" if self.total_memory_diff >= 0 else ""
        lines.append(
            f"  Total Memory: {self.name_a}={self.total_memory_a:.2f}MB  "
            f"{self.name_b}={self.total_memory_b:.2f}MB  "
            f"({mem_sign}{self.total_memory_diff:.2f}MB)"
        )
        lines.append("")

        # Speedups
        speedups = self.get_speedups()
        if speedups:
            lines.append(f"SPEEDUPS ({self.name_b} faster than {self.name_a}):")
            lines.append("-" * 50)
            for name, ratio in speedups[:5]:
                lines.append(f"  {name[:45]:<45} {ratio:.2f}x faster")
            lines.append("")

        # Slowdowns
        slowdowns = self.get_slowdowns()
        if slowdowns:
            lines.append(f"SLOWDOWNS ({self.name_b} slower than {self.name_a}):")
            lines.append("-" * 50)
            for name, ratio in slowdowns[:5]:
                lines.append(f"  {name[:45]:<45} {ratio:.2f}x slower")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dataframe(self):
        """Convert comparison to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

        rows = []
        for name, comp in self.layer_comparisons.items():
            rows.append(
                {
                    "layer": name,
                    f"time_{self.name_a}_ms": comp.time_a,
                    f"time_{self.name_b}_ms": comp.time_b,
                    "time_diff_ms": comp.time_diff,
                    "time_ratio": comp.time_ratio,
                    "time_pct_change": comp.time_pct_change,
                    f"memory_{self.name_a}_mb": comp.memory_a,
                    f"memory_{self.name_b}_mb": comp.memory_b,
                    "memory_diff_mb": comp.memory_diff,
                    "memory_ratio": comp.memory_ratio,
                }
            )

        return pd.DataFrame(rows)


def compare_reports(
    report_a: DiagnosticsReport,
    report_b: DiagnosticsReport,
    name_a: str = "baseline",
    name_b: str = "candidate",
) -> ReportComparison:
    """
    Compare two diagnostics reports.

    Args:
        report_a: First (baseline) report
        report_b: Second (candidate) report
        name_a: Label for first report
        name_b: Label for second report

    Returns:
        ReportComparison with detailed comparison data

    Example:
        >>> # Compare two model configurations
        >>> report_baseline = diagnose_forward(model_fp32, inputs)
        >>> report_quantized = diagnose_forward(model_int8, inputs)
        >>> comparison = compare_reports(
        ...     report_baseline, report_quantized,
        ...     name_a="FP32", name_b="INT8"
        ... )
        >>> print(comparison.summary())
    """
    return ReportComparison(
        report_a=report_a,
        report_b=report_b,
        name_a=name_a,
        name_b=name_b,
    )
