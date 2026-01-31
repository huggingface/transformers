"""
Diagnostics configuration for model introspection.

This module provides the DiagnosticsConfig class which controls what metrics
are collected during model forward passes.
"""

from dataclasses import dataclass, field


@dataclass
class DiagnosticsConfig:
    """
    Configuration for model diagnostics and introspection.

    This class controls what metrics are collected during model forward passes
    when using ModelProbe.

    Args:
        track_memory: Whether to track memory usage per layer. Requires CUDA for
            accurate GPU memory tracking. CPU memory is estimated.
        track_timing: Whether to track execution time per layer. Uses
            time.perf_counter with optional CUDA synchronization.
        track_activations: Whether to collect activation statistics (mean, std, etc.)
            for layer outputs.
        layer_patterns: List of regex patterns to match layer names. Only matching
            layers will be instrumented. If empty, all layers are instrumented.
        activation_stats: List of statistics to compute for activations. Options:
            "mean", "std", "min", "max", "norm", "abs_mean", "sparsity", "shape".
        sync_cuda: Whether to synchronize CUDA before timing measurements. Required
            for accurate timing but adds overhead. Default True when CUDA available.
        include_input_stats: Whether to also track input tensor statistics (in addition
            to outputs).
        max_activation_elements: Maximum number of elements to sample for activation
            stats computation. Use -1 for all elements. Default 1M to limit memory.
        aggregate_calls: Whether to aggregate stats across multiple forward passes.
            If False, only the last forward pass is recorded.
        record_shapes: Whether to record tensor shapes in the report.
        enabled: Master switch to enable/disable all diagnostics. Useful for
            conditional profiling.

    Example:
        >>> config = DiagnosticsConfig(
        ...     track_memory=True,
        ...     track_timing=True,
        ...     track_activations=True,
        ...     layer_patterns=[".*attention.*", ".*mlp.*"],
        ...     activation_stats=["mean", "std", "norm"],
        ... )
    """

    # What to track
    track_memory: bool = True
    track_timing: bool = True
    track_activations: bool = False

    # Layer selection
    layer_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    # Activation statistics configuration
    activation_stats: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "norm"])
    include_input_stats: bool = False
    max_activation_elements: int = 1_000_000

    # Timing configuration
    sync_cuda: bool = True

    # Aggregation
    aggregate_calls: bool = True

    # Output configuration
    record_shapes: bool = True

    # Master switch
    enabled: bool = True

    # Computed fields (set in __post_init__)
    _compiled_patterns: list | None = field(default=None, repr=False, init=False)
    _compiled_excludes: list | None = field(default=None, repr=False, init=False)
    _activation_stats_set: set[str] = field(default_factory=set, repr=False, init=False)

    # Valid activation stat names
    VALID_ACTIVATION_STATS: set[str] = field(
        default_factory=lambda: {
            "mean",
            "std",
            "min",
            "max",
            "norm",
            "abs_mean",
            "sparsity",
            "shape",
            "numel",
            "dtype",
        },
        repr=False,
        init=False,
    )

    def __post_init__(self):
        """Compile regex patterns and validate configuration."""
        import re

        # Compile layer patterns
        if self.layer_patterns:
            self._compiled_patterns = [re.compile(p) for p in self.layer_patterns]
        else:
            self._compiled_patterns = None

        # Compile exclude patterns
        if self.exclude_patterns:
            self._compiled_excludes = [re.compile(p) for p in self.exclude_patterns]
        else:
            self._compiled_excludes = None

        # Validate and store activation stats as set
        invalid_stats = set(self.activation_stats) - self.VALID_ACTIVATION_STATS
        if invalid_stats:
            raise ValueError(
                f"Invalid activation stats: {invalid_stats}. Valid options: {self.VALID_ACTIVATION_STATS}"
            )
        self._activation_stats_set = set(self.activation_stats)

    def should_instrument_layer(self, layer_name: str) -> bool:
        """
        Check if a layer should be instrumented based on patterns.

        Args:
            layer_name: The name of the layer (e.g., "encoder.layer.0.attention")

        Returns:
            True if the layer should be instrumented.
        """
        # Check excludes first
        if self._compiled_excludes:
            for pattern in self._compiled_excludes:
                if pattern.search(layer_name):
                    return False

        # If no include patterns, instrument everything not excluded
        if not self._compiled_patterns:
            return True

        # Check if any include pattern matches
        for pattern in self._compiled_patterns:
            if pattern.search(layer_name):
                return True

        return False

    def get_activation_stats(self) -> set[str]:
        """Get the set of activation statistics to compute."""
        return self._activation_stats_set

    @classmethod
    def memory_only(cls) -> "DiagnosticsConfig":
        """Create a config that only tracks memory."""
        return cls(track_memory=True, track_timing=False, track_activations=False)

    @classmethod
    def timing_only(cls) -> "DiagnosticsConfig":
        """Create a config that only tracks timing."""
        return cls(track_memory=False, track_timing=True, track_activations=False)

    @classmethod
    def full(cls) -> "DiagnosticsConfig":
        """Create a config that tracks everything."""
        return cls(
            track_memory=True,
            track_timing=True,
            track_activations=True,
            activation_stats=["mean", "std", "min", "max", "norm", "abs_mean", "sparsity"],
        )

    @classmethod
    def minimal(cls) -> "DiagnosticsConfig":
        """Create a minimal config with low overhead."""
        return cls(
            track_memory=True,
            track_timing=True,
            track_activations=False,
            sync_cuda=False,  # Skip CUDA sync for lower overhead
        )
