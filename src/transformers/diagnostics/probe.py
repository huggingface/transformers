"""
ModelProbe - Hook-based model instrumentation for diagnostics collection.

This module provides the main ModelProbe class that attaches to PyTorch models
and collects diagnostics during forward passes.
"""

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from .analyzers import ActivationAnalyzer, MemoryAnalyzer, TimingAnalyzer
from .config import DiagnosticsConfig
from .report import (
    DiagnosticsReport,
    LayerDiagnostics,
)


class ModelProbe:
    """
    Hook-based model instrumentation for collecting diagnostics.

    ModelProbe attaches forward hooks to model layers and collects various
    metrics (memory, timing, activations) during forward passes.

    Can be used as a context manager for automatic cleanup:

        >>> with ModelProbe(model, config) as probe:
        ...     outputs = model(**inputs)
        ...     report = probe.get_report()

    Or manually attached/detached:

        >>> probe = ModelProbe(model, config)
        >>> probe.attach()
        >>> outputs = model(**inputs)
        >>> report = probe.get_report()
        >>> probe.detach()

    Args:
        model: The PyTorch model to instrument
        config: DiagnosticsConfig controlling what to track
        model_name: Optional name for the model in reports
    """

    def __init__(
        self,
        model: nn.Module,
        config: DiagnosticsConfig | None = None,
        model_name: str | None = None,
    ):
        self.model = model
        self.config = config or DiagnosticsConfig()
        self.model_name = model_name or self._infer_model_name(model)

        # State
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._attached = False
        self._num_forward_passes = 0
        self._total_forward_time = 0.0
        self._lock = threading.Lock()

        # Analyzers
        self._memory_analyzer: MemoryAnalyzer | None = None
        self._timing_analyzer: TimingAnalyzer | None = None
        self._activation_analyzer: ActivationAnalyzer | None = None

        # Layer tracking
        self._instrumented_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self._layer_diagnostics: OrderedDict[str, LayerDiagnostics] = OrderedDict()

        # Hook state storage (for pre-hook to post-hook communication)
        self._hook_state: dict[str, dict[str, Any]] = {}

        # Initialize analyzers based on config
        self._init_analyzers()

    def _infer_model_name(self, model: nn.Module) -> str:
        """Try to infer model name from class or config."""
        # Try to get name from model config (transformers models)
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            return model.config._name_or_path
        if hasattr(model, "name_or_path"):
            return model.name_or_path
        # Fall back to class name
        return model.__class__.__name__

    def _init_analyzers(self) -> None:
        """Initialize analyzers based on config."""
        if self.config.track_memory:
            self._memory_analyzer = MemoryAnalyzer()

        if self.config.track_timing:
            self._timing_analyzer = TimingAnalyzer(sync_cuda=self.config.sync_cuda)

        if self.config.track_activations:
            self._activation_analyzer = ActivationAnalyzer(
                stats=self.config.get_activation_stats(),
                max_elements=self.config.max_activation_elements,
                include_inputs=self.config.include_input_stats,
            )

    def attach(self) -> "ModelProbe":
        """
        Attach hooks to the model.

        Returns:
            self for method chaining
        """
        if self._attached:
            return self

        if not self.config.enabled:
            return self

        # Find and instrument layers
        for name, module in self.model.named_modules():
            if self.config.should_instrument_layer(name):
                self._instrument_layer(name, module)

        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attached = False

    def _instrument_layer(self, name: str, module: nn.Module) -> None:
        """Attach hooks to a single layer."""
        self._instrumented_layers[name] = module
        self._layer_diagnostics[name] = LayerDiagnostics(name=name)

        # Create closure to capture layer name
        def make_pre_hook(layer_name: str):
            def pre_hook(module: nn.Module, inputs: tuple) -> None:
                self._pre_forward(layer_name, module, inputs)

            return pre_hook

        def make_post_hook(layer_name: str):
            def post_hook(module: nn.Module, inputs: tuple, outputs: Any) -> None:
                self._post_forward(layer_name, module, inputs, outputs)

            return post_hook

        # Register hooks
        pre_handle = module.register_forward_pre_hook(make_pre_hook(name))
        post_handle = module.register_forward_hook(make_post_hook(name))

        self._hooks.extend([pre_handle, post_handle])

    def _pre_forward(self, layer_name: str, module: nn.Module, inputs: tuple) -> None:
        """Pre-forward hook: capture state before layer execution."""
        with self._lock:
            state = {}

            if self._memory_analyzer:
                state["memory"] = self._memory_analyzer.before_forward(module, layer_name, inputs)

            if self._timing_analyzer:
                state["timing"] = self._timing_analyzer.before_forward(module, layer_name, inputs)

            if self._activation_analyzer:
                state["activations"] = self._activation_analyzer.before_forward(module, layer_name, inputs)

            self._hook_state[layer_name] = state

    def _post_forward(self, layer_name: str, module: nn.Module, inputs: tuple, outputs: Any) -> None:
        """Post-forward hook: collect metrics after layer execution."""
        with self._lock:
            state = self._hook_state.get(layer_name, {})
            diag = self._layer_diagnostics[layer_name]

            if self._memory_analyzer:
                diag.memory = self._memory_analyzer.after_forward(
                    module, layer_name, inputs, outputs, state.get("memory", {})
                )

            if self._timing_analyzer:
                diag.timing = self._timing_analyzer.after_forward(
                    module, layer_name, inputs, outputs, state.get("timing", {})
                )

            if self._activation_analyzer:
                diag.activations = self._activation_analyzer.after_forward(
                    module, layer_name, inputs, outputs, state.get("activations", {})
                )
                if self.config.include_input_stats:
                    diag.input_activations = self._activation_analyzer.get_input_stats(layer_name)

            # Clean up state
            self._hook_state.pop(layer_name, None)

    def get_report(self) -> DiagnosticsReport:
        """
        Generate a diagnostics report from collected metrics.

        Returns:
            DiagnosticsReport with all collected metrics
        """
        report = DiagnosticsReport(
            layers=self._layer_diagnostics.copy(),
            model_name=self.model_name,
            num_forward_passes=max(self._num_forward_passes, 1),
        )

        # Aggregate timing
        if self._timing_analyzer:
            report.total_time = self._timing_analyzer.get_total_time()

        # Aggregate memory
        if self._memory_analyzer:
            report.total_memory = self._memory_analyzer.get_total_stats()

        # Add metadata
        report.metadata = {
            "num_layers_instrumented": len(self._instrumented_layers),
            "config": {
                "track_memory": self.config.track_memory,
                "track_timing": self.config.track_timing,
                "track_activations": self.config.track_activations,
                "sync_cuda": self.config.sync_cuda,
            },
        }

        return report

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            if self._memory_analyzer:
                self._memory_analyzer.reset()
            if self._timing_analyzer:
                self._timing_analyzer.reset()
            if self._activation_analyzer:
                self._activation_analyzer.reset()

            # Reset layer diagnostics
            for name in self._layer_diagnostics:
                self._layer_diagnostics[name] = LayerDiagnostics(name=name)

            self._num_forward_passes = 0
            self._total_forward_time = 0.0
            self._hook_state.clear()

    def record_forward_pass(self) -> None:
        """Increment forward pass counter (call after model forward)."""
        self._num_forward_passes += 1

    @property
    def is_attached(self) -> bool:
        """Whether hooks are currently attached."""
        return self._attached

    @property
    def num_instrumented_layers(self) -> int:
        """Number of layers being instrumented."""
        return len(self._instrumented_layers)

    def get_instrumented_layers(self) -> list[str]:
        """Get list of instrumented layer names."""
        return list(self._instrumented_layers.keys())

    def __enter__(self) -> "ModelProbe":
        """Context manager entry: attach hooks."""
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: detach hooks."""
        self.detach()

    def __repr__(self) -> str:
        status = "attached" if self._attached else "detached"
        return f"ModelProbe(model={self.model_name!r}, layers={len(self._instrumented_layers)}, status={status})"


@contextmanager
def probe_model(model: nn.Module, config: DiagnosticsConfig | None = None, **kwargs):
    """
    Context manager for probing a model.

    Convenience function that creates a ModelProbe, yields it,
    and automatically cleans up.

    Args:
        model: The model to probe
        config: Optional DiagnosticsConfig
        **kwargs: Additional arguments to ModelProbe

    Yields:
        ModelProbe instance

    Example:
        >>> with probe_model(model, DiagnosticsConfig.timing_only()) as probe:
        ...     output = model(input)
        ...     report = probe.get_report()
        ...     print(report.summary())
    """
    probe = ModelProbe(model, config, **kwargs)
    try:
        probe.attach()
        yield probe
    finally:
        probe.detach()


def diagnose_forward(
    model: nn.Module,
    inputs: dict[str, Any] | tuple[Any, ...],
    config: DiagnosticsConfig | None = None,
    num_runs: int = 1,
    warmup_runs: int = 0,
) -> DiagnosticsReport:
    """
    Run diagnostics on a single forward pass (or multiple runs).

    Convenience function for quick diagnostics without manual probe management.

    Args:
        model: The model to diagnose
        inputs: Input to the model (dict for **kwargs, tuple for *args)
        config: Optional DiagnosticsConfig
        num_runs: Number of forward passes to run
        warmup_runs: Number of warmup runs before collecting metrics

    Returns:
        DiagnosticsReport with collected metrics

    Example:
        >>> report = diagnose_forward(
        ...     model,
        ...     {"input_ids": input_ids, "attention_mask": attention_mask},
        ...     num_runs=5,
        ...     warmup_runs=2,
        ... )
        >>> print(report.summary())
    """
    config = config or DiagnosticsConfig()

    with probe_model(model, config) as probe:
        # Warmup runs
        for _ in range(warmup_runs):
            if isinstance(inputs, dict):
                with torch.no_grad():
                    model(**inputs)
            else:
                with torch.no_grad():
                    model(*inputs)

        # Reset after warmup
        probe.reset()

        # Measurement runs
        for _ in range(num_runs):
            if isinstance(inputs, dict):
                with torch.no_grad():
                    model(**inputs)
            else:
                with torch.no_grad():
                    model(*inputs)
            probe.record_forward_pass()

        return probe.get_report()
