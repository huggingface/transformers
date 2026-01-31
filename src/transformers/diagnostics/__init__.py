"""
Unified Model Introspection & Diagnostics Framework for HuggingFace Transformers.

This module provides tools for profiling and analyzing transformer models:
- Per-layer memory profiling
- Per-layer timing analysis
- Activation statistics collection
- Model comparison utilities

Basic Usage:
    >>> from transformers import AutoModel
    >>> from transformers.diagnostics import DiagnosticsConfig, ModelProbe

    >>> model = AutoModel.from_pretrained("bert-base-uncased")
    >>> config = DiagnosticsConfig(track_memory=True, track_timing=True)

    >>> with ModelProbe(model, config) as probe:
    ...     outputs = model(**inputs)
    ...     report = probe.get_report()

    >>> print(report.summary())

Quick Diagnostics:
    >>> from transformers.diagnostics import diagnose_forward

    >>> report = diagnose_forward(model, inputs, num_runs=5, warmup_runs=2)
    >>> print(report.summary())

Model Comparison:
    >>> from transformers.diagnostics import compare_reports

    >>> report_a = diagnose_forward(model_a, inputs)
    >>> report_b = diagnose_forward(model_b, inputs)
    >>> comparison = compare_reports(report_a, report_b)
    >>> print(comparison.summary())
"""

from .config import DiagnosticsConfig
from .probe import ModelProbe, diagnose_forward, probe_model
from .report import (
    ActivationStats,
    DiagnosticsReport,
    LayerDiagnostics,
    MemoryStats,
    TimingStats,
)
from .utils.comparison import ReportComparison, compare_reports


__all__ = [
    # Main classes
    "DiagnosticsConfig",
    "ModelProbe",
    "DiagnosticsReport",
    # Convenience functions
    "probe_model",
    "diagnose_forward",
    "compare_reports",
    # Data classes
    "LayerDiagnostics",
    "MemoryStats",
    "TimingStats",
    "ActivationStats",
    "ReportComparison",
]

__version__ = "0.1.0"
