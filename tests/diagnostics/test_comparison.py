"""Tests for comparison utilities."""

import pytest

from transformers.diagnostics.report import (
    DiagnosticsReport,
    LayerDiagnostics,
    TimingStats,
    MemoryStats,
)
from transformers.diagnostics.utils.comparison import compare_reports, ReportComparison


class TestReportComparison:
    """Test ReportComparison class."""
    
    def setup_method(self):
        """Set up test reports."""
        # Baseline report
        self.report_a = DiagnosticsReport(
            model_name="model_a",
            total_time=1.0,
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.5, num_calls=1),
                    memory=MemoryStats(allocated_delta=10 * 1024 * 1024),
                ),
                "layer2": LayerDiagnostics(
                    name="layer2",
                    timing=TimingStats(total_time=0.3, num_calls=1),
                    memory=MemoryStats(allocated_delta=5 * 1024 * 1024),
                ),
            },
        )
        
        # Faster/smaller report
        self.report_b = DiagnosticsReport(
            model_name="model_b",
            total_time=0.6,
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.3, num_calls=1),  # Faster
                    memory=MemoryStats(allocated_delta=8 * 1024 * 1024),  # Smaller
                ),
                "layer2": LayerDiagnostics(
                    name="layer2",
                    timing=TimingStats(total_time=0.2, num_calls=1),  # Faster
                    memory=MemoryStats(allocated_delta=4 * 1024 * 1024),  # Smaller
                ),
            },
        )
    
    def test_basic_comparison(self):
        """Test basic comparison creation."""
        comparison = compare_reports(
            self.report_a, 
            self.report_b,
            name_a="baseline",
            name_b="optimized",
        )
        
        assert comparison.name_a == "baseline"
        assert comparison.name_b == "optimized"
        assert len(comparison.layer_comparisons) == 2
    
    def test_time_metrics(self):
        """Test timing comparison metrics."""
        comparison = compare_reports(self.report_a, self.report_b)
        
        # Report B is faster
        assert comparison.total_time_diff < 0
        assert comparison.total_time_ratio < 1.0
        assert comparison.total_time_pct_change < 0
    
    def test_layer_comparison(self):
        """Test individual layer comparisons."""
        comparison = compare_reports(self.report_a, self.report_b)
        
        layer1 = comparison.layer_comparisons["layer1"]
        
        # layer1 is faster in B
        assert layer1.time_a > layer1.time_b
        assert layer1.time_diff < 0
        assert layer1.time_ratio < 1.0
        assert layer1.time_pct_change < 0
    
    def test_get_speedups(self):
        """Test speedup detection."""
        comparison = compare_reports(self.report_a, self.report_b)
        
        speedups = comparison.get_speedups(min_speedup=1.1)
        
        assert len(speedups) > 0
        # All speedups should be > 1.1x
        for name, ratio in speedups:
            assert ratio >= 1.1
    
    def test_get_slowdowns(self):
        """Test slowdown detection."""
        # Create a report where B is slower
        slow_report = DiagnosticsReport(
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=1.0, num_calls=1),  # Slower
                ),
            },
        )
        
        comparison = compare_reports(self.report_a, slow_report)
        slowdowns = comparison.get_slowdowns(min_slowdown=1.1)
        
        assert len(slowdowns) > 0
    
    def test_summary(self):
        """Test summary generation."""
        comparison = compare_reports(
            self.report_a, 
            self.report_b,
            name_a="FP32",
            name_b="INT8",
        )
        
        summary = comparison.summary()
        
        assert "FP32 vs INT8" in summary
        assert "OVERALL METRICS" in summary
        assert "Total Time" in summary
    
    def test_to_dataframe(self):
        """Test DataFrame export."""
        pytest.importorskip("pandas")
        
        comparison = compare_reports(self.report_a, self.report_b)
        df = comparison.to_dataframe()
        
        assert len(df) == 2
        assert "layer" in df.columns
        assert "time_diff_ms" in df.columns
    
    def test_missing_layers(self):
        """Test comparison when reports have different layers."""
        report_with_extra = DiagnosticsReport(
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.5),
                ),
                "layer3": LayerDiagnostics(  # Extra layer
                    name="layer3",
                    timing=TimingStats(total_time=0.2),
                ),
            },
        )
        
        comparison = compare_reports(self.report_a, report_with_extra)
        
        # Should include all layers from both
        assert "layer1" in comparison.layer_comparisons
        assert "layer2" in comparison.layer_comparisons
        assert "layer3" in comparison.layer_comparisons
        
        # layer3 should only have B data
        layer3 = comparison.layer_comparisons["layer3"]
        assert layer3.time_a is None
        assert layer3.time_b is not None


class TestCompareReports:
    """Test compare_reports function."""
    
    def test_function_creates_comparison(self):
        """Test function creates ReportComparison instance."""
        report_a = DiagnosticsReport()
        report_b = DiagnosticsReport()
        
        result = compare_reports(report_a, report_b)
        
        assert isinstance(result, ReportComparison)
    
    def test_custom_names(self):
        """Test custom report names."""
        report_a = DiagnosticsReport()
        report_b = DiagnosticsReport()
        
        result = compare_reports(
            report_a, report_b,
            name_a="before",
            name_b="after",
        )
        
        assert result.name_a == "before"
        assert result.name_b == "after"
