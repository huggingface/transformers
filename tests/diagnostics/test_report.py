"""Tests for DiagnosticsReport and related dataclasses."""

import pytest
import json
import tempfile
from pathlib import Path

from transformers.diagnostics.report import (
    MemoryStats,
    TimingStats,
    ActivationStats,
    LayerDiagnostics,
    DiagnosticsReport,
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""
    
    def test_allocated_mb(self):
        """Test MB conversion."""
        stats = MemoryStats(allocated_delta=10 * 1024 * 1024)  # 10 MB
        assert stats.allocated_mb == 10.0
    
    def test_peak_mb(self):
        """Test peak MB conversion."""
        stats = MemoryStats(peak_delta=20 * 1024 * 1024)  # 20 MB
        assert stats.peak_mb == 20.0
    
    def test_repr(self):
        """Test string representation."""
        stats = MemoryStats(
            allocated_delta=1024 * 1024,
            peak_delta=2 * 1024 * 1024,
            is_cuda=True,
            device_index=0,
        )
        repr_str = repr(stats)
        assert "cuda:0" in repr_str
        assert "alloc=" in repr_str


class TestTimingStats:
    """Test TimingStats dataclass."""
    
    def test_update(self):
        """Test timing update with Welford's algorithm."""
        stats = TimingStats()
        
        stats.update(0.1)
        stats.update(0.2)
        stats.update(0.3)
        
        assert stats.num_calls == 3
        assert abs(stats.total_time - 0.6) < 1e-6
        assert stats.min_time == 0.1
        assert stats.max_time == 0.3
    
    def test_mean_time(self):
        """Test mean time calculation."""
        stats = TimingStats()
        stats.update(0.1)
        stats.update(0.3)
        
        assert abs(stats.mean_time - 0.2) < 1e-6
    
    def test_std_time(self):
        """Test standard deviation calculation."""
        stats = TimingStats()
        # Add same value multiple times for zero std
        stats.update(0.1)
        stats.update(0.1)
        stats.update(0.1)
        
        assert stats.std_time < 1e-6
    
    def test_mean_time_ms(self):
        """Test millisecond conversion."""
        stats = TimingStats(total_time=1.0, num_calls=10)
        assert stats.mean_time_ms == 100.0


class TestActivationStats:
    """Test ActivationStats dataclass."""
    
    def test_basic_stats(self):
        """Test basic activation stats."""
        stats = ActivationStats(
            mean=0.5,
            std=0.1,
            min_val=-1.0,
            max_val=1.0,
            shape=(2, 10, 768),
        )
        
        assert stats.mean == 0.5
        assert stats.std == 0.1
        assert stats.shape == (2, 10, 768)
    
    def test_repr(self):
        """Test string representation."""
        stats = ActivationStats(mean=0.5, std=0.1, shape=(2, 10))
        repr_str = repr(stats)
        
        assert "mean=" in repr_str
        assert "std=" in repr_str
        assert "shape=" in repr_str


class TestLayerDiagnostics:
    """Test LayerDiagnostics dataclass."""
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        diag = LayerDiagnostics(
            name="encoder.layer.0",
            timing=TimingStats(total_time=0.1, num_calls=1),
            memory=MemoryStats(allocated_delta=1024),
        )
        
        data = diag.to_dict()
        
        assert data["name"] == "encoder.layer.0"
        assert "timing" in data
        assert "memory" in data


class TestDiagnosticsReport:
    """Test DiagnosticsReport dataclass."""
    
    def test_layer_names(self):
        """Test layer_names property."""
        report = DiagnosticsReport(
            layers={
                "layer1": LayerDiagnostics(name="layer1"),
                "layer2": LayerDiagnostics(name="layer2"),
            }
        )
        
        assert report.layer_names == ["layer1", "layer2"]
    
    def test_timing_by_layer(self):
        """Test timing_by_layer property."""
        report = DiagnosticsReport(
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.1)
                ),
                "layer2": LayerDiagnostics(name="layer2"),  # No timing
            }
        )
        
        timing = report.timing_by_layer
        assert len(timing) == 1
        assert "layer1" in timing
    
    def test_get_slowest_layers(self):
        """Test get_slowest_layers method."""
        report = DiagnosticsReport(
            layers={
                "fast": LayerDiagnostics(
                    name="fast",
                    timing=TimingStats(total_time=0.01)
                ),
                "slow": LayerDiagnostics(
                    name="slow",
                    timing=TimingStats(total_time=0.5)
                ),
                "medium": LayerDiagnostics(
                    name="medium",
                    timing=TimingStats(total_time=0.1)
                ),
            },
            total_time=0.61,
        )
        
        slowest = report.get_slowest_layers(n=2)
        assert len(slowest) == 2
        assert slowest[0][0] == "slow"
        assert slowest[1][0] == "medium"
    
    def test_to_json(self):
        """Test JSON serialization."""
        report = DiagnosticsReport(
            model_name="test-model",
            total_time=0.1,
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.1)
                ),
            },
        )
        
        json_str = report.to_json()
        data = json.loads(json_str)
        
        assert data["model_name"] == "test-model"
        assert "layers" in data
    
    def test_save_load(self):
        """Test save and load functionality."""
        report = DiagnosticsReport(
            model_name="test-model",
            num_forward_passes=5,
            total_time=0.5,
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.5, num_calls=5),
                    memory=MemoryStats(allocated_delta=1024),
                ),
            },
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(str(path))
            
            loaded = DiagnosticsReport.load(str(path))
        
        assert loaded.model_name == "test-model"
        assert loaded.num_forward_passes == 5
        assert "layer1" in loaded.layers
    
    def test_summary(self):
        """Test summary generation."""
        report = DiagnosticsReport(
            model_name="test-model",
            total_time=1.0,
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.5),
                    memory=MemoryStats(allocated_delta=10 * 1024 * 1024),
                ),
            },
        )
        
        summary = report.summary()
        
        assert "test-model" in summary
        assert "Total time" in summary
        assert "SLOWEST LAYERS" in summary


class TestDiagnosticsReportDataFrame:
    """Test DataFrame conversion (requires pandas)."""
    
    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        pytest.importorskip("pandas")
        
        report = DiagnosticsReport(
            total_time=1.0,
            layers={
                "layer1": LayerDiagnostics(
                    name="layer1",
                    timing=TimingStats(total_time=0.5, num_calls=1),
                ),
                "layer2": LayerDiagnostics(
                    name="layer2",
                    timing=TimingStats(total_time=0.3, num_calls=1),
                ),
            },
        )
        
        df = report.to_dataframe()
        
        assert len(df) == 2
        assert "layer_name" in df.columns
        assert "time_ms" in df.columns
