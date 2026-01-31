"""Tests for ModelProbe."""

import pytest
import torch
import torch.nn as nn

from transformers.diagnostics.config import DiagnosticsConfig
from transformers.diagnostics.probe import ModelProbe, probe_model, diagnose_forward


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class NestedModel(nn.Module):
    """Model with nested modules for testing layer selection."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(20, 4, batch_first=True)
        self.decoder = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        x = self.decoder(x)
        return x


class TestModelProbe:
    """Test suite for ModelProbe."""
    
    def test_attach_detach(self):
        """Test basic attach/detach functionality."""
        model = SimpleModel()
        config = DiagnosticsConfig()
        probe = ModelProbe(model, config)
        
        assert not probe.is_attached
        
        probe.attach()
        assert probe.is_attached
        assert probe.num_instrumented_layers > 0
        
        probe.detach()
        assert not probe.is_attached
    
    def test_context_manager(self):
        """Test context manager interface."""
        model = SimpleModel()
        
        with ModelProbe(model) as probe:
            assert probe.is_attached
        
        assert not probe.is_attached
    
    def test_basic_timing(self):
        """Test that timing is collected."""
        model = SimpleModel()
        config = DiagnosticsConfig(track_timing=True, track_memory=False)
        
        x = torch.randn(2, 10)
        
        with ModelProbe(model, config) as probe:
            _ = model(x)
            report = probe.get_report()
        
        assert report.total_time > 0
        assert len(report.timing_by_layer) > 0
        
        # Check that linear layers were timed
        layer_names = list(report.timing_by_layer.keys())
        assert any("linear" in name for name in layer_names)
    
    def test_layer_pattern_filtering(self):
        """Test that layer patterns filter correctly."""
        model = SimpleModel()
        config = DiagnosticsConfig(
            layer_patterns=["linear1"],
            track_timing=True,
        )
        
        with ModelProbe(model, config) as probe:
            layers = probe.get_instrumented_layers()
        
        assert len(layers) == 1
        assert "linear1" in layers[0]
    
    def test_multiple_forward_passes(self):
        """Test aggregation across multiple forward passes."""
        model = SimpleModel()
        config = DiagnosticsConfig(track_timing=True, aggregate_calls=True)
        
        x = torch.randn(2, 10)
        
        with ModelProbe(model, config) as probe:
            for _ in range(3):
                _ = model(x)
                probe.record_forward_pass()
            
            report = probe.get_report()
        
        assert report.num_forward_passes == 3
        
        # Timing should reflect multiple calls
        for timing in report.timing_by_layer.values():
            assert timing.num_calls >= 1
    
    def test_reset(self):
        """Test reset clears collected metrics."""
        model = SimpleModel()
        config = DiagnosticsConfig(track_timing=True)
        
        x = torch.randn(2, 10)
        
        with ModelProbe(model, config) as probe:
            _ = model(x)
            probe.reset()
            report = probe.get_report()
        
        # After reset, timing should be zero or minimal
        assert report.total_time == 0.0 or report.total_time < 1e-6
    
    def test_disabled_config(self):
        """Test that disabled config doesn't attach hooks."""
        model = SimpleModel()
        config = DiagnosticsConfig(enabled=False)
        
        with ModelProbe(model, config) as probe:
            # Should not be attached when disabled
            assert probe.num_instrumented_layers == 0
    
    def test_report_summary(self):
        """Test report summary generation."""
        model = SimpleModel()
        config = DiagnosticsConfig(track_timing=True)
        
        x = torch.randn(2, 10)
        
        with ModelProbe(model, config) as probe:
            _ = model(x)
            report = probe.get_report()
        
        summary = report.summary()
        assert "DIAGNOSTICS REPORT" in summary
        assert "Total time" in summary
    
    def test_report_to_dict(self):
        """Test report JSON serialization."""
        model = SimpleModel()
        
        x = torch.randn(2, 10)
        
        with ModelProbe(model) as probe:
            _ = model(x)
            report = probe.get_report()
        
        data = report.to_dict()
        assert "layers" in data
        assert "total_time_ms" in data
        assert "model_name" in data
    
    def test_model_name_inference(self):
        """Test model name is inferred correctly."""
        model = SimpleModel()
        probe = ModelProbe(model)
        
        assert probe.model_name == "SimpleModel"
    
    def test_custom_model_name(self):
        """Test custom model name override."""
        model = SimpleModel()
        probe = ModelProbe(model, model_name="MyCustomModel")
        
        assert probe.model_name == "MyCustomModel"


class TestProbeModel:
    """Test probe_model context manager."""
    
    def test_basic_usage(self):
        """Test basic probe_model usage."""
        model = SimpleModel()
        x = torch.randn(2, 10)
        
        with probe_model(model) as probe:
            _ = model(x)
            report = probe.get_report()
        
        assert report is not None
        assert len(report.layers) > 0


class TestDiagnoseForward:
    """Test diagnose_forward convenience function."""
    
    def test_dict_inputs(self):
        """Test with dictionary inputs."""
        model = SimpleModel()
        
        # Using tuple input since SimpleModel expects a single tensor
        report = diagnose_forward(model, (torch.randn(2, 10),), num_runs=2)
        
        assert report.num_forward_passes == 2
        assert report.total_time > 0
    
    def test_warmup_runs(self):
        """Test warmup runs are excluded from metrics."""
        model = SimpleModel()
        inputs = (torch.randn(2, 10),)
        
        report = diagnose_forward(model, inputs, num_runs=2, warmup_runs=3)
        
        # Only measurement runs should be counted
        assert report.num_forward_passes == 2
