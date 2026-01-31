"""Tests for DiagnosticsConfig."""

import pytest

from transformers.diagnostics.config import DiagnosticsConfig


class TestDiagnosticsConfig:
    """Test suite for DiagnosticsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DiagnosticsConfig()

        assert config.track_memory is True
        assert config.track_timing is True
        assert config.track_activations is False
        assert config.enabled is True
        assert config.sync_cuda is True
        assert config.aggregate_calls is True

    def test_layer_pattern_matching(self):
        """Test layer name pattern matching."""
        config = DiagnosticsConfig(layer_patterns=[".*attention.*", ".*mlp.*"])

        # Should match
        assert config.should_instrument_layer("encoder.layer.0.attention.self")
        assert config.should_instrument_layer("decoder.mlp.dense")

        # Should not match
        assert not config.should_instrument_layer("encoder.layer.0.output")
        assert not config.should_instrument_layer("embeddings")

    def test_exclude_patterns(self):
        """Test exclude patterns take precedence."""
        config = DiagnosticsConfig(
            layer_patterns=[".*"],  # Match everything
            exclude_patterns=[".*dropout.*"],  # Except dropout
        )

        assert config.should_instrument_layer("encoder.attention")
        assert not config.should_instrument_layer("encoder.attention.dropout")

    def test_empty_patterns_matches_all(self):
        """Test that empty patterns match all layers."""
        config = DiagnosticsConfig(layer_patterns=[])

        assert config.should_instrument_layer("any.layer.name")
        assert config.should_instrument_layer("another.layer")

    def test_invalid_activation_stats_raises(self):
        """Test that invalid activation stats raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DiagnosticsConfig(activation_stats=["invalid_stat"])

        assert "Invalid activation stats" in str(exc_info.value)

    def test_valid_activation_stats(self):
        """Test valid activation statistics."""
        config = DiagnosticsConfig(activation_stats=["mean", "std", "norm", "sparsity"])

        stats = config.get_activation_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "norm" in stats
        assert "sparsity" in stats

    def test_factory_methods(self):
        """Test factory class methods."""
        # Memory only
        mem_config = DiagnosticsConfig.memory_only()
        assert mem_config.track_memory is True
        assert mem_config.track_timing is False
        assert mem_config.track_activations is False

        # Timing only
        time_config = DiagnosticsConfig.timing_only()
        assert time_config.track_memory is False
        assert time_config.track_timing is True
        assert time_config.track_activations is False

        # Full
        full_config = DiagnosticsConfig.full()
        assert full_config.track_memory is True
        assert full_config.track_timing is True
        assert full_config.track_activations is True

        # Minimal
        min_config = DiagnosticsConfig.minimal()
        assert min_config.sync_cuda is False

    def test_config_repr(self):
        """Test string representation."""
        config = DiagnosticsConfig()
        repr_str = repr(config)

        assert "DiagnosticsConfig" in repr_str
        assert "track_memory=True" in repr_str
