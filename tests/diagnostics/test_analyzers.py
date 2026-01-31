"""Tests for diagnostic analyzers."""

import pytest
import torch
import torch.nn as nn

from transformers.diagnostics.analyzers.activations import ActivationAnalyzer, compute_activation_stats
from transformers.diagnostics.analyzers.memory import MemoryAnalyzer
from transformers.diagnostics.analyzers.timing import CudaEventTimer, TimingAnalyzer


class TestMemoryAnalyzer:
    """Test MemoryAnalyzer."""

    def test_cpu_memory_estimation(self):
        """Test CPU memory estimation from output tensors."""
        analyzer = MemoryAnalyzer()
        module = nn.Linear(10, 20)

        state = analyzer.before_forward(module, "test_layer", (torch.randn(2, 10),))
        output = module(torch.randn(2, 10))
        stats = analyzer.after_forward(module, "test_layer", (torch.randn(2, 10),), output, state)

        # Should estimate memory based on output tensor
        assert stats.allocated_delta > 0
        assert not stats.is_cuda

    def test_reset(self):
        """Test reset clears accumulated stats."""
        analyzer = MemoryAnalyzer()
        module = nn.Linear(10, 20)

        state = analyzer.before_forward(module, "test", ())
        analyzer.after_forward(module, "test", (), torch.randn(2, 20), state)

        assert len(analyzer.get_all_stats()) > 0

        analyzer.reset()
        assert len(analyzer.get_all_stats()) == 0

    def test_get_total_stats(self):
        """Test aggregated stats."""
        analyzer = MemoryAnalyzer()
        module = nn.Linear(10, 20)

        for i in range(3):
            state = analyzer.before_forward(module, f"layer_{i}", ())
            analyzer.after_forward(module, f"layer_{i}", (), torch.randn(2, 20), state)

        total = analyzer.get_total_stats()
        assert total.allocated_delta > 0

    def test_disabled(self):
        """Test disabled analyzer returns empty stats."""
        analyzer = MemoryAnalyzer()
        analyzer.enabled = False
        module = nn.Linear(10, 20)

        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), torch.randn(2, 20), state)

        assert stats.allocated_delta == 0


class TestTimingAnalyzer:
    """Test TimingAnalyzer."""

    def test_basic_timing(self):
        """Test basic timing measurement."""
        analyzer = TimingAnalyzer(sync_cuda=False)
        module = nn.Linear(10, 20)

        state = analyzer.before_forward(module, "test_layer", ())
        # Simulate some work
        _ = torch.randn(100, 100) @ torch.randn(100, 100)
        stats = analyzer.after_forward(module, "test_layer", (), torch.randn(2, 20), state)

        assert stats.total_time > 0
        assert stats.num_calls == 1

    def test_multiple_calls(self):
        """Test timing accumulation across calls."""
        analyzer = TimingAnalyzer(sync_cuda=False)
        module = nn.Linear(10, 20)

        for _ in range(5):
            state = analyzer.before_forward(module, "test", ())
            analyzer.after_forward(module, "test", (), torch.randn(2, 20), state)

        stats = analyzer.get_stats("test")
        assert stats.num_calls == 5
        assert stats.mean_time > 0

    def test_get_percentage_by_layer(self):
        """Test percentage calculation."""
        analyzer = TimingAnalyzer(sync_cuda=False)
        module = nn.Linear(10, 20)

        # Two layers with different timing
        for name in ["layer_a", "layer_b"]:
            state = analyzer.before_forward(module, name, ())
            analyzer.after_forward(module, name, (), torch.randn(2, 20), state)

        percentages = analyzer.get_percentage_by_layer()
        assert len(percentages) == 2
        assert abs(sum(percentages.values()) - 100) < 1  # Should sum to ~100%

    def test_reset(self):
        """Test reset clears stats."""
        analyzer = TimingAnalyzer()
        module = nn.Linear(10, 20)

        state = analyzer.before_forward(module, "test", ())
        analyzer.after_forward(module, "test", (), torch.randn(2, 20), state)

        analyzer.reset()
        assert analyzer.get_total_time() == 0.0


class TestCudaEventTimer:
    """Test CudaEventTimer context manager."""

    def test_cpu_fallback(self):
        """Test timer works on CPU."""
        with CudaEventTimer() as timer:
            _ = torch.randn(100, 100) @ torch.randn(100, 100)

        assert timer.elapsed_ms > 0
        assert timer.elapsed_s > 0

    def test_not_stopped_raises(self):
        """Test accessing elapsed before stopping raises."""
        timer = CudaEventTimer()
        timer.__enter__()

        with pytest.raises(RuntimeError):
            _ = timer.elapsed_ms


class TestActivationAnalyzer:
    """Test ActivationAnalyzer."""

    def test_basic_stats(self):
        """Test basic activation statistics."""
        analyzer = ActivationAnalyzer(stats={"mean", "std", "min", "max"})
        module = nn.Linear(10, 20)

        output = torch.randn(2, 20)
        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), output, state)

        assert stats.mean is not None
        assert stats.std is not None
        assert stats.min_val is not None
        assert stats.max_val is not None

    def test_norm_stat(self):
        """Test norm calculation."""
        analyzer = ActivationAnalyzer(stats={"norm"})
        module = nn.Identity()

        # Known tensor with predictable norm
        output = torch.ones(4)  # norm should be 2.0
        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), output, state)

        assert abs(stats.norm - 2.0) < 1e-5

    def test_sparsity_stat(self):
        """Test sparsity calculation."""
        analyzer = ActivationAnalyzer(stats={"sparsity"})
        module = nn.Identity()

        # 50% zeros
        output = torch.tensor([0.0, 1.0, 0.0, 1.0])
        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), output, state)

        assert abs(stats.sparsity - 0.5) < 1e-5

    def test_shape_stat(self):
        """Test shape recording."""
        analyzer = ActivationAnalyzer(stats={"shape", "numel", "dtype"})
        module = nn.Identity()

        output = torch.randn(2, 10, 768)
        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), output, state)

        assert stats.shape == (2, 10, 768)
        assert stats.numel == 2 * 10 * 768
        assert "float" in stats.dtype.lower()

    def test_tuple_output(self):
        """Test handling tuple outputs."""
        analyzer = ActivationAnalyzer(stats={"mean"})
        module = nn.Identity()

        # Tuple output (like attention layers)
        output = (torch.randn(2, 10), torch.randn(2, 10))
        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), output, state)

        # Should extract first tensor
        assert stats.mean is not None

    def test_include_inputs(self):
        """Test input statistics collection."""
        analyzer = ActivationAnalyzer(stats={"mean"}, include_inputs=True)
        module = nn.Linear(10, 20)

        input_tensor = torch.randn(2, 10)
        state = analyzer.before_forward(module, "test", (input_tensor,))
        analyzer.after_forward(module, "test", (input_tensor,), torch.randn(2, 20), state)

        input_stats = analyzer.get_input_stats("test")
        assert input_stats is not None
        assert input_stats.mean is not None

    def test_max_elements_sampling(self):
        """Test sampling for large tensors."""
        analyzer = ActivationAnalyzer(stats={"mean"}, max_elements=100)
        module = nn.Identity()

        # Large tensor
        output = torch.randn(1000, 1000)
        state = analyzer.before_forward(module, "test", ())
        stats = analyzer.after_forward(module, "test", (), output, state)

        # Should still compute stats (with sampling)
        assert stats.mean is not None

    def test_invalid_stats_raises(self):
        """Test invalid stat names raise error."""
        with pytest.raises(ValueError):
            ActivationAnalyzer(stats={"invalid_stat"})


class TestComputeActivationStats:
    """Test standalone compute_activation_stats function."""

    def test_basic_usage(self):
        """Test basic usage."""
        tensor = torch.randn(10, 20)
        stats = compute_activation_stats(tensor)

        assert "mean" in stats
        assert "std" in stats
        assert "norm" in stats

    def test_custom_stats(self):
        """Test with custom stat selection."""
        tensor = torch.randn(10, 20)
        stats = compute_activation_stats(tensor, stats={"mean", "min", "max"})

        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" not in stats
