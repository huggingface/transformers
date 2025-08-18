#!/usr/bin/env python3
"""
Test script for the benchmarking framework v2.
Tests core functionality without requiring heavy ML dependencies.
"""

import json
import logging
import os
import sys
import tempfile
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

def test_framework_imports():
    """Test that core framework modules can be imported."""
    print("Testing framework imports...")
    try:
        from framework import (
            BenchmarkConfig, TimingResult, BenchmarkStatistics,
            HardwareInfo, BenchmarkMetadata, GPUMonitor,
            ModelBenchmark, BenchmarkRunner, create_config_variants,
            get_hardware_info, get_git_commit_id, flush_memory
        )
        print("✓ All framework imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_benchmark_config():
    """Test BenchmarkConfig creation and validation."""
    print("Testing BenchmarkConfig...")
    try:
        from framework import BenchmarkConfig
        
        config = BenchmarkConfig(
            name="test_config",
            model_id="test-model",
            variant="eager",
            warmup_iterations=2,
            measurement_iterations=3
        )
        
        assert config.name == "test_config"
        assert config.variant == "eager"
        assert config.generation_config["do_sample"] is False  # Default value
        print("✓ BenchmarkConfig creation and defaults work")
        return True
    except Exception as e:
        print(f"✗ BenchmarkConfig test failed: {e}")
        return False


def test_timing_result():
    """Test TimingResult creation."""
    print("Testing TimingResult...")
    try:
        from framework import TimingResult
        
        result = TimingResult(
            latency=1.5,
            tokens_per_second=50.0,
            total_tokens_generated=75,
            metadata={"test": "value"}
        )
        
        assert result.latency == 1.5
        assert result.tokens_per_second == 50.0
        print("✓ TimingResult creation works")
        return True
    except Exception as e:
        print(f"✗ TimingResult test failed: {e}")
        return False


def test_benchmark_statistics():
    """Test BenchmarkStatistics calculation."""
    print("Testing BenchmarkStatistics...")
    try:
        from framework import BenchmarkStatistics
        
        measurements = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        stats = BenchmarkStatistics.from_measurements("test_metric", measurements)
        
        assert stats.name == "test_metric"
        assert abs(stats.mean - 1.5) < 0.01  # Should be close to 1.5
        assert stats.min == 1.0
        assert stats.max == 2.0
        assert stats.p95 > stats.median  # 95th percentile should be higher than median
        print("✓ BenchmarkStatistics calculation works")
        return True
    except Exception as e:
        print(f"✗ BenchmarkStatistics test failed: {e}")
        return False


def test_config_variants():
    """Test configuration variant generation."""
    print("Testing config variants...")
    try:
        from framework import BenchmarkConfig, create_config_variants
        
        base_config = BenchmarkConfig(
            name="base",
            model_id="test-model"
        )
        
        variants = create_config_variants(base_config, {
            "variant": ["eager", "compiled"],
            "batch_size": [1, 2]
        })
        
        assert len(variants) == 4  # 2 x 2 combinations
        
        # Check that names are generated correctly
        names = [v.name for v in variants]
        assert "base_variant=eager_batch_size=1" in names
        assert "base_variant=compiled_batch_size=2" in names
        
        print("✓ Config variants generation works")
        return True
    except Exception as e:
        print(f"✗ Config variants test failed: {e}")
        return False


def test_hardware_info():
    """Test hardware information collection."""
    print("Testing hardware info collection...")
    try:
        from framework import get_hardware_info
        
        hw_info = get_hardware_info()
        
        assert hasattr(hw_info, 'cpu_count')
        assert hasattr(hw_info, 'memory_total')
        assert hasattr(hw_info, 'gpu_name')
        assert hw_info.cpu_count > 0
        assert hw_info.memory_total > 0
        
        print("✓ Hardware info collection works")
        return True
    except Exception as e:
        print(f"✗ Hardware info test failed: {e}")
        return False


def test_benchmark_discovery():
    """Test benchmark discovery functionality."""
    print("Testing benchmark discovery...")
    try:
        from run_benchmarks import discover_benchmarks
        
        benchmarks = discover_benchmarks('./benches')
        
        assert len(benchmarks) > 0, "Should find at least one benchmark"
        
        llama_benchmark = None
        for b in benchmarks:
            if b['name'] == 'llama_benchmark':
                llama_benchmark = b
                break
        
        assert llama_benchmark is not None, "Should find llama_benchmark"
        assert callable(llama_benchmark['runner_function'])
        
        print("✓ Benchmark discovery works")
        return True
    except Exception as e:
        print(f"✗ Benchmark discovery test failed: {e}")
        return False


def test_json_output_structure():
    """Test that we can create the expected JSON output structure."""
    print("Testing JSON output structure...")
    try:
        from framework import BenchmarkConfig, BenchmarkStatistics, HardwareInfo, BenchmarkMetadata
        from dataclasses import asdict
        
        # Create mock data similar to what would be generated
        config = BenchmarkConfig(name="test", model_id="test-model")
        hw_info = HardwareInfo(
            gpu_name="Test GPU",
            gpu_memory_total=8192,
            cpu_count=8,
            memory_total=16384,
            python_version="3.8.0"
        )
        metadata = BenchmarkMetadata(
            timestamp=datetime.utcnow().isoformat(),
            commit_id="test123",
            repository="test-repo",
            branch="main",
            hardware_info=hw_info,
            config=config
        )
        
        # Create mock measurements
        latency_stats = BenchmarkStatistics.from_measurements(
            "latency", [1.0, 1.1, 1.2, 1.3, 1.4]
        )
        
        # Create the output structure
        output_data = {
            "model_name": "test_model",
            "benchmark_scenarios": [{
                "scenario_name": "test_scenario",
                "metadata": asdict(metadata),
                "measurements": {
                    "latency": asdict(latency_stats)
                },
                "gpu_metrics": {
                    "gpu_utilization_mean": 85.0
                }
            }]
        }
        
        # Test JSON serialization
        json_str = json.dumps(output_data, indent=2, default=str)
        parsed = json.loads(json_str)
        
        assert parsed["model_name"] == "test_model"
        assert len(parsed["benchmark_scenarios"]) == 1
        assert "latency" in parsed["benchmark_scenarios"][0]["measurements"]
        
        print("✓ JSON output structure works")
        return True
    except Exception as e:
        print(f"✗ JSON output structure test failed: {e}")
        return False


def test_benchmark_runner_creation():
    """Test BenchmarkRunner creation without actual benchmarking."""
    print("Testing BenchmarkRunner creation...")
    try:
        from framework import BenchmarkRunner
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = logging.getLogger("test")
            runner = BenchmarkRunner(logger, temp_dir)
            
            assert runner.output_dir == temp_dir
            assert os.path.exists(temp_dir)
            
        print("✓ BenchmarkRunner creation works")
        return True
    except Exception as e:
        print(f"✗ BenchmarkRunner creation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BENCHMARKING FRAMEWORK V2 - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_framework_imports,
        test_benchmark_config,
        test_timing_result,
        test_benchmark_statistics,
        test_config_variants,
        test_hardware_info,
        test_benchmark_discovery,
        test_json_output_structure,
        test_benchmark_runner_creation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Framework is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 