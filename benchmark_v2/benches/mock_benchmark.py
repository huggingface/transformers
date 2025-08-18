# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mock benchmark for testing and demonstration purposes.
This benchmark simulates model operations without requiring heavy ML dependencies.
It's skipped by default but can be enabled for testing the framework.
"""

import os
import sys
import time
import random
import logging
from typing import Any, Optional

# Add parent directory to path for framework imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import ModelBenchmark, BenchmarkConfig, TimingResult, flush_memory


class MockBenchmark(ModelBenchmark):
    """
    Mock benchmark that simulates model operations for testing purposes.
    
    This benchmark doesn't require any ML dependencies and generates
    realistic-looking timing data for framework testing.
    """
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.model_loaded = False
        self.base_latency = 1.0  # Base latency in seconds
        self.base_ttft = 0.15    # Base time-to-first-token in seconds
        self.tokens_per_sec_base = 50.0  # Base tokens per second
        
    def setup_model(self, config: BenchmarkConfig) -> None:
        """Setup the mock model based on configuration."""
        self.logger.info(f"Setting up mock model: {config.model_id} with variant: {config.variant}")
        self.device = config.device
        self.config = config
        
        # Simulate model loading time based on variant
        if config.variant == "eager":
            load_time = 0.5
        elif config.variant == "compiled":
            load_time = 2.0  # Compilation takes longer
        elif config.variant == "kernelized":
            load_time = 3.0  # Kernel optimization takes longest
        else:
            load_time = 0.5
        
        self.logger.info(f"Simulating model loading for {load_time:.1f} seconds...")
        time.sleep(load_time)
        
        # Adjust base performance based on variant
        if config.variant == "compiled":
            self.base_latency *= 0.8  # Compiled is 20% faster
            self.base_ttft *= 0.9     # Slightly faster TTFT
            self.tokens_per_sec_base *= 1.25  # 25% higher throughput
        elif config.variant == "kernelized":
            self.base_latency *= 0.6  # Kernelized is 40% faster
            self.base_ttft *= 0.7     # Much faster TTFT
            self.tokens_per_sec_base *= 1.67  # 67% higher throughput
        
        # Adjust for compile mode
        if config.compile_mode == "max-autotune":
            self.base_latency *= 0.9
            self.tokens_per_sec_base *= 1.1
        elif config.compile_mode == "reduce-overhead":
            self.base_latency *= 0.95
            self.tokens_per_sec_base *= 1.05
        
        # Adjust for SDPA backend (simulate performance differences)
        if config.sdpa_backend == "flash_attention":
            self.base_latency *= 0.85  # Flash attention is faster
            self.tokens_per_sec_base *= 1.18
        elif config.sdpa_backend == "efficient_attention":
            self.base_latency *= 0.92  # Memory efficient is moderately faster
            self.tokens_per_sec_base *= 1.09
        elif config.sdpa_backend == "math":
            self.base_latency *= 1.05  # Math backend is slower (fallback)
            self.tokens_per_sec_base *= 0.95
        
        self.model_loaded = True
        self.logger.info("Mock model setup complete")
    
    def cleanup_model(self) -> None:
        """Cleanup mock model resources."""
        if self.model_loaded:
            self.logger.info("Cleaning up mock model...")
            time.sleep(0.1)  # Simulate cleanup time
            self.model_loaded = False
            
        # Simulate memory flush
        flush_memory()
    
    def measure_time_to_first_token(self, config: BenchmarkConfig) -> float:
        """Measure mock time to first token generation."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Simulate processing time with some realistic variation
        base_time = self.base_ttft
        
        # Add variation based on sequence length
        if config.sequence_length:
            base_time += config.sequence_length * 0.001  # 1ms per token
        
        # Add some random variation (±10%)
        variation = random.uniform(-0.1, 0.1)
        actual_time = base_time * (1 + variation)
        
        # Simulate the actual processing
        time.sleep(min(actual_time, 0.01))  # Cap sleep time for testing
        
        return actual_time
    
    def measure_latency(self, config: BenchmarkConfig) -> TimingResult:
        """Measure mock full generation latency and compute tokens/sec."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Calculate base latency based on tokens to generate
        tokens_to_generate = config.num_tokens_to_generate
        base_latency = self.base_latency + (tokens_to_generate / self.tokens_per_sec_base)
        
        # Add variation based on batch size (larger batches are slightly less efficient per sample)
        if config.batch_size > 1:
            base_latency *= (1 + (config.batch_size - 1) * 0.1)
        
        # Add some random variation (±5%)
        variation = random.uniform(-0.05, 0.05)
        actual_latency = base_latency * (1 + variation)
        
        # Calculate tokens per second
        actual_tokens_per_sec = tokens_to_generate / actual_latency
        
        # Simulate the actual processing (cap sleep time for testing)
        time.sleep(min(actual_latency * 0.1, 0.05))
        
        return TimingResult(
            latency=actual_latency,
            tokens_per_second=actual_tokens_per_sec,
            total_tokens_generated=tokens_to_generate,
            metadata={
                "variant": config.variant,
                "compile_mode": config.compile_mode,
                "batch_size": config.batch_size,
                "attn_implementation": config.attn_implementation,
                "sdpa_backend": config.sdpa_backend,
                "mock_benchmark": True
            }
        )
    
    def prepare_inputs(self, config: BenchmarkConfig) -> Any:
        """Prepare mock inputs."""
        return {
            "input_tokens": 10,  # Mock input length
            "prompt": "Mock prompt for testing"
        }


def create_mock_configs(
    warmup_iterations: int = 2,
    measurement_iterations: int = 3,
    num_tokens_to_generate: int = 50
) -> list:
    """Create a set of mock benchmark configurations for testing."""
    from framework import BenchmarkConfig, create_config_variants
    
    base_config = BenchmarkConfig(
        name="mock_benchmark",
        model_id="mock-model-v1",
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        num_tokens_to_generate=num_tokens_to_generate,
        device="cpu",  # Mock doesn't need GPU
        torch_dtype="float32"
    )
    
    # Create variants for different execution modes and SDPA backends
    configs = create_config_variants(
        base_config,
        {
            "variant": ["eager", "compiled"],  # Reduced for mock
            "compile_mode": [None, "default"],
            "batch_size": [1, 2],
            "sdpa_backend": [None, "math", "flash_attention"]  # Mock SDPA variants
        }
    )
    
    # Filter out invalid combinations (same as in llama_benchmark)
    valid_configs = []
    for config in configs:
        # Only apply compile_mode when variant is compiled or kernelized
        if config.variant == "eager" and config.compile_mode is not None:
            continue
        # Kernelized mode should use max-autotune
        if config.variant == "kernelized" and config.compile_mode not in ["max-autotune", None]:
            continue
        
        valid_configs.append(config)
    
    return valid_configs


def run_mock_benchmark(
    logger: logging.Logger,
    output_dir: str = "benchmark_results",
    skip_by_default: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Run the mock benchmark with multiple configurations.
    
    Args:
        logger: Logger instance
        output_dir: Output directory for results
        skip_by_default: If True, skip the benchmark unless explicitly enabled
        **kwargs: Additional configuration parameters
        
    Returns:
        Path to the saved results JSON file, or None if skipped
    """
    # Check if benchmark should be skipped
    enable_mock = kwargs.get('enable_mock', False)
    logger.debug(f"Mock benchmark: skip_by_default={skip_by_default}, enable_mock={enable_mock}")
    if skip_by_default and not enable_mock:
        logger.info("Mock benchmark is skipped by default. Use --enable-mock to run it.")
        return None
    
    from framework import BenchmarkRunner
    
    logger.info("Running mock benchmark (for testing/demonstration)")
    
    # Create benchmark and runner
    benchmark = MockBenchmark(logger)
    runner = BenchmarkRunner(logger, output_dir)
    
    # Create configurations with fast settings for testing
    configs = create_mock_configs(
        warmup_iterations=kwargs.get('warmup_iterations', 1),
        measurement_iterations=kwargs.get('measurement_iterations', 2),
        num_tokens_to_generate=kwargs.get('num_tokens_to_generate', 20)
    )
    
    logger.info(f"Running mock benchmark with {len(configs)} configurations")
    
    # Run benchmarks
    results = runner.run_benchmark(benchmark, configs, collect_gpu_metrics=False)
    
    # Save results
    output_file = runner.save_results("mock_model", results)
    
    logger.info(f"Mock benchmark completed. Results saved to: {output_file}")
    return output_file


# Alternative function name for discovery
def run_benchmark(*args, **kwargs):
    """Alias for run_mock_benchmark to match discovery pattern."""
    return run_mock_benchmark(*args, **kwargs)


if __name__ == "__main__":
    # For testing purposes
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s - %(asctime)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run a quick test
    logger.info("Running mock benchmark test")
    
    try:
        output_file = run_mock_benchmark(
            logger=logger,
            skip_by_default=False,  # Enable for direct testing
            warmup_iterations=1,
            measurement_iterations=2,
            num_tokens_to_generate=10
        )
        if output_file:
            logger.info(f"Test completed successfully: {output_file}")
        else:
            logger.info("Test was skipped")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 