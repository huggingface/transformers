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

import time
import random
import logging
from typing import Dict, Any, List

from benchmark_framework import AbstractModelBenchmark, BenchmarkConfig, BenchmarkScenario, TimingResult


class MockBenchmark(AbstractModelBenchmark):
    """Mock benchmark for testing and demonstration purposes."""
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.mock_model = None
        self.mock_tokenizer = None
    
    def create_scenarios(self, **kwargs) -> Dict[str, BenchmarkScenario]:
        """Create mock benchmark scenarios."""
        scenarios = {}
        
        # Extract parameters
        model_id = kwargs.get('model_id', 'mock/model')
        warmup_iterations = kwargs.get('warmup_iterations', 2)
        measurement_iterations = kwargs.get('measurement_iterations', 3)
        num_tokens_to_generate = kwargs.get('num_tokens_to_generate', 50)
        device = kwargs.get('device', 'cpu')
        
        # Create simple mock scenarios
        scenario_configs = [
            {"variant": "eager", "description": "Mock eager execution"},
            {"variant": "compiled", "description": "Mock compiled execution"},
        ]
        
        for scenario_config in scenario_configs:
            config = BenchmarkConfig(
                name=scenario_config['variant'],
                model_id=model_id,
                variant=scenario_config["variant"],
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                num_tokens_to_generate=num_tokens_to_generate,
                device=device,
                torch_dtype="float32"
            )
            
            scenario = BenchmarkScenario(
                name=scenario_config["variant"],
                config=config,
                description=scenario_config["description"]
            )
            
            scenarios[scenario_config["variant"]] = scenario
        
        return scenarios
    
    def setup_model(self, config: BenchmarkConfig) -> None:
        """Setup mock model."""
        self.logger.info(f"Setting up mock model: {config.model_id}")
        self.device = config.device
        
        # Simulate model loading time
        time.sleep(0.1)
        
        self.mock_model = {"type": "mock", "device": config.device}
        self.mock_tokenizer = {"type": "mock"}
        
        self.logger.info("Mock model setup complete")
    
    def cleanup_model(self) -> None:
        """Cleanup mock model."""
        self.mock_model = None
        self.mock_tokenizer = None
        self.logger.debug("Mock model cleaned up")
    
    def measure_time_to_first_token(self, config: BenchmarkConfig) -> float:
        """Mock time to first token measurement."""
        # Simulate variable timing
        base_time = 0.01
        variation = random.uniform(0.005, 0.02)
        
        time.sleep(base_time + variation)
        return base_time + variation
    
    def measure_latency(self, config: BenchmarkConfig) -> TimingResult:
        """Mock latency measurement."""
        # Simulate longer generation time
        base_latency = 0.1
        variation = random.uniform(0.02, 0.05)
        latency = base_latency + variation
        
        time.sleep(latency)
        
        # Mock token generation metrics
        tokens_generated = config.num_tokens_to_generate
        tokens_per_second = tokens_generated / latency
        time_per_output_token = latency / tokens_generated
        
        return TimingResult(
            latency=latency,
            tokens_per_second=tokens_per_second,
            time_per_output_token_seconds=time_per_output_token,
            total_tokens_generated=tokens_generated,
            metadata={
                "input_length": 10,  # Mock input length
                "output_length": 10 + tokens_generated,
                "variant": config.variant,
                "mock": True
            }
        )


def run_mock_benchmark(logger, output_dir, enable_mock=False, **kwargs):
    """
    Run mock benchmark with the given configuration.
    
    Args:
        logger: Logger instance
        output_dir: Output directory for results
        enable_mock: Whether mock benchmark is enabled
        **kwargs: Additional configuration options
        
    Returns:
        Path to output file if successful, None if disabled
    """
    # Check if mock benchmark is enabled
    if not enable_mock:
        logger.info("Mock benchmark is disabled (use --enable-mock to enable)")
        return None
    
    from benchmark_framework import BenchmarkRunner
    
    # Extract parameters with defaults
    model_id = kwargs.get('model_id', 'mock/test-model')
    warmup_iterations = kwargs.get('warmup_iterations', 2)
    measurement_iterations = kwargs.get('measurement_iterations', 3)
    num_tokens_to_generate = kwargs.get('num_tokens_to_generate', 50)
    device = kwargs.get('device', 'cpu')
    
    logger.info(f"Starting mock benchmark for model: {model_id}")
    logger.info(f"Configuration: warmup={warmup_iterations}, measurement={measurement_iterations}, tokens={num_tokens_to_generate}")
    
    try:
        # Create benchmark instance
        benchmark = MockBenchmark(logger)
        
        # Create scenarios
        scenarios = benchmark.create_scenarios(
            model_id=model_id,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            num_tokens_to_generate=num_tokens_to_generate,
            device=device
        )
        
        logger.info(f"Created {len(scenarios)} mock benchmark scenarios")
        
        # Create runner and execute benchmarks
        runner = BenchmarkRunner(logger, output_dir)
        results = runner.run_benchmark(benchmark, scenarios, collect_gpu_metrics=False)
        
        if not results:
            logger.warning("No successful mock benchmark results")
            return None
        
        # Save results
        model_name = model_id.split('/')[-1]  # Extract model name from ID
        output_file = runner.save_results(model_name, results)
        
        logger.info(f"Mock benchmark completed successfully. Results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Mock benchmark failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
