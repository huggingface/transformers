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

import os
import logging
from typing import Dict, Any, List

from benchmark_framework import ModelBenchmark

import torch

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.set_float32_matmul_precision("high")

class LLaMABenchmark(ModelBenchmark):
    """Simplified LLaMA model benchmark implementation using the ModelBenchmark base class."""
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self._default_prompt = "Why dogs are so cute?"  # Custom prompt for LLaMA
    

    
    def get_scenario_configs(self) -> List[Dict[str, Any]]:
        """
        Get LLaMA-specific scenario configurations.
        
        Returns:
            List of scenario configuration dictionaries
        """
        return [
            # Eager variants
            {"variant": "eager", "compile_mode": None, "use_cache": True, "description": "Eager execution with cache"},
            
            # Compiled variants
            {"variant": "compiled", "compile_mode": "max-autotune", "use_cache": True, "description": "Compiled with max autotune"},
            
            # Kernelized variant (if available)
            {"variant": "kernelized", "compile_mode": "max-autotune", "use_cache": True, "description": "Kernelized execution"},
        ]
    
    def _is_kernelization_available(self) -> bool:
        """Check if kernelization is available for LLaMA."""
        try:
            from kernels import Mode, kernelize
            return True
        except ImportError:
            self.logger.debug("Kernelization not available: kernels module not found")
            return False
    
    def get_default_generation_config(self) -> Dict[str, Any]:
        """Get LLaMA-specific generation configuration."""
        return {
            "do_sample": False,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": None,  # Will be set per scenario
        }
    
    def get_model_init_kwargs(self, config) -> Dict[str, Any]:
        """Get LLaMA-specific model initialization kwargs."""
        from benchmark_framework import BenchmarkConfig
        return {
            "torch_dtype": getattr(torch, config.torch_dtype),
            "attn_implementation": config.attn_implementation,
            "use_cache": True,
        }
    
    def get_default_torch_dtype(self) -> str:
        """Get default torch dtype for LLaMA."""
        return "float16"  # LLaMA works well with float16
    
    def get_default_device(self) -> str:
        """Get default device for LLaMA."""
        return "cuda"  # LLaMA prefers CUDA


def run_llama(logger, output_dir, **kwargs):
    """
    Run LLaMA benchmark with the given configuration.
    
    Args:
        logger: Logger instance
        output_dir: Output directory for results
        **kwargs: Additional configuration options
        
    Returns:
        Path to output file if successful
    """
    from benchmark_framework import BenchmarkRunner
    
    # Extract parameters with defaults
    model_id = kwargs.get('model_id', 'meta-llama/Llama-2-7b-hf')
    warmup_iterations = kwargs.get('warmup_iterations', 3)
    measurement_iterations = kwargs.get('measurement_iterations', 5)
    num_tokens_to_generate = kwargs.get('num_tokens_to_generate', 100)
    include_sdpa_variants = kwargs.get('include_sdpa_variants', True)
    device = kwargs.get('device', 'cuda')
    torch_dtype = kwargs.get('torch_dtype', 'float16')
    batch_size = kwargs.get('batch_size', 1)
    commit_id = kwargs.get('commit_id', None)
    
    logger.info(f"Starting LLaMA benchmark for model: {model_id}")
    logger.info(f"Configuration: warmup={warmup_iterations}, measurement={measurement_iterations}, tokens={num_tokens_to_generate}")
    
    try:
        # Create benchmark instance
        benchmark = LLaMABenchmark(logger)
        
        # Create scenarios
        scenarios = benchmark.create_scenarios(
            model_id=model_id,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            num_tokens_to_generate=num_tokens_to_generate,
            include_sdpa_variants=include_sdpa_variants,
            device=device,
            torch_dtype=torch_dtype,
            batch_size=batch_size
        )
        
        logger.info(f"Created {len(scenarios)} benchmark scenarios")
        
        # Create runner and execute benchmarks
        runner = BenchmarkRunner(logger, output_dir)
        results = runner.run_benchmark(benchmark, scenarios, commit_id=commit_id)
        
        if not results:
            logger.warning("No successful benchmark results")
            return None
        
        # Save results
        model_name = model_id.split('/')[-1]  # Extract model name from ID
        output_file = runner.save_results(model_name, results)
        
        logger.info(f"LLaMA benchmark completed successfully. Results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"LLaMA benchmark failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise