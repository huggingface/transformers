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

import logging
import os
from typing import Any

import torch
from benchmark_framework import ModelBenchmark


os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.set_float32_matmul_precision("high")


class LLaMABenchmark(ModelBenchmark):
    """Simplified LLaMA model benchmark implementation using the ModelBenchmark base class."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self._default_prompt = "Why dogs are so cute?"  # Custom prompt for LLaMA

    def get_default_generation_config(self) -> dict[str, Any]:
        """Get LLaMA-specific generation configuration."""
        return {
            "do_sample": False,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": None,  # Will be set per scenario
        }

    def get_model_init_kwargs(self, config) -> dict[str, Any]:
        """Get LLaMA-specific model initialization kwargs."""
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
    # Extract parameters with common defaults
    from benchmark_framework import BenchmarkRunner, ModelBenchmark

    params = ModelBenchmark.extract_benchmark_kwargs("meta-llama/Llama-2-7b-hf", **kwargs)

    logger.info(f"Starting LLaMA benchmark for model: {params['model_id']}")
    logger.info(
        f"Configuration: warmup={params['warmup_iterations']}, measurement={params['measurement_iterations']}, tokens={params['num_tokens_to_generate']}"
    )

    try:
        # Create benchmark instance
        benchmark = LLaMABenchmark(logger)

        # Create scenarios
        scenario_kwargs = {k: v for k, v in params.items() if k != "commit_id"}
        scenarios = benchmark.create_scenarios(**scenario_kwargs)

        logger.info(f"Created {len(scenarios)} benchmark scenarios")

        # Create runner and execute benchmarks
        runner = BenchmarkRunner(logger, output_dir)
        results = runner.run_benchmark(benchmark, scenarios, commit_id=params["commit_id"])

        if not results:
            logger.warning("No successful benchmark results")
            return None

        # Save results
        model_name = params["model_id"].split("/")[-1]  # Extract model name from ID
        output_file = runner.save_results(model_name, results)

        logger.info(f"LLaMA benchmark completed successfully. Results saved to: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"LLaMA benchmark failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        raise
