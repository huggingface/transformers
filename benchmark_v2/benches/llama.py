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
    
    @property
    def model_type(self) -> str:
        """Model type identifier."""
        return "llama"
    
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


def run_benchmark(*args, **kwargs):
    """
    Generic benchmark runner function for discovery compatibility.
    Delegates to run_llama_benchmark.
    """
    return run_llama_benchmark(*args, **kwargs)