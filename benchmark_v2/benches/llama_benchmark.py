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
import sys
import time
import logging
from typing import Any, Optional

# Add parent directory to path for framework imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import ModelBenchmark, BenchmarkConfig, TimingResult, flush_memory, SDPAContext

# Optional ML dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    GenerationConfig = None
    StaticCache = None

# Set up environment variables for optimal performance
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "1"

# Only set torch precision if torch is available
if TRANSFORMERS_AVAILABLE:
    torch.set_float32_matmul_precision("high")


class LLaMABenchmark(ModelBenchmark):
    """LLaMA model benchmark implementation with support for eager, compiled, and kernelized variants."""
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.inputs = None
        self.prompt = "Why dogs are so cute?"
        self.compiled_model = None
        self.past_key_values = None
        self.config = None
        
    def setup_model(self, config: BenchmarkConfig) -> None:
        """Setup the LLaMA model for benchmarking with the given configuration."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Transformers and torch are required to run the LLaMA benchmark. "
                "Please install them with: pip install torch transformers"
            )
        
        self.logger.info(f"Setting up model: {config.model_id} with variant: {config.variant}")
        self.device = config.device
        self.config = config
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare generation config
        gen_config = GenerationConfig(**config.generation_config)
        
        # Load model
        self.logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, 
            torch_dtype=getattr(torch, config.torch_dtype),
            generation_config=gen_config,
            device_map="auto" if config.device == "cuda" else None,
            attn_implementation=config.attn_implementation
        ).eval()
        
        if config.device != "auto":
            self.model.to(self.device)
        
        # Prepare inputs
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt")
        if config.device == "cuda":
            self.inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        
        seq_length = self.inputs["input_ids"].shape[1]
        self.model.generation_config.max_length = seq_length + config.num_tokens_to_generate
        
        # Setup model variant
        self._setup_variant(config)
        
        self.logger.info("Model setup complete")
    
    def _setup_variant(self, config: BenchmarkConfig) -> None:
        """Setup the specific model variant (eager, compiled, kernelized)."""
        if config.variant == "eager":
            # Nothing special needed for eager mode
            self.logger.info("Using eager execution mode")
            
        elif config.variant == "compiled":
            self.logger.info(f"Setting up compiled mode with: {config.compile_mode}")
            if config.compile_mode == "max-autotune":
                self.compiled_model = torch.compile(
                    self.model, 
                    mode="max-autotune", 
                    fullgraph=True,
                    **config.compile_options
                )
            elif config.compile_mode == "reduce-overhead":
                self.compiled_model = torch.compile(
                    self.model, 
                    mode="reduce-overhead",
                    **config.compile_options
                )
            else:
                self.compiled_model = torch.compile(self.model, **config.compile_options)
            
            # Setup static cache for compiled mode
            if config.use_cache:
                seq_length = self.inputs["input_ids"].shape[1]
                self.past_key_values = StaticCache(
                    self.model.config,
                    max_batch_size=config.batch_size,
                    device=self.device,
                    dtype=getattr(torch, config.torch_dtype),
                    max_cache_len=seq_length + config.num_tokens_to_generate,
                )
                
        elif config.variant == "kernelized":
            # For kernelized mode, we'll use specific kernel optimizations
            self.logger.info("Setting up kernelized mode")
            # This would typically involve setting up specific CUDA kernels
            # For now, we'll use torch.compile with specific backend configurations
            try:
                self.compiled_model = torch.compile(
                    self.model,
                    backend="inductor",
                    mode="max-autotune",
                    **config.compile_options
                )
            except Exception as e:
                self.logger.warning(f"Failed to setup kernelized mode: {e}")
                self.logger.warning("Falling back to eager mode")
                config.variant = "eager"
        else:
            raise ValueError(f"Unknown variant: {config.variant}")
    
    def cleanup_model(self) -> None:
        """Cleanup model resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'compiled_model') and self.compiled_model is not None:
            del self.compiled_model
            self.compiled_model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, 'past_key_values') and self.past_key_values is not None:
            del self.past_key_values
            self.past_key_values = None
        
        # Clear CUDA cache
        flush_memory()
    
    def measure_time_to_first_token(self, config: BenchmarkConfig) -> float:
        """Measure time to first token generation."""
        model_to_use = self.compiled_model if self.compiled_model is not None else self.model
        
        # Prepare generation kwargs
        generation_kwargs = {
            **self.inputs,
            "max_new_tokens": 1,  # Only generate first token
            "do_sample": False,
            "temperature": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if self.past_key_values is not None and config.variant == "compiled":
            # Reset cache for each measurement
            seq_length = self.inputs["input_ids"].shape[1]
            fresh_cache = StaticCache(
                self.model.config,
                max_batch_size=config.batch_size,
                device=self.device,
                dtype=getattr(torch, config.torch_dtype),
                max_cache_len=seq_length + 1,
            )
            generation_kwargs["past_key_values"] = fresh_cache
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Use SDPA context if specified
        with SDPAContext(config.sdpa_backend, self.logger):
            with torch.no_grad():
                outputs = model_to_use.generate(**generation_kwargs)
        
        # Synchronize after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def measure_latency(self, config: BenchmarkConfig) -> TimingResult:
        """Measure full generation latency and compute tokens/sec."""
        model_to_use = self.compiled_model if self.compiled_model is not None else self.model
        
        # Prepare generation kwargs
        generation_kwargs = {
            **self.inputs,
            "max_new_tokens": config.num_tokens_to_generate,
            "do_sample": config.generation_config.get("do_sample", False),
            "temperature": config.generation_config.get("temperature", 1.0),
            "top_p": config.generation_config.get("top_p", 1.0),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if self.past_key_values is not None and config.variant == "compiled":
            # Reset cache for each measurement
            seq_length = self.inputs["input_ids"].shape[1]
            fresh_cache = StaticCache(
                self.model.config,
                max_batch_size=config.batch_size,
                device=self.device,
                dtype=getattr(torch, config.torch_dtype),
                max_cache_len=seq_length + config.num_tokens_to_generate,
            )
            generation_kwargs["past_key_values"] = fresh_cache
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Use SDPA context if specified
        with SDPAContext(config.sdpa_backend, self.logger):
            with torch.no_grad():
                outputs = model_to_use.generate(**generation_kwargs)
        
        # Synchronize after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        latency = end_time - start_time
        input_length = self.inputs["input_ids"].shape[1]
        output_length = outputs.shape[1]
        tokens_generated = output_length - input_length
        
        tokens_per_second = tokens_generated / latency if latency > 0 else 0
        
        return TimingResult(
            latency=latency,
            tokens_per_second=tokens_per_second,
            total_tokens_generated=tokens_generated,
            metadata={
                "input_length": input_length,
                "output_length": output_length,
                "variant": config.variant,
                "compile_mode": config.compile_mode,
                "attn_implementation": config.attn_implementation,
                "sdpa_backend": config.sdpa_backend
            }
        )


def create_llama_configs(
    model_id: str = "meta-llama/Llama-2-7b-hf",
    warmup_iterations: int = 3,
    measurement_iterations: int = 5,
    num_tokens_to_generate: int = 100,
    include_sdpa_variants: bool = True,
    **kwargs  # Accept additional arguments and ignore them
) -> list:
    """Create a comprehensive set of LLaMA benchmark configurations."""
    from framework import BenchmarkConfig, create_config_variants, get_available_sdpa_backends
    
    base_config = BenchmarkConfig(
        name="llama_benchmark",
        model_id=model_id,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        num_tokens_to_generate=num_tokens_to_generate,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="float16"
    )
    
    # Get available SDPA backends
    available_backends = get_available_sdpa_backends() if include_sdpa_variants else ["math"]
    sdpa_backends = [None] + available_backends  # None means default behavior
    
    # Create variants for different execution modes and attention implementations
    variant_dict = {
        "variant": ["eager", "compiled", "kernelized"],
        "compile_mode": [None, "default", "reduce-overhead", "max-autotune"],
        "use_cache": [True, False],
    }
    
    # Add SDPA backend variants if requested
    if include_sdpa_variants and available_backends:
        variant_dict["sdpa_backend"] = sdpa_backends
    
    configs = create_config_variants(base_config, variant_dict)
    
    # Filter out invalid combinations
    valid_configs = []
    for config in configs:
        # Only apply compile_mode when variant is compiled or kernelized
        if config.variant == "eager" and config.compile_mode is not None:
            continue
        # Only use cache with compiled variants
        if config.variant == "eager" and config.use_cache is False:
            continue
        # Kernelized mode should always use max-autotune
        if config.variant == "kernelized" and config.compile_mode not in ["max-autotune", None]:
            continue
        
        valid_configs.append(config)
    
    return valid_configs


def run_llama_benchmark(
    logger: logging.Logger,
    output_dir: str = "benchmark_results",
    model_id: str = "meta-llama/Llama-2-7b-hf",
    **kwargs
) -> str:
    """
    Run the LLaMA benchmark with multiple configurations.
    
    Returns:
        Path to the saved results JSON file
    """
    from framework import BenchmarkRunner
    
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Transformers and torch are required. "
            "Install with: pip install torch transformers"
        )
    
    # Create benchmark and runner
    benchmark = LLaMABenchmark(logger)
    runner = BenchmarkRunner(logger, output_dir)
    
    # Create configurations
    configs = create_llama_configs(model_id=model_id, **kwargs)
    
    logger.info(f"Running LLaMA benchmark with {len(configs)} configurations")
    
    # Run benchmarks
    results = runner.run_benchmark(benchmark, configs)
    
    # Save results
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    output_file = runner.save_results(model_name, results)
    
    logger.info(f"LLaMA benchmark completed. Results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # For testing purposes
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s - %(asctime)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run a quick test with a smaller model
    test_model = "gpt2"  # Use GPT-2 for faster testing
    logger.info(f"Running test benchmark with {test_model}")
    
    try:
        output_file = run_llama_benchmark(
            logger=logger,
            model_id=test_model,
            warmup_iterations=1,
            measurement_iterations=2,
            num_tokens_to_generate=10
        )
        logger.info(f"Test completed successfully: {output_file}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 