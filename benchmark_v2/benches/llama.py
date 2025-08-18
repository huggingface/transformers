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
from typing import Any, Optional, Dict

from benchmark_framework import ModelBenchmark, BenchmarkConfig, BenchmarkScenario, TimingResult, flush_memory, SDPAContext, CUDATimer

from kernels import Mode, kernelize

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "1"
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
    
    def create_scenarios(self, **kwargs) -> Dict[str, BenchmarkScenario]:
        """Create benchmark scenarios for LLaMA."""
        scenarios = {}
        
        # Extract parameters
        model_id = kwargs.get('model_id', 'meta-llama/Llama-2-7b-hf')
        warmup_iterations = kwargs.get('warmup_iterations', 3)
        measurement_iterations = kwargs.get('measurement_iterations', 5)
        num_tokens_to_generate = kwargs.get('num_tokens_to_generate', 100)
        include_sdpa_variants = kwargs.get('include_sdpa_variants', True)
        
        # Generate scenarios for different attention implementations and backends
        # The framework will automatically skip scenarios that don't work
        attention_configs = [
            {"attn_implementation": "eager", "sdpa_backends": [None], "desc_suffix": " with eager attention"},
        ]
        
        # Add SDPA variants if requested
        if include_sdpa_variants:
            attention_configs.append({
                "attn_implementation": "sdpa", 
                "sdpa_backends": [None, "math", "flash_attention", "efficient_attention"],
                "desc_suffix": ""
            })
        
        # Define scenario configurations
        scenario_configs = [
            # Eager variants
            {"variant": "eager", "compile_mode": None, "use_cache": True, "description": "Eager execution with cache"},
            
            # Compiled variants
            {"variant": "compiled", "compile_mode": "max-autotune", "use_cache": True, "description": "Compiled with max autotune"},
            
            # Kernelized variant
            {"variant": "kernelized", "compile_mode": "max-autotune", "use_cache": True, "description": "Kernelized execution"},
        ]
        
        # Create scenarios for each attention config and variant combination
        for attn_config in attention_configs:
            attn_implementation = attn_config["attn_implementation"]
            sdpa_backends = attn_config["sdpa_backends"]
            desc_suffix = attn_config["desc_suffix"]
            
            for scenario_config in scenario_configs:
                for sdpa_backend in sdpa_backends:
                    # Create unique config for this scenario
                    config = BenchmarkConfig(
                        name=f"llama_{scenario_config['variant']}",
                        model_id=model_id,
                        variant=scenario_config["variant"],
                        compile_mode=scenario_config["compile_mode"],
                        use_cache=scenario_config["use_cache"],
                        warmup_iterations=warmup_iterations,
                        measurement_iterations=measurement_iterations,
                        num_tokens_to_generate=num_tokens_to_generate,
                        device="cuda",
                        torch_dtype="float16",
                        attn_implementation=attn_implementation,
                        sdpa_backend=sdpa_backend if attn_implementation == "sdpa" else None
                    )
                    
                    # Create scenario name
                    scenario_name_parts = [scenario_config["variant"]]
                    if scenario_config["compile_mode"]:
                        scenario_name_parts.append(f"compile_{scenario_config['compile_mode']}")
                    
                    # Add attention implementation to name
                    if attn_implementation == "eager":
                        scenario_name_parts.append("eager_attn")
                    elif attn_implementation == "sdpa":
                        if sdpa_backend:
                            scenario_name_parts.append(f"sdpa_{sdpa_backend}")
                        else:
                            scenario_name_parts.append("sdpa_default")
                    
                    scenario_name = "_".join(scenario_name_parts)
                    
                    # Create description
                    description = scenario_config["description"]
                    if attn_implementation == "sdpa" and sdpa_backend:
                        description += f" with SDPA {sdpa_backend} backend"
                    elif attn_implementation == "sdpa":
                        description += " with SDPA default backend"
                    else:
                        description += desc_suffix
                    
                    # Create scenario
                    scenario = BenchmarkScenario(
                        name=scenario_name,
                        config=config,
                        description=description
                    )
                    
                    # Add setup callbacks if needed
                    if scenario_config["variant"] == "compiled":
                        scenario.add_setup_callback(self._setup_compilation)
                    elif scenario_config["variant"] == "kernelized":
                        scenario.add_setup_callback(self._setup_kernelization)
                    
                    scenarios[scenario_name] = scenario
        
        return scenarios
    
    def _setup_compilation(self, model, tokenizer, config, logger):
        """Setup callback for compilation scenarios."""
        if logger:
            logger.info(f"Setting up compilation with mode: {config.compile_mode}")
        
        # Perform torch.compile
        if config.compile_mode is not None:
            self.compiled_model = torch.compile(
                model, 
                mode=config.compile_mode, 
                **config.compile_options
            )
        else:
            self.compiled_model = torch.compile(model, **config.compile_options)
        
        # Setup static cache for compiled mode
        if config.use_cache:
            seq_length = self.inputs["input_ids"].shape[1]
            self.past_key_values = StaticCache(
                model.config,
                max_batch_size=config.batch_size,
                device=self.device,
                dtype=getattr(torch, config.torch_dtype),
                max_cache_len=seq_length + config.num_tokens_to_generate,
            )
        
    def _setup_kernelization(self, model, tokenizer, config, logger):
        """Setup callback for kernelization scenarios.""" 
        if logger:
            logger.info("Setting up kernelization")
        
        try:
            self.compiled_model = kernelize(
                model,
                mode=Mode.INFERENCE
            )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to setup kernelized mode: {e}")
                logger.warning("Falling back to eager mode")
            config.variant = "eager"
        
    def setup_model(self, config: BenchmarkConfig) -> None:
        """Setup the LLaMA model for benchmarking with the given configuration."""
        
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
            self._setup_compilation(self.model, self.tokenizer, config, self.logger)
                
        elif config.variant == "kernelized":
            self.logger.info("Setting up kernelized mode")
            self._setup_kernelization(self.model, self.tokenizer, config, self.logger)
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
            "max_new_tokens": 1,
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
        
        # Use CUDA timer for high-precision measurement
        with CUDATimer(device=config.device) as timer:
            # Use SDPA context if specified
            with SDPAContext(config.sdpa_backend, self.logger):
                with torch.no_grad():
                    outputs = model_to_use.generate(**generation_kwargs)
        
        return timer.elapsed_time()
    
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
        
        # Use CUDA timer for high-precision measurement
        with CUDATimer(device=config.device) as timer:
            # Use SDPA context if specified
            with SDPAContext(config.sdpa_backend, self.logger):
                with torch.no_grad():
                    outputs = model_to_use.generate(**generation_kwargs)
        
        # Calculate metrics
        latency = timer.elapsed_time()
        input_length = self.inputs["input_ids"].shape[1]
        output_length = outputs.shape[1]
        tokens_generated = output_length - input_length
        
        tokens_per_second = tokens_generated / latency if latency > 0 else 0
        time_per_output_token = latency / tokens_generated if tokens_generated > 0 else None
        
        return TimingResult(
            latency=latency,
            tokens_per_second=tokens_per_second,
            time_per_output_token_seconds=time_per_output_token,
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
    from benchmark_framework import BenchmarkRunner
    
    # Create benchmark and runner
    benchmark = LLaMABenchmark(logger)
    runner = BenchmarkRunner(logger, output_dir)
    
    # Run scenarios
    scenarios = benchmark.get_scenarios(model_id=model_id, **kwargs)
    logger.info(f"Running LLaMA benchmark with {len(scenarios)} scenarios")
    
    # Run benchmarks
    results = runner.run_benchmark(benchmark, scenarios)
    
    # Save results
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    output_file = runner.save_results(model_name, results)
    
    logger.info(f"LLaMA benchmark completed. Results saved to: {output_file}")
    return output_file


def run_benchmark(*args, **kwargs):
    """
    Generic benchmark runner function for discovery compatibility.
    Delegates to run_llama_benchmark.
    """
    return run_llama_benchmark(*args, **kwargs)