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

import gc
import json
import os
import subprocess
import sys
import time
import statistics
import threading
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, TypedDict
import logging

import numpy as np
import psutil
import gpustat

import torch


class GPUMetrics(TypedDict):
    """GPU monitoring result with GPU metrics."""
    gpu_utilization_mean: float
    gpu_utilization_max: float
    gpu_utilization_min: float
    gpu_memory_used_mean: float
    gpu_memory_used_max: float
    gpu_memory_used_min: float
    sample_count: int
    gpu_monitoring_status: str


class NoGPU(TypedDict):
    """GPU monitoring result without GPU metrics."""
    gpu_monitoring_status: str
    gpu_monitoring_reason: str


class ArchAwareTimer:
    """Architecture-aware timer for supposedly better prescision"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize architecture-aware timer.
        
        Args:
            device: Device to use. If None, uses current device.
        """
        self.device = device
        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            if device and device != "cpu":
                self.device_obj = torch.device(device)
            else:
                # Fall back to CPU timing if device is CPU or CUDA not available
                self.use_cuda = False
        
        if self.use_cuda:
            try:
                # Create CUDA events for timing
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
            except RuntimeError:
                # Fall back to CPU timing if CUDA events fail
                self.use_cuda = False
        
        if not self.use_cuda:
            self.start_time = None
            self.end_time = None
    
    def start(self):
        """Start timing."""
        if self.use_cuda:
            torch.cuda.synchronize(self.device_obj)
            self.start_event.record(stream=torch.cuda.current_stream(self.device_obj))
        else:
            self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing."""
        if self.use_cuda:
            self.end_event.record(stream=torch.cuda.current_stream(self.device_obj))
            torch.cuda.synchronize(self.device_obj)
        else:
            self.end_time = time.perf_counter()
    
    def elapsed_time(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self.use_cuda:
            # CUDA events return time in milliseconds, convert to seconds
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            if self.start_time is None or self.end_time is None:
                raise RuntimeError("Timer not properly started/stopped")
            return self.end_time - self.start_time
    
    @property
    def timing_method(self) -> str:
        """Get the timing method being used."""
        return "CUDA Events" if self.use_cuda else "CPU perf_counter"
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark scenario."""
    name: str
    model_id: str
    variant: str = "eager"  # "eager", "compiled", "kernelized"
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    num_tokens_to_generate: int = 100
    device: str = "cuda"
    torch_dtype: str = "float16"
    compile_mode: Optional[str] = None  # None, "default", "reduce-overhead", "max-autotune"
    compile_options: Dict[str, Any] = field(default_factory=dict)
    use_cache: bool = True
    batch_size: int = 1
    sequence_length: Optional[int] = None
    attn_implementation: str = "sdpa"  # "eager", "sdpa", "flash_attention_2"
    sdpa_backend: Optional[str] = None  # None, "math", "flash_attention", "efficient_attention", "cudnn_attention"
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BenchmarkScenario:
    """
    A benchmark scenario that encapsulates both configuration and setup logic.
    This makes it easier to define and adapt benchmarks for different models.
    """
    
    def __init__(self, name: str, config: BenchmarkConfig, description: str = ""):
        self.name = name
        self.config = config
        self.description = description
        self._setup_callbacks = []
        self._teardown_callbacks = []
    
    def add_setup_callback(self, callback: callable):
        """Add a callback to be executed during scenario setup."""
        self._setup_callbacks.append(callback)
    
    def add_teardown_callback(self, callback: callable):
        """Add a callback to be executed during scenario teardown."""
        self._teardown_callbacks.append(callback)
    
    def setup(self, model, tokenizer, logger=None):
        """Execute setup callbacks for this scenario."""
        for callback in self._setup_callbacks:
            try:
                callback(model, tokenizer, self.config, logger)
            except Exception as e:
                if logger:
                    logger.warning(f"Setup callback failed for scenario {self.name}: {e}")
    
    def teardown(self, model, tokenizer, logger=None):
        """Execute teardown callbacks for this scenario."""
        for callback in self._teardown_callbacks:
            try:
                callback(model, tokenizer, self.config, logger)
            except Exception as e:
                if logger:
                    logger.warning(f"Teardown callback failed for scenario {self.name}: {e}")
    
    def __repr__(self):
        return f"BenchmarkScenario(name='{self.name}', variant='{self.config.variant}')"




@dataclass
class TimingResult:
    """Result from a timing measurement."""
    time_to_first_token_seconds: Optional[float] = None
    latency_seconds: float = 0.0
    tokens_per_second: Optional[float] = None
    time_per_output_token_seconds: Optional[float] = None
    total_tokens_generated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkStatistics:
    """Statistical analysis of benchmark measurements."""
    name: str
    measurements: List[float]
    mean: float
    median: float
    std: float
    min: float
    max: float
    p25: float  # 25th percentile
    p75: float  # 75th percentile
    p90: float  # 90th percentile
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    unit: str = "seconds"

    @classmethod
    def from_measurements(cls, name: str, measurements: List[float], unit: str = "seconds") -> 'BenchmarkStatistics':
        """Create statistics from a list of measurements."""
        if not measurements:
            raise ValueError("Cannot create statistics from empty measurements")
        
        measurements_array = np.array(measurements)
        
        return cls(
            name=name,
            measurements=measurements,
            mean=float(np.mean(measurements_array)),
            median=float(np.median(measurements_array)),
            std=float(np.std(measurements_array)),
            min=float(np.min(measurements_array)),
            max=float(np.max(measurements_array)),
            p25=float(np.percentile(measurements_array, 25)),
            p75=float(np.percentile(measurements_array, 75)),
            p90=float(np.percentile(measurements_array, 90)),
            p95=float(np.percentile(measurements_array, 95)),
            p99=float(np.percentile(measurements_array, 99)),
            unit=unit
        )


@dataclass 
class HardwareInfo:
    """Hardware information collected during benchmarking."""
    gpu_name: str
    gpu_memory_total_mb: int
    cpu_count: int
    memory_total_mb: int
    python_version: str
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class BenchmarkMetadata:
    """Metadata collected for each benchmark run."""
    timestamp: str
    commit_id: str
    hardware_info: HardwareInfo
    config: BenchmarkConfig


class GPUMonitor:
    """Monitor GPU utilization during benchmark execution."""
    
    def __init__(self, sample_interval: float = 0.1, logger: logging.Logger = None):
        self.sample_interval = sample_interval
        self.logger = logger or logging.getLogger(__name__)
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.timestamps = []
        self.gpu_available = False
        self.warning_logged = False
        
        # Test GPU availability on initialization
        self._test_gpu_availability()
        
    def _test_gpu_availability(self):
        """Test if GPU monitoring is available."""
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            if gpu_stats and len(gpu_stats) > 0:
                self.gpu_available = True
                self.logger.debug(f"GPU monitoring available: {len(gpu_stats)} GPU(s) detected")
            else:
                self.gpu_available = False
                self.logger.debug("No GPUs detected by gpustat")
        except Exception as e:
            self.gpu_available = False
            self.logger.debug(f"GPU monitoring not available: {e}")
        
    def start(self):
        """Start monitoring GPU metrics."""
        if not self.gpu_available:
            self.logger.debug("GPU monitoring disabled: no GPUs available")
            return
            
        # Clear the stop event to enable monitoring
        self.stop_event.clear()
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.timestamps = []
        self.warning_logged = False  # Reset warning flag for new monitoring session
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        self.logger.debug("GPU monitoring started")
        
    def stop_and_collect(self) -> Union[GPUMetrics, NoGPU]:
        """Stop monitoring and return collected metrics."""
        if not self.gpu_available:
            return NoGPU(
                gpu_monitoring_status="disabled",
                gpu_monitoring_reason="no_gpus_available"
            )
            
        # Signal the monitoring thread to stop
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        
        if self.gpu_utilization:
            metrics = GPUMetrics(
                gpu_utilization_mean=statistics.mean(self.gpu_utilization),
                gpu_utilization_max=max(self.gpu_utilization),
                gpu_utilization_min=min(self.gpu_utilization),
                gpu_memory_used_mean=statistics.mean(self.gpu_memory_used),
                gpu_memory_used_max=max(self.gpu_memory_used),
                gpu_memory_used_min=min(self.gpu_memory_used),
                sample_count=len(self.gpu_utilization),
                gpu_monitoring_status="success"
            )
            self.logger.debug(f"GPU monitoring completed: {len(self.gpu_utilization)} samples collected")
            return metrics
        else:
            return NoGPU(
                gpu_monitoring_status="failed",
                gpu_monitoring_reason="no_samples_collected"
            )
    
    def _monitor_loop(self):
        """Background monitoring loop using threading.Event for communication."""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Continue monitoring until stop_event is set
        while not self.stop_event.is_set():
            try:
                gpu_stats = gpustat.GPUStatCollection.new_query()
                if gpu_stats and len(gpu_stats) > 0:
                    gpu = gpu_stats[0]
                    self.gpu_utilization.append(gpu["utilization.gpu"])
                    self.gpu_memory_used.append(gpu["memory.used"])
                    self.timestamps.append(time.time())
                    consecutive_failures = 0  # Reset failure counter on success
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures and not self.warning_logged:
                        self.logger.warning("GPU monitoring: No GPU data returned by gpustat")
                        self.warning_logged = True
                        
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures and not self.warning_logged:
                    self.logger.warning(f"GPU monitoring failed after {max_consecutive_failures} attempts: {e}")
                    self.warning_logged = True
            
            # Use Event.wait() with timeout instead of time.sleep()
            # This allows for immediate response to stop signal while still maintaining sample interval
            if self.stop_event.wait(timeout=self.sample_interval):
                # Event was set, break out of loop immediately
                break


def get_hardware_info() -> HardwareInfo:
    """Collect hardware information."""
    gpu_name = "unknown"
    gpu_memory_total = 0
    
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        if gpu_stats and len(gpu_stats) > 0:
            gpu = gpu_stats[0]
            gpu_name = gpu["name"]
            gpu_memory_total = gpu["memory.total"]
    except Exception:
        pass
    
    torch_version = torch.__version__
    cuda_version = None
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        cuda_version = torch.version.cuda
    
    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_memory_total_mb=gpu_memory_total,
        cpu_count=psutil.cpu_count(),
        memory_total_mb=int(psutil.virtual_memory().total / (1024 * 1024)),
        python_version=f"{sys.version.split()[0]}",
        torch_version=torch_version,
        cuda_version=cuda_version
    )


def flush_memory():
    """Flush GPU memory and run garbage collection."""
    gc.collect()
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_sdpa_backend(backend_name: Optional[str]):
    """Get the SDPA backend enum from string name."""
    if backend_name is None:
        return None
    
    try:
        backend_map = {
            "math": torch.nn.attention.SDPBackend.MATH,
            "flash_attention": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            "efficient_attention": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "cudnn_attention": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        }
        return backend_map.get(backend_name.lower())
    except AttributeError:
        # torch.nn.attention.SDPBackend not available in older torch versions
        return None





class SDPAContext:
    """Context manager for SDPA kernel selection."""
    
    def __init__(self, backend_name: Optional[str], logger: logging.Logger = None):
        self.backend_name = backend_name
        self.logger = logger or logging.getLogger(__name__)
        self.backend = get_sdpa_backend(backend_name) if backend_name else None
        self.context = None
        
    def __enter__(self):
        if self.backend is not None:
            try:
                self.context = torch.nn.attention.sdpa_kernel(self.backend)
                self.context.__enter__()
                if self.logger:
                    self.logger.debug(f"Using SDPA backend: {self.backend_name}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to set SDPA backend {self.backend_name}: {e}")
                self.context = None
        elif self.backend_name and self.logger:
            self.logger.debug(f"SDPA backend '{self.backend_name}' requested but not using kernel context (backend={self.backend})")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error exiting SDPA context: {e}")
        return False


class AbstractModelBenchmark(ABC):
    """Abstract base class for model benchmarks."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.device = None
        self.scenarios = {}  # Map of scenario_name -> BenchmarkScenario
        
    @abstractmethod
    def create_scenarios(self, **kwargs) -> Dict[str, 'BenchmarkScenario']:
        """Create and return a dictionary of benchmark scenarios."""
        pass
        
    @abstractmethod
    def setup_model(self, config: BenchmarkConfig) -> None:
        """Setup the model for benchmarking with the given configuration."""
        pass
    
    @abstractmethod
    def cleanup_model(self) -> None:
        """Cleanup model resources."""
        pass
    
    @abstractmethod
    def measure_time_to_first_token(self, config: BenchmarkConfig) -> float:
        """Measure time to first token generation."""
        pass
    
    @abstractmethod
    def measure_latency(self, config: BenchmarkConfig) -> TimingResult:
        """Measure full generation latency and compute tokens/sec."""
        pass
    
    def prepare_inputs(self, config: BenchmarkConfig) -> Any:
        """Prepare inputs for the model. Override if needed."""
        return None
    
    def get_scenarios(self, **kwargs) -> Dict[str, 'BenchmarkScenario']:
        """Get benchmark scenarios. Creates them if they don't exist."""
        if not self.scenarios:
            self.scenarios = self.create_scenarios(**kwargs)
        return self.scenarios


class ModelBenchmark(AbstractModelBenchmark):
    """
    Base class for HuggingFace Transformers model benchmarks.
    
    This class provides common scenario creation logic and handles the standard
    patterns for eager, compiled, and kernelized execution variants with different
    attention implementations and SDPA backends.
    """
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.inputs = None
        self.compiled_model = None
        self.past_key_values = None
        self.config = None
        self._default_prompt = "Why dogs are so cute?"
        
    @property
    def default_prompt(self) -> str:
        """Default prompt for text generation. Override in subclasses if needed."""
        return self._default_prompt
    

    
    def get_attention_configs(self, include_sdpa_variants: bool = True) -> List[Dict[str, Any]]:
        """
        Get attention implementation configurations.
        
        Args:
            include_sdpa_variants: Whether to include SDPA backend variants
            
        Returns:
            List of attention configuration dictionaries
        """
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
        
        return attention_configs
    
    def get_scenario_configs(self) -> List[Dict[str, Any]]:
        """
        Get base scenario configurations. Override in subclasses to customize.
        
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
        """Check if kernelization is available. Override in subclasses."""
        try:
            from kernels import Mode, kernelize
            return True
        except ImportError:
            return False
    
    def get_default_generation_config(self) -> Dict[str, Any]:
        """Get default generation configuration. Override in subclasses for model-specific defaults."""
        return {
            "do_sample": False,
            "top_p": 1.0,
            "temperature": 1.0
        }
    
    def get_model_init_kwargs(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Get model initialization kwargs. Override in subclasses for model-specific parameters."""
        return {
            "torch_dtype": getattr(torch, config.torch_dtype),
            "attn_implementation": config.attn_implementation
        }
    
    def get_default_torch_dtype(self) -> str:
        """Get default torch dtype. Override in subclasses."""
        return "float16"
    
    def get_default_device(self) -> str:
        """Get default device. Override in subclasses."""
        return "cuda"
    
    def create_scenarios(self, **kwargs) -> Dict[str, 'BenchmarkScenario']:
        """Create benchmark scenarios for HuggingFace models."""
        scenarios = {}
        
        # Extract parameters with model-specific defaults
        model_id = kwargs.get('model_id', 'microsoft/DialoGPT-medium')
        warmup_iterations = kwargs.get('warmup_iterations', 3)
        measurement_iterations = kwargs.get('measurement_iterations', 5)
        num_tokens_to_generate = kwargs.get('num_tokens_to_generate', 100)
        include_sdpa_variants = kwargs.get('include_sdpa_variants', True)
        device = kwargs.get('device', self.get_default_device())
        torch_dtype = kwargs.get('torch_dtype', self.get_default_torch_dtype())
        batch_size = kwargs.get('batch_size', 1)
        
        # Get configurations
        attention_configs = self.get_attention_configs(include_sdpa_variants)
        scenario_configs = self.get_scenario_configs()
        
        # Create scenarios for each attention config and variant combination
        for attn_config in attention_configs:
            attn_implementation = attn_config["attn_implementation"]
            sdpa_backends = attn_config["sdpa_backends"]
            desc_suffix = attn_config["desc_suffix"]
            
            for scenario_config in scenario_configs:
                for sdpa_backend in sdpa_backends:
                    # Skip kernelized if not available
                    if scenario_config["variant"] == "kernelized" and not self._is_kernelization_available():
                        continue
                    
                    # Create unique config for this scenario
                    config = BenchmarkConfig(
                        name=scenario_config['variant'],
                        model_id=model_id,
                        variant=scenario_config["variant"],
                        compile_mode=scenario_config["compile_mode"],
                        use_cache=scenario_config["use_cache"],
                        warmup_iterations=warmup_iterations,
                        measurement_iterations=measurement_iterations,
                        num_tokens_to_generate=num_tokens_to_generate,
                        device=device,
                        torch_dtype=torch_dtype,
                        batch_size=batch_size,
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
                    
                    # Add setup callbacks based on variant
                    if scenario_config["variant"] == "compiled":
                        scenario.add_setup_callback(self._setup_compilation_callback)
                    elif scenario_config["variant"] == "kernelized":
                        scenario.add_setup_callback(self._setup_kernelization_callback)
                    
                    scenarios[scenario_name] = scenario
        
        return scenarios
    
    def _setup_compilation_callback(self, model, tokenizer, config, logger):
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
        
        # Setup static cache for compiled mode if needed
        if config.use_cache and hasattr(self, 'inputs') and self.inputs is not None:
            self._setup_static_cache(config)
    
    def _setup_kernelization_callback(self, model, tokenizer, config, logger):
        """Setup callback for kernelization scenarios.""" 
        if logger:
            logger.info("Setting up kernelization")
        
        try:
            from kernels import Mode, kernelize
            self.compiled_model = kernelize(
                model,
                mode=Mode.INFERENCE
            )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to setup kernelized mode: {e}")
                logger.warning("Falling back to eager mode")
            config.variant = "eager"
    
    def _setup_static_cache(self, config: BenchmarkConfig):
        """Setup static cache for compiled models. Override if needed."""
        if hasattr(self, 'inputs') and self.inputs is not None:
            try:
                from transformers import StaticCache
                seq_length = self.inputs["input_ids"].shape[1]
                
                # Get the actual device the model is on
                if hasattr(self.model, 'device'):
                    cache_device = self.model.device
                else:
                    cache_device = self.device
                
                self.past_key_values = StaticCache(
                    config=self.model.config,
                    max_batch_size=config.batch_size,
                    max_cache_len=seq_length + config.num_tokens_to_generate,
                    device=cache_device,
                    dtype=getattr(torch, config.torch_dtype)
                )
                self.logger.debug(f"StaticCache created on device: {cache_device}")
            except (ImportError, TypeError) as e:
                # StaticCache not available or incompatible, continue without it
                self.logger.debug(f"StaticCache setup failed: {e}, continuing without cache")
                self.past_key_values = None
    
    def setup_model(self, config: BenchmarkConfig) -> None:
        """Setup the HuggingFace model for benchmarking with the given configuration."""
        
        self.logger.info(f"Setting up model: {config.model_id} with variant: {config.variant}")
        self.device = config.device
        self.config = config
        
        # Load model and tokenizer
        self._load_model_and_tokenizer(config)
        
        # Prepare inputs
        self._prepare_model_inputs(config)
        
        # Configure generation settings
        self._configure_generation(config)
        
        self.logger.info("Model setup complete")
    
    def _load_model_and_tokenizer(self, config: BenchmarkConfig):
        """Load the model and tokenizer. Override in subclasses for custom loading."""

        
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare generation config
        generation_config_dict = self.get_default_generation_config()
        gen_config = GenerationConfig(**generation_config_dict)
        
        # Load model
        self.logger.info("Loading model...")
        
        target_device = config.device    
        # Get model initialization kwargs
        model_init_kwargs = self.get_model_init_kwargs(config)
        model_init_kwargs.update({
            "generation_config": gen_config
        })
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, 
            **model_init_kwargs
        ).eval()
        
        # Move model to target device
        self.logger.info(f"Moving model to device: {target_device}")
        self.model.to(target_device)
        self.device = target_device  # Update device to match actual device used
    
    def _prepare_model_inputs(self, config: BenchmarkConfig):
        """Prepare model inputs. Override in subclasses for custom inputs."""
        # Prepare inputs
        self.inputs = self.tokenizer(self.default_prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        if hasattr(self.model, 'device'):
            # Model is on a single device
            model_device = self.model.device
        else:
            # Model might be distributed, use self.device which was set during model loading
            model_device = self.device
            
        self.inputs = {k: v.to(model_device) for k, v in self.inputs.items()}
        self.logger.debug(f"Moved inputs to device: {model_device}")
    
    def _configure_generation(self, config: BenchmarkConfig):
        """Configure generation settings."""
        seq_length = self.inputs["input_ids"].shape[1]
        self.model.generation_config.max_length = seq_length + config.num_tokens_to_generate
    
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
        generation_kwargs = self._get_generation_kwargs(config, max_new_tokens=1)
        
        # Use CUDA timer for high-precision measurement
        with ArchAwareTimer(device=config.device) as timer:
            # Use SDPA context if specified
            with SDPAContext(config.sdpa_backend, self.logger):
                with torch.no_grad():
                    outputs = model_to_use.generate(**generation_kwargs)
        
        return timer.elapsed_time()
    
    def measure_latency(self, config: BenchmarkConfig) -> TimingResult:
        """Measure full generation latency and compute tokens/sec."""
        model_to_use = self.compiled_model if self.compiled_model is not None else self.model
        
        # Prepare generation kwargs
        generation_kwargs = self._get_generation_kwargs(config, max_new_tokens=config.num_tokens_to_generate)
        
        # Use CUDA timer for high-precision measurement
        with ArchAwareTimer(device=config.device) as timer:
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
            latency_seconds=latency,
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
    
    def _get_generation_kwargs(self, config: BenchmarkConfig, max_new_tokens: int) -> Dict[str, Any]:
        """Get generation kwargs. Override in subclasses for custom generation."""
        generation_config_dict = self.get_default_generation_config()
        generation_kwargs = {
            **self.inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": generation_config_dict.get("do_sample", False),
            "temperature": generation_config_dict.get("temperature", 1.0),
            "top_p": generation_config_dict.get("top_p", 1.0),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Handle static cache for compiled models
        if self.past_key_values is not None and config.variant == "compiled":
            try:
                from transformers import StaticCache
                # Reset cache for each measurement
                seq_length = self.inputs["input_ids"].shape[1]
                
                # Get the actual device the model is on
                if hasattr(self.model, 'device'):
                    cache_device = self.model.device
                else:
                    cache_device = self.device
                
                fresh_cache = StaticCache(
                    config=self.model.config,
                    max_batch_size=config.batch_size,
                    max_cache_len=seq_length + max_new_tokens,
                    device=cache_device,
                    dtype=getattr(torch, config.torch_dtype)
                )
                generation_kwargs["past_key_values"] = fresh_cache
            except (ImportError, TypeError) as e:
                self.logger.debug(f"Fresh StaticCache creation failed: {e}")
                pass
        
        return generation_kwargs


class BenchmarkRunner:
    """Main benchmark runner that coordinates benchmark execution."""
    
    def __init__(self, logger: logging.Logger, output_dir: str = "benchmark_results"):
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        

    def run_benchmark(
        self, 
        benchmark: ModelBenchmark, 
        scenarios: Dict[str, BenchmarkScenario],
        collect_gpu_metrics: bool = True,
        commit_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks using scenarios.
        
        Args:
            benchmark: The benchmark instance to run
            scenarios: Dictionary mapping scenario names to BenchmarkScenario instances
            collect_gpu_metrics: Whether to collect GPU utilization metrics
            commit_id: Git commit ID for metadata (if not provided, will auto-detect from git)
            
        Returns:
            Dictionary mapping scenario names to results with statistics
        """
        all_results = {}
        
        for scenario_name, scenario in scenarios.items():
            self.logger.info(f"Running benchmark scenario: {scenario_name}")
            config = scenario.config
            
            try:
                # Setup model for this configuration
                benchmark.setup_model(config)
                
                # Run scenario setup callbacks
                scenario.setup(benchmark.model, benchmark.tokenizer, self.logger)
                
                # Quick validation: try one measurement first to see if this scenario works
                try:
                    flush_memory()
                    test_result = benchmark.measure_time_to_first_token(config)
                    if test_result is None or test_result <= 0:
                        raise ValueError("Invalid measurement result")
                except Exception as validation_error:
                    self.logger.warning(f"Skipping scenario {scenario_name}: validation failed - {validation_error}")
                    # Clean up and skip this scenario
                    try:
                        scenario.teardown(benchmark.model, benchmark.tokenizer, self.logger)
                        benchmark.cleanup_model()
                    except Exception:
                        pass
                    continue
                
                # Collect metadata
                metadata = BenchmarkMetadata(
                    timestamp=datetime.utcnow().isoformat(),
                    commit_id=commit_id,
                    hardware_info=get_hardware_info(),
                    config=config
                )
                
                # Initialize GPU monitor
                gpu_monitor = None
                if collect_gpu_metrics:
                    gpu_monitor = GPUMonitor(logger=self.logger)
                
                # Warmup runs
                self.logger.info(f"Warming up with {config.warmup_iterations} iterations...")
                warmup_failures = 0
                for i in range(config.warmup_iterations):
                    try:
                        _ = benchmark.measure_latency(config)
                    except Exception as e:
                        warmup_failures += 1
                        self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
                
                # If more than half the warmup iterations failed, skip this scenario
                if warmup_failures > config.warmup_iterations // 2:
                    self.logger.warning(f"Skipping scenario {scenario_name}: too many warmup failures ({warmup_failures}/{config.warmup_iterations})")
                    try:
                        scenario.teardown(benchmark.model, benchmark.tokenizer, self.logger)
                        benchmark.cleanup_model()
                    except Exception:
                        pass
                    continue
                
                # Start GPU monitoring
                if gpu_monitor:
                    gpu_monitor.start()
                
                # Measurement runs for latency
                self.logger.info(f"Measuring latency with {config.measurement_iterations} iterations...")
                latency_measurements = []
                ttft_measurements = []
                tokens_per_sec_measurements = []
                itl_measurements = []  # Inter-Token Latency
                measurement_failures = 0
                
                for i in range(config.measurement_iterations):
                    try:                        
                        # Measure time to first token
                        ttft = benchmark.measure_time_to_first_token(config)
                        ttft_measurements.append(ttft)
                        
                        # Measure full latency
                        timing_result = benchmark.measure_latency(config)
                        latency_measurements.append(timing_result.latency_seconds)
                        
                        if timing_result.tokens_per_second is not None:
                            tokens_per_sec_measurements.append(timing_result.tokens_per_second)
                        
                        if timing_result.time_per_output_token_seconds is not None:
                            itl_measurements.append(timing_result.time_per_output_token_seconds)
                        
                        itl_str = f", itl={timing_result.time_per_output_token_seconds:.4f}s/token" if timing_result.time_per_output_token_seconds else ""
                        self.logger.debug(f"Iteration {i+1}: latency={timing_result.latency_seconds:.4f}s, ttft={ttft:.4f}s{itl_str}")
                        
                    except Exception as e:
                        measurement_failures += 1
                        self.logger.warning(f"Measurement iteration {i+1} failed: {e}")
                
                # Stop GPU monitoring
                gpu_metrics = {}
                if gpu_monitor:
                    gpu_metrics = gpu_monitor.stop_and_collect()
                
                # If we don't have enough successful measurements, skip this scenario
                if not latency_measurements or len(latency_measurements) < config.measurement_iterations // 2:
                    self.logger.warning(f"Skipping scenario {scenario_name}: insufficient successful measurements ({len(latency_measurements)}/{config.measurement_iterations})")
                    try:
                        scenario.teardown(benchmark.model, benchmark.tokenizer, self.logger)
                        benchmark.cleanup_model()
                    except Exception:
                        pass
                    continue
                
                # Calculate statistics
                scenario_results = {
                    "metadata": asdict(metadata),
                    "measurements": {},
                    "gpu_metrics": gpu_metrics,
                    "scenario_description": scenario.description
                }
                
                if latency_measurements:
                    latency_stats = BenchmarkStatistics.from_measurements("latency_seconds", latency_measurements)
                    scenario_results["measurements"]["latency_seconds"] = asdict(latency_stats)
                
                if ttft_measurements:
                    ttft_stats = BenchmarkStatistics.from_measurements("time_to_first_token_seconds", ttft_measurements)
                    scenario_results["measurements"]["time_to_first_token_seconds"] = asdict(ttft_stats)
                
                if tokens_per_sec_measurements:
                    tps_stats = BenchmarkStatistics.from_measurements("tokens_per_second", tokens_per_sec_measurements, "tokens/sec")
                    scenario_results["measurements"]["tokens_per_second"] = asdict(tps_stats)
                
                if itl_measurements:
                    itl_stats = BenchmarkStatistics.from_measurements("time_per_output_token_seconds", itl_measurements, "seconds/token")
                    scenario_results["measurements"]["time_per_output_token_seconds"] = asdict(itl_stats)
                
                # Log summary
                if latency_measurements:
                    self.logger.info(f"Latency: {latency_stats.mean:.4f}±{latency_stats.std:.4f}s (mean±std)")
                if ttft_measurements:
                    self.logger.info(f"TTFT: {ttft_stats.mean:.4f}±{ttft_stats.std:.4f}s (mean±std)")
                if tokens_per_sec_measurements:
                    self.logger.info(f"Throughput: {tps_stats.mean:.2f}±{tps_stats.std:.2f} tokens/sec (mean±std)")
                if itl_measurements:
                    self.logger.info(f"ITL: {itl_stats.mean:.4f}±{itl_stats.std:.4f}s/token (mean±std)")
                
                # Add note about partial results if some measurements failed
                if measurement_failures > 0:
                    scenario_results["warnings"] = [f"Some measurements failed ({measurement_failures} failures)"]
                    self.logger.info(f"Scenario completed with {measurement_failures} measurement failures")
                
                # Run scenario teardown callbacks
                scenario.teardown(benchmark.model, benchmark.tokenizer, self.logger)
                
                # Cleanup model
                benchmark.cleanup_model()
                
                all_results[scenario_name] = scenario_results
                
            except Exception as e:
                self.logger.warning(f"Skipping scenario {scenario_name}: setup failed - {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                
                # Try to clean up if possible
                try:
                    scenario.teardown(benchmark.model, benchmark.tokenizer, self.logger)
                    benchmark.cleanup_model()
                except Exception:
                    pass
                # Skip storing failed scenarios - just continue to the next one
            finally:
                try:
                    scenario.teardown(benchmark.model, benchmark.tokenizer, self.logger)
                    benchmark.cleanup_model()
                except Exception as cleanup_error:
                    self.logger.warning(f"Cleanup failed for scenario {scenario_name}: {cleanup_error}")
                
                flush_memory()
        
        return all_results
    
    def save_results(self, model_name: str, results: Dict[str, Dict[str, Any]]) -> str:
        """Save benchmark results to JSON file."""
        # Create model-specific subdirectory
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_benchmark_{timestamp}.json"
        filepath = os.path.join(model_dir, filename)
        
        # Prepare output structure
        output_data = {
            "model_name": model_name,
            "benchmark_scenarios": []
        }
        
        for config_name, config_results in results.items():
            scenario = {
                "scenario_name": config_name,
                "metadata": config_results["metadata"],
                "measurements": config_results["measurements"],
                "gpu_metrics": config_results.get("gpu_metrics", {})
            }
            output_data["benchmark_scenarios"].append(scenario)
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
        return filepath
 