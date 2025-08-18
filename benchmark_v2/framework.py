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
from typing import Any, Callable, Dict, List, Optional, Union
import logging

import numpy as np
import psutil
import gpustat

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


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
    generation_config: Dict[str, Any] = field(default_factory=dict)
    attn_implementation: str = "sdpa"  # "eager", "sdpa", "flash_attention_2"
    sdpa_backend: Optional[str] = None  # None, "math", "flash_attention", "efficient_attention", "cudnn_attention"
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generation_config:
            self.generation_config = {
                "do_sample": False,
                "top_p": 1.0,
                "temperature": 1.0
            }


@dataclass
class TimingResult:
    """Result from a timing measurement."""
    time_to_first_token: Optional[float] = None
    latency: float = 0.0
    tokens_per_second: Optional[float] = None
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
    gpu_memory_total: int  # MB
    cpu_count: int
    memory_total: int  # MB
    python_version: str
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class BenchmarkMetadata:
    """Metadata collected for each benchmark run."""
    timestamp: str
    commit_id: str
    repository: str
    branch: str
    hardware_info: HardwareInfo
    config: BenchmarkConfig


class GPUMonitor:
    """Monitor GPU utilization during benchmark execution."""
    
    def __init__(self, sample_interval: float = 0.1, logger: logging.Logger = None):
        self.sample_interval = sample_interval
        self.logger = logger or logging.getLogger(__name__)
        self.monitoring = False
        self.thread = None
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.timestamps = []
        self.gpu_available = False
        self.error_logged = False
        
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
            
        self.monitoring = True
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        self.logger.debug("GPU monitoring started")
        
    def stop(self):
        """Stop monitoring and return collected metrics."""
        if not self.gpu_available:
            return {
                "gpu_monitoring_status": "disabled",
                "gpu_monitoring_reason": "no_gpus_available"
            }
            
        self.monitoring = False
        if self.thread:
            self.thread.join()
        
        if self.gpu_utilization:
            metrics = {
                "gpu_utilization_mean": statistics.mean(self.gpu_utilization),
                "gpu_utilization_max": max(self.gpu_utilization),
                "gpu_utilization_min": min(self.gpu_utilization),
                "gpu_memory_used_mean": statistics.mean(self.gpu_memory_used),
                "gpu_memory_used_max": max(self.gpu_memory_used),
                "gpu_memory_used_min": min(self.gpu_memory_used),
                "sample_count": len(self.gpu_utilization),
                "gpu_monitoring_status": "success"
            }
            self.logger.debug(f"GPU monitoring completed: {len(self.gpu_utilization)} samples collected")
            return metrics
        else:
            return {
                "gpu_monitoring_status": "failed",
                "gpu_monitoring_reason": "no_samples_collected"
            }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.monitoring:
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
                    if consecutive_failures >= max_consecutive_failures and not self.error_logged:
                        self.logger.warning("GPU monitoring: No GPU data returned by gpustat")
                        self.error_logged = True
                        
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures and not self.error_logged:
                    self.logger.warning(f"GPU monitoring failed after {max_consecutive_failures} attempts: {e}")
                    self.error_logged = True
                    
            time.sleep(self.sample_interval)


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
    
    torch_version = None
    cuda_version = None
    if TORCH_AVAILABLE and hasattr(torch, '__version__'):
        torch_version = torch.__version__
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            cuda_version = torch.version.cuda
    
    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_memory_total=gpu_memory_total,
        cpu_count=psutil.cpu_count(),
        memory_total=int(psutil.virtual_memory().total / (1024 * 1024)),  # MB
        python_version=f"{sys.version.split()[0]}",
        torch_version=torch_version,
        cuda_version=cuda_version
    )


def get_git_commit_id() -> str:
    """Get current git commit ID."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def flush_memory():
    """Flush GPU memory and run garbage collection."""
    gc.collect()
    if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_sdpa_backend(backend_name: Optional[str]):
    """Get the SDPA backend enum from string name."""
    if not TORCH_AVAILABLE:
        return None
    
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


def get_available_sdpa_backends():
    """Get list of available SDPA backends on this system."""
    if not TORCH_AVAILABLE:
        return []
    
    try:
        backends = []
        backend_names = ["math", "flash_attention", "efficient_attention", "cudnn_attention"]
        
        for name in backend_names:
            backend = get_sdpa_backend(name)
            if backend is not None:
                backends.append(name)
        
        return backends
    except Exception:
        return []


class SDPAContext:
    """Context manager for SDPA kernel selection."""
    
    def __init__(self, backend_name: Optional[str], logger: logging.Logger = None):
        self.backend_name = backend_name
        self.logger = logger or logging.getLogger(__name__)
        self.backend = get_sdpa_backend(backend_name) if backend_name else None
        self.context = None
        
    def __enter__(self):
        if self.backend is not None and TORCH_AVAILABLE:
            try:
                self.context = torch.nn.attention.sdpa_kernel(self.backend)
                self.context.__enter__()
                if self.logger:
                    self.logger.debug(f"Using SDPA backend: {self.backend_name}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to set SDPA backend {self.backend_name}: {e}")
                self.context = None
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error exiting SDPA context: {e}")
        return False


class ModelBenchmark(ABC):
    """Abstract base class for model benchmarks."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.device = None
        
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


class BenchmarkRunner:
    """Main benchmark runner that coordinates benchmark execution."""
    
    def __init__(self, logger: logging.Logger, output_dir: str = "benchmark_results"):
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_benchmark(
        self, 
        benchmark: ModelBenchmark, 
        configs: List[BenchmarkConfig],
        collect_gpu_metrics: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks across multiple configurations.
        
        Args:
            benchmark: The benchmark instance to run
            configs: List of configurations to test
            collect_gpu_metrics: Whether to collect GPU utilization metrics
            
        Returns:
            Dictionary mapping config names to results with statistics
        """
        all_results = {}
        
        for config in configs:
            self.logger.info(f"Running benchmark for config: {config.name}")
            
            try:
                # Setup model for this configuration
                benchmark.setup_model(config)
                
                # Collect metadata
                metadata = BenchmarkMetadata(
                    timestamp=datetime.utcnow().isoformat(),
                    commit_id=get_git_commit_id(),
                    repository="transformers",  # TODO: make configurable
                    branch="main",  # TODO: make configurable
                    hardware_info=get_hardware_info(),
                    config=config
                )
                
                # Initialize GPU monitor
                gpu_monitor = None
                if collect_gpu_metrics:
                    gpu_monitor = GPUMonitor(logger=self.logger)
                
                # Warmup runs
                self.logger.info(f"Warming up with {config.warmup_iterations} iterations...")
                for i in range(config.warmup_iterations):
                    try:
                        flush_memory()
                        _ = benchmark.measure_latency(config)
                    except Exception as e:
                        self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
                
                # Start GPU monitoring
                if gpu_monitor:
                    gpu_monitor.start()
                
                # Measurement runs for latency
                self.logger.info(f"Measuring latency with {config.measurement_iterations} iterations...")
                latency_measurements = []
                ttft_measurements = []
                tokens_per_sec_measurements = []
                
                for i in range(config.measurement_iterations):
                    try:
                        flush_memory()
                        
                        # Measure time to first token
                        ttft = benchmark.measure_time_to_first_token(config)
                        ttft_measurements.append(ttft)
                        
                        # Measure full latency
                        timing_result = benchmark.measure_latency(config)
                        latency_measurements.append(timing_result.latency)
                        
                        if timing_result.tokens_per_second is not None:
                            tokens_per_sec_measurements.append(timing_result.tokens_per_second)
                        
                        self.logger.debug(f"Iteration {i+1}: latency={timing_result.latency:.4f}s, ttft={ttft:.4f}s")
                        
                    except Exception as e:
                        self.logger.warning(f"Measurement iteration {i+1} failed: {e}")
                
                # Stop GPU monitoring
                gpu_metrics = {}
                if gpu_monitor:
                    gpu_metrics = gpu_monitor.stop()
                
                # Calculate statistics
                config_results = {
                    "metadata": asdict(metadata),
                    "measurements": {},
                    "gpu_metrics": gpu_metrics
                }
                
                if latency_measurements:
                    latency_stats = BenchmarkStatistics.from_measurements("latency", latency_measurements)
                    config_results["measurements"]["latency"] = asdict(latency_stats)
                
                if ttft_measurements:
                    ttft_stats = BenchmarkStatistics.from_measurements("time_to_first_token", ttft_measurements)
                    config_results["measurements"]["time_to_first_token"] = asdict(ttft_stats)
                
                if tokens_per_sec_measurements:
                    tps_stats = BenchmarkStatistics.from_measurements("tokens_per_second", tokens_per_sec_measurements, "tokens/sec")
                    config_results["measurements"]["tokens_per_second"] = asdict(tps_stats)
                
                # Log summary
                if latency_measurements:
                    self.logger.info(f"Latency: {latency_stats.mean:.4f}±{latency_stats.std:.4f}s (mean±std)")
                if ttft_measurements:
                    self.logger.info(f"TTFT: {ttft_stats.mean:.4f}±{ttft_stats.std:.4f}s (mean±std)")
                if tokens_per_sec_measurements:
                    self.logger.info(f"Throughput: {tps_stats.mean:.2f}±{tps_stats.std:.2f} tokens/sec (mean±std)")
                
                # Cleanup model
                benchmark.cleanup_model()
                
                all_results[config.name] = config_results
                
            except Exception as e:
                self.logger.error(f"Failed to run benchmark for config {config.name}: {e}")
                benchmark.cleanup_model()
        
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


def create_config_variants(
    base_config: BenchmarkConfig,
    variants: Dict[str, List[Any]]
) -> List[BenchmarkConfig]:
    """
    Create multiple configuration variants from a base configuration.
    
    Args:
        base_config: Base configuration to vary
        variants: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of configuration variants
    """
    import itertools
    from copy import deepcopy
    
    # Get all combinations of variant parameters
    param_names = list(variants.keys())
    param_values = list(variants.values())
    
    configs = []
    for combination in itertools.product(*param_values):
        # Create a new config from the base
        config = deepcopy(base_config)
        
        # Apply this combination of parameters
        name_parts = [base_config.name]
        for param_name, param_value in zip(param_names, combination):
            setattr(config, param_name, param_value)
            name_parts.append(f"{param_name}={param_value}")
        
        # Update the config name
        config.name = "_".join(name_parts)
        configs.append(config)
    
    return configs 