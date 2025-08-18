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


class CUDATimer:
    """CUDA event-based timer for supposedly better prescision"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize CUDA timer.
        
        Args:
            device: CUDA device to use. If None, uses current device.
        """
        self.device = device
        self.use_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        
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
    generation_config: Dict[str, Any] = field(default_factory=dict)
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
        import torch
        available_backends = []
        
        # Test actual availability by trying to use SDPA with different backends
        # Create small test tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        query = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
        key = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
        value = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
        
        # Test math backend (should always work)
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                torch.nn.functional.scaled_dot_product_attention(query, key, value)
            available_backends.append("math")
        except Exception:
            pass
        
        # Test flash attention backend
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(query, key, value)
            available_backends.append("flash_attention")
        except Exception:
            pass
        
        # Test efficient attention backend
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(query, key, value)
            available_backends.append("efficient_attention")
        except Exception:
            pass
        
        # Test cuDNN attention backend
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(query, key, value)
            available_backends.append("cudnn_attention")
        except Exception:
            pass
        
        return available_backends
        
    except Exception:
        return []


class SDPAContext:
    """Context manager for SDPA kernel selection."""
    
    def __init__(self, backend_name: Optional[str], logger: logging.Logger = None):
        self.backend_name = backend_name
        self.logger = logger or logging.getLogger(__name__)
        self.backend = get_sdpa_backend(backend_name) if backend_name else None
        self.context = None
        self.available_backends = None
        
    def __enter__(self):
        if self.backend is not None and TORCH_AVAILABLE:
            try:
                # Check if this backend is actually available before using it
                if self.available_backends is None:
                    self.available_backends = get_available_sdpa_backends()
                
                if self.backend_name not in self.available_backends:
                    if self.logger:
                        self.logger.warning(f"SDPA backend '{self.backend_name}' not available, using default")
                    return self
                
                self.context = torch.nn.attention.sdpa_kernel(self.backend)
                self.context.__enter__()
                if self.logger:
                    self.logger.debug(f"Using SDPA backend: {self.backend_name}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to set SDPA backend {self.backend_name}: {e}")
                self.context = None
        elif self.backend_name and self.logger:
            self.logger.debug(f"SDPA backend '{self.backend_name}' requested but not using kernel context (backend={self.backend}, torch={TORCH_AVAILABLE})")
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
        collect_gpu_metrics: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks using scenarios.
        
        Args:
            benchmark: The benchmark instance to run
            scenarios: Dictionary mapping scenario names to BenchmarkScenario instances
            collect_gpu_metrics: Whether to collect GPU utilization metrics
            
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
                warmup_failures = 0
                for i in range(config.warmup_iterations):
                    try:
                        flush_memory()
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
                tpot_measurements = []  # Time Per Output Token
                measurement_failures = 0
                
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
                        
                        if timing_result.time_per_output_token_seconds is not None:
                            tpot_measurements.append(timing_result.time_per_output_token_seconds)
                        
                        tpot_str = f", tpot={timing_result.time_per_output_token_seconds:.4f}s/token" if timing_result.time_per_output_token_seconds else ""
                        self.logger.debug(f"Iteration {i+1}: latency={timing_result.latency:.4f}s, ttft={ttft:.4f}s{tpot_str}")
                        
                    except Exception as e:
                        measurement_failures += 1
                        self.logger.warning(f"Measurement iteration {i+1} failed: {e}")
                
                # Stop GPU monitoring
                gpu_metrics = {}
                if gpu_monitor:
                    gpu_metrics = gpu_monitor.stop()
                
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
                    latency_stats = BenchmarkStatistics.from_measurements("latency", latency_measurements)
                    scenario_results["measurements"]["latency"] = asdict(latency_stats)
                
                if ttft_measurements:
                    ttft_stats = BenchmarkStatistics.from_measurements("time_to_first_token", ttft_measurements)
                    scenario_results["measurements"]["time_to_first_token"] = asdict(ttft_stats)
                
                if tokens_per_sec_measurements:
                    tps_stats = BenchmarkStatistics.from_measurements("tokens_per_second", tokens_per_sec_measurements, "tokens/sec")
                    scenario_results["measurements"]["tokens_per_second"] = asdict(tps_stats)
                
                if tpot_measurements:
                    tpot_stats = BenchmarkStatistics.from_measurements("time_per_output_token_seconds", tpot_measurements, "seconds/token")
                    scenario_results["measurements"]["time_per_output_token_seconds"] = asdict(tpot_stats)
                
                # Log summary
                if latency_measurements:
                    self.logger.info(f"Latency: {latency_stats.mean:.4f}±{latency_stats.std:.4f}s (mean±std)")
                if ttft_measurements:
                    self.logger.info(f"TTFT: {ttft_stats.mean:.4f}±{ttft_stats.std:.4f}s (mean±std)")
                if tokens_per_sec_measurements:
                    self.logger.info(f"Throughput: {tps_stats.mean:.2f}±{tps_stats.std:.2f} tokens/sec (mean±std)")
                if tpot_measurements:
                    self.logger.info(f"TPOT: {tpot_stats.mean:.4f}±{tpot_stats.std:.4f}s/token (mean±std)")
                
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


 