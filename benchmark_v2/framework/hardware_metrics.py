import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from logging import Logger

import gpustat
import psutil
import torch


# Data class to hold the hardware information
def get_device_name_and_memory_total() -> tuple[str, float]:
    """Returns the name and memory total of GPU 0."""
    device_name = torch.cuda.get_device_properties(0).name
    device_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return device_name, device_memory_total


class HardwareInfo:
    """A class to hold information about the hardware."""

    def __init__(self) -> None:
        # Retrieve GPU stats
        try:
            self.gpu_name, self.gpu_memory_total_gb = get_device_name_and_memory_total()
        except Exception:
            self.gpu_name, self.gpu_memory_total_gb = None, None
        # Retrieve python, torch and CUDA version
        self.python_version = f"{sys.version.split()[0]}"
        self.torch_version = torch.__version__
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            self.cuda_version = torch.version.cuda
        else:
            self.cuda_version = None
        # Retrieve general hardware information
        self.cpu_count = psutil.cpu_count()
        self.memory_total_mb = int(psutil.virtual_memory().total / (1024 * 1024))

    def to_dict(self) -> dict[str, None | int | float | str]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
        }


# Functions to get information about the GPU
def get_amd_gpu_stats() -> tuple[int, float]:
    """Returns the utilization and memory used of an AMD GPU, both in percent"""
    rocm_smi_output = subprocess.check_output(["rocm-smi", "--json", "--showuse", "--showmeminfo", "VRAM"])
    gpu_stats = json.loads(rocm_smi_output.decode("utf-8"))
    gpu_stats = [
        (card_id, stats["GPU use (%)"], stats["VRAM Total Used Memory (B)"]) for card_id, stats in gpu_stats.items()
    ]
    gpu_stats.sort(key=lambda x: x[1], reverse=True)
    return int(gpu_stats[0][1]), float(gpu_stats[0][2]) / 1024**3


def get_nvidia_gpu_stats() -> tuple[int, float]:
    """Returns the utilization and memory used of an NVIDIA GPU, both in percent"""
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_stats = gpu_stats[0]
    return int(gpu_stats["utilization.gpu"]), float(gpu_stats["memory.used"]) / 1024**3


class GPUStatsCollector:
    """A class to get statistics about the GPU. It serves as a wrapper that holds the GPU total memory and its name,
    which is used to call the right function to get the utilization and memory used."""

    def __init__(self) -> None:
        self.device_name, self.device_memory_total = get_device_name_and_memory_total()
        # Monkey patch the get_utilization_and_memory_used method based on the GPU type
        if "amd" in self.device_name.lower():
            self.get_utilization_and_memory_used = get_amd_gpu_stats
        elif "nvidia" in self.device_name.lower():
            self.get_utilization_and_memory_used = get_nvidia_gpu_stats
        else:
            raise RuntimeError(f"Unsupported GPU: {self.device_name}")

    def get_measurements(self) -> tuple[int, float]:
        """Get the utilization and memory used of the GPU, both in percent"""
        raise NotImplementedError("This method is meant to be monkey patched during __init__")


# Simple data classes to hold the raw GPU metrics
class GPUMonitoringStatus(Enum):
    """Status of GPU monitoring."""

    SUCCESS = "success"
    FAILED = "failed"
    NO_GPUS_AVAILABLE = "no_gpus_available"
    NO_SAMPLES_COLLECTED = "no_samples_collected"


@dataclass
class GPURawMetrics:
    """Raw values for GPU utilization and memory used."""

    utilization: list[float]  # in percent
    memory_used: list[float]  # in GB
    timestamps: list[float]  # in seconds
    timestamp_0: float  # in seconds
    monitoring_status: GPUMonitoringStatus

    def to_dict(self) -> dict[str, None | int | float | str]:
        return {
            "utilization": self.utilization,
            "memory_used": self.memory_used,
            "timestamps": self.timestamps,
            "timestamp_0": self.timestamp_0,
            "monitoring_status": self.monitoring_status.value,
        }


# Main class, used to monitor the GPU utilization during benchmark execution
class GPUMonitor:
    """Monitor GPU utilization during benchmark execution."""

    def __init__(self, sample_interval_sec: float = 0.1, logger: Logger | None = None):
        self.sample_interval_sec = sample_interval_sec
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.num_available_gpus = torch.cuda.device_count()
        if self.num_available_gpus == 0:
            raise RuntimeError("No GPUs detected by torch.cuda.device_count().")
        self.gpu_stats_getter = GPUStatsCollector()

    def start(self):
        """Start monitoring GPU metrics."""
        # Clear the stop event to enable monitoring
        self.stop_event = threading.Event()
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        self.logger.debug("GPU monitoring started")

    def stop_and_collect(self) -> GPURawMetrics:
        """Stop monitoring and return collected metrics."""
        self.stop_event.set()
        self.thread.join()
        if self.gpu_utilization:
            timestamp_0 = self.timestamps[0]
            metrics = GPURawMetrics(
                utilization=self.gpu_utilization,
                memory_used=self.gpu_memory_used,
                timestamps=[t - timestamp_0 for t in self.timestamps],
                timestamp_0=timestamp_0,
                monitoring_status=GPUMonitoringStatus.SUCCESS,
            )
            self.logger.debug(f"GPU monitoring completed: {len(self.gpu_utilization)} samples collected")
        else:
            metrics = GPURawMetrics(monitoring_status=GPUMonitoringStatus.NO_SAMPLES_COLLECTED)
        return metrics

    def _monitor_loop(self):
        """Background monitoring loop using threading.Event for communication."""
        while not self.stop_event.is_set():
            utilization, memory_used = self.gpu_stats_getter.get_utilization_and_memory_used()
            self.gpu_utilization.append(utilization)
            self.gpu_memory_used.append(memory_used)
            self.timestamps.append(time.time())
            if self.stop_event.wait(timeout=self.sample_interval_sec):
                break
