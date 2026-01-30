import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from transformers.utils.import_utils import is_cuda_platform, is_rocm_platform


if is_cuda_platform():
    import pynvml

if is_rocm_platform():
    import amdsmi

import psutil
import torch

from transformers.utils import is_torch_accelerator_available


# Data class to hold the hardware information
def get_device_name_and_memory_total() -> tuple[str, float]:
    """Returns the name and memory total of GPU 0."""
    device_type = torch.accelerator.current_accelerator().type if is_torch_accelerator_available() else "cuda"
    torch_accelerator_module = getattr(torch, device_type, torch.cuda)
    device_name = torch_accelerator_module.get_device_properties(0).name
    device_memory_total = torch_accelerator_module.get_device_properties(0).total_memory / 1024**3
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
def get_amd_gpu_stats(device_handle) -> tuple[int, float]:
    """Get AMD GPU stats using amdsmi library."""
    utilization = amdsmi.amdsmi_get_gpu_activity(device_handle)["gfx_activity"]
    memory_used = amdsmi.amdsmi_get_gpu_vram_usage(device_handle)["vram_used"]
    return int(utilization), float(memory_used) / 1024**3  # Convert bytes to GB


def get_intel_xpu_stats() -> tuple[int, float]:
    """Returns the utilization and memory used of an Intel XPU"""
    # xpu-smi outputs CSV format: Timestamp, DeviceId, GPU Memory Utilization (%), GPU Memory Used (MiB)
    xpu_smi_output = subprocess.check_output(["xpu-smi", "dump", "-m", "5,18", "-n", "1"])
    lines = xpu_smi_output.decode("utf-8").strip().split("\n")

    # Parse all data lines (skip header) and collect stats from all cards
    xpu_stats = []
    for line in lines[1:]:
        data_line = line.split(",")
        if len(data_line) < 4:
            continue
        device_id = data_line[1].strip()
        utilization_str = data_line[2].strip()
        memory_used_str = data_line[3].strip()
        if utilization_str != "N/A" and memory_used_str != "N/A":
            utilization = int(float(utilization_str))
            memory_used_mib = float(memory_used_str)
            xpu_stats.append((device_id, utilization, memory_used_mib))

    if not xpu_stats:
        return 0, 0.0

    # Sort by utilization (descending) and pick the highest
    xpu_stats.sort(key=lambda x: x[1], reverse=True)
    device_id, utilization, memory_used_mib = xpu_stats[0]
    memory_used_gb = memory_used_mib / 1024
    return utilization, memory_used_gb


def get_nvidia_gpu_stats(device_handle) -> tuple[int, float]:
    """Returns the utilization and memory used of an NVIDIA GPU using pynvml."""
    utilization = pynvml.nvmlDeviceGetUtilizationRates(device_handle).gpu
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
    memory_used_gb = memory_info.used / 1024**3
    return int(utilization), float(memory_used_gb)


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

    @classmethod
    def from_dict(cls, data: dict[str, None | int | float | str]) -> "GPURawMetrics":
        """Create a GPURawMetrics instance from a dictionary."""
        return cls(
            utilization=data["utilization"],
            memory_used=data["memory_used"],
            timestamps=data["timestamps"],
            timestamp_0=data["timestamp_0"],
            monitoring_status=GPUMonitoringStatus(data["monitoring_status"]),
        )


# Main class, used to monitor the GPU utilization during benchmark execution
class GPUMonitor:
    """Monitor GPU utilization during benchmark execution using a separate process."""

    def __init__(self, sample_interval_sec: float = 0.05, logger: Logger | None = None):
        self.sample_interval_sec = sample_interval_sec
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.gpu_type = None
        self.process = None

        device_type = torch.accelerator.current_accelerator().type if is_torch_accelerator_available() else "cuda"
        torch_accelerator_module = getattr(torch, device_type, torch.cuda)
        self.num_available_gpus = torch_accelerator_module.device_count()
        if self.num_available_gpus == 0:
            self.logger.warning(f"No GPUs detected by torch.{device_type}.device_count().")
            return

        # Determine GPU type
        device_name, _ = get_device_name_and_memory_total()
        if "amd" in device_name.lower():
            self.gpu_type = "amd"
        elif "nvidia" in device_name.lower():
            self.gpu_type = "nvidia"
        elif "intel" in device_name.lower() or device_type == "xpu":
            self.gpu_type = "intel"
        else:
            self.logger.warning(f"Unsupported GPU for monitoring: {device_name}")

    @staticmethod
    def _monitor_worker(gpu_type: str, sample_interval_sec: float, connection: Connection):
        """Worker process for GPU monitoring."""
        gpu_utilization = []
        gpu_memory_used = []
        timestamps = []
        device_handle = None

        # Initialize GPU-specific monitoring
        if gpu_type == "amd":
            amdsmi.amdsmi_init()
            device_handle = amdsmi.amdsmi_get_processor_handles()[0]
        elif gpu_type == "nvidia":
            pynvml.nvmlInit()
            device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Signal ready
        try:
            connection.send(0)
        except Exception:
            return

        # Monitoring loop
        stop = False
        while not stop:
            try:
                if gpu_type == "amd":
                    utilization, memory_used = get_amd_gpu_stats(device_handle)
                elif gpu_type == "nvidia":
                    utilization, memory_used = get_nvidia_gpu_stats(device_handle)
                elif gpu_type == "intel":
                    utilization, memory_used = get_intel_xpu_stats()
                else:
                    break

                gpu_utilization.append(utilization)
                gpu_memory_used.append(memory_used)
                timestamps.append(time.time())
            except Exception:
                pass  # Skip failed measurements

            stop = connection.poll(sample_interval_sec)

        # Cleanup
        if gpu_type == "amd":
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass
        elif gpu_type == "nvidia":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        # Send results back
        try:
            connection.send((gpu_utilization, gpu_memory_used, timestamps))
        except Exception:
            pass

        connection.close()

    def start(self):
        """Start monitoring GPU metrics in a separate process."""
        if self.gpu_type is None:
            self.logger.debug("GPU monitoring skipped (no supported GPU)")
            return

        self.child_connection, self.parent_connection = Pipe()
        self.process = Process(
            target=GPUMonitor._monitor_worker,
            args=(self.gpu_type, self.sample_interval_sec, self.child_connection),
            daemon=True,
        )
        self.process.start()

        # Wait for worker to signal ready
        if self.process.is_alive():
            self.parent_connection.recv()
        self.logger.debug("GPU monitoring started (multiprocessing)")

    def stop_and_collect(self) -> GPURawMetrics:
        """Stop monitoring and return collected metrics."""
        # No GPU available or unsupported GPU
        if self.process is None:
            return GPURawMetrics(
                utilization=[],
                memory_used=[],
                timestamps=[],
                timestamp_0=0.0,
                monitoring_status=GPUMonitoringStatus.NO_GPUS_AVAILABLE,
            )

        # Process crashed before we could collect results
        process_failed = False
        if not self.process.is_alive():
            process_failed = True
            gpu_utilization, gpu_memory_used, timestamps = [], [], []
        else:
            # Signal stop
            self.parent_connection.send(0)
            # Get results
            try:
                gpu_utilization, gpu_memory_used, timestamps = self.parent_connection.recv()
            except Exception:
                process_failed = True
                gpu_utilization, gpu_memory_used, timestamps = [], [], []

        self.parent_connection.close()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()

        if gpu_utilization:
            timestamp_0 = timestamps[0]
            metrics = GPURawMetrics(
                utilization=gpu_utilization,
                memory_used=gpu_memory_used,
                timestamps=[t - timestamp_0 for t in timestamps],
                timestamp_0=timestamp_0,
                monitoring_status=GPUMonitoringStatus.SUCCESS,
            )
            self.logger.debug(f"GPU monitoring completed: {len(gpu_utilization)} samples collected")
        elif process_failed:
            metrics = GPURawMetrics(
                utilization=[],
                memory_used=[],
                timestamps=[],
                timestamp_0=0.0,
                monitoring_status=GPUMonitoringStatus.FAILED,
            )
            self.logger.warning("GPU monitoring failed (process crashed or timed out)")
        else:
            metrics = GPURawMetrics(
                utilization=[],
                memory_used=[],
                timestamps=[],
                timestamp_0=0.0,
                monitoring_status=GPUMonitoringStatus.NO_SAMPLES_COLLECTED,
            )
        return metrics
