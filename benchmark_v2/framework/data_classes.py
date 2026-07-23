from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .hardware_metrics import GPURawMetrics, HardwareInfo


def compute_basic_statistics(measurements: list[float]) -> dict[str, float]:
    return {
        "avg": np.mean(measurements) if measurements else 0,
        "std": np.std(measurements) if measurements else 0,
        "min": np.min(measurements) if measurements else 0,
        "med": np.median(measurements) if measurements else 0,
        "max": np.max(measurements) if measurements else 0,
        "p95": np.percentile(measurements, 95) if measurements else 0,
    }


def add_unit_to_duration(stats: dict[str, float]) -> dict[str, str]:
    for key in list(stats.keys()):
        value = stats[key]
        if value > 3600:
            stats[key] = f"{(value / 3600):.2f}hr"
        elif value > 60:
            stats[key] = f"{(value / 60):.2f}min"
        elif value > 1:
            stats[key] = f"{value:.2f}s"
        elif value > 1e-3:
            stats[key] = f"{(value * 1e3):.2f}ms"
        elif value > 1e-6:
            stats[key] = f"{(value * 1e6):.2f}us"
        else:
            stats[key] = f"{(value * 1e9):.2f}ns"
    return stats


def equalize_lengths_and_collate(stats: dict[str, dict[str, str]]) -> dict[str, str]:
    """Note: This operation is destructive as it will update values in place before returning a new correctly formatted dict"""
    keys = ["avg", "std", "min", "med", "max", "p95"]
    for key in keys:
        max_length = max(len(stat[key]) for stat in stats.values())
        for stat in stats.values():
            stat[key] = stat[key].ljust(max_length, " ")
    return {name: " ".join([f"{key}={stat[key]}" for key in keys]) for name, stat in stats.items()}


def pretty_print_dict(data: dict[str, str], tabs: int = 0) -> None:
    max_key_length = max([len(key) for key in data.keys()])
    for key, value in data.items():
        tabs_str = "  " * tabs
        padded_key = key.ljust(max_key_length + 1, ".")
        print(f"{tabs_str}{padded_key}: {value}")


@dataclass
class BenchmarkMetadata:
    """Metadata collected for each benchmark run."""

    model_id: str
    timestamp: str
    branch_name: str
    commit_id: str
    commit_message: str
    hardware_info: HardwareInfo
    success: bool

    def __init__(
        self, model_id: str, commit_id: str, branch_name: str = "main", commit_message: str = "", success: bool = True
    ) -> None:
        self.model_id = model_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.branch_name = branch_name
        self.commit_id = commit_id
        self.commit_message = commit_message
        self.hardware_info = HardwareInfo()
        self.success = success

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "branch_name": self.branch_name,
            "commit_id": self.commit_id,
            "commit_message": self.commit_message,
            "hardware_info": self.hardware_info.to_dict(),
            "success": self.success,
        }


class BenchmarkResult:
    """Result from a series of benchmark runs."""

    def __init__(self) -> None:
        self.e2e_latency = []
        self._timestamps = []
        self.time_to_first_token = []
        self.inter_token_latency = []
        self.shape_and_decoded_outputs = []
        self.gpu_metrics = []

    def accumulate(
        self,
        e2e_latency: float,
        timestamps: list[float],
        shape_and_decoded_output: str,
        gpu_metrics: GPURawMetrics | None,
    ) -> None:
        self.e2e_latency.append(e2e_latency)
        self._timestamps.append(timestamps)
        self._accumulate_ttft_and_itl(timestamps)
        self.shape_and_decoded_outputs.append(shape_and_decoded_output)
        self.gpu_metrics.append(gpu_metrics)

    def _accumulate_ttft_and_itl(self, timestamps: list[float]) -> None:
        timestamps = np.array(timestamps)
        tftt = np.min(timestamps[:, 0])
        itl = np.mean(timestamps[:, -1] - timestamps[:, 0]) / (timestamps.shape[1] - 1)
        self.time_to_first_token.append(tftt)
        self.inter_token_latency.append(itl)

    def to_dict(self, summarized: bool = False) -> dict[str, Any]:
        # Save GPU metrics as None if it contains only None values or if we are summarizing
        if summarized or all(gm is None for gm in self.gpu_metrics):
            gpu_metrics = None
        else:
            gpu_metrics = [gm.to_dict() for gm in self.gpu_metrics]
        return {
            "e2e_latency": self.e2e_latency,
            "time_to_first_token": self.time_to_first_token,
            "inter_token_latency": self.inter_token_latency,
            "shape_and_decoded_outputs": self.shape_and_decoded_outputs,
            "gpu_metrics": gpu_metrics,
            "timestamps": None if summarized else self._timestamps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        # Handle GPU metrics, which is saved as None if it contains only None values
        if data["gpu_metrics"] is None:
            gpu_metrics = [None for _ in range(len(data["e2e_latency"]))]
        else:
            gpu_metrics = [GPURawMetrics.from_dict(gm) for gm in data["gpu_metrics"]]
        # Handle timestamps, which can be saved as None to reduce file size
        if data["timestamps"] is None:
            timestamps = [None for _ in range(len(data["e2e_latency"]))]
        else:
            timestamps = data["timestamps"]
        # Create a new instance and accumulate the data
        new_instance = cls()
        new_instance.e2e_latency = data["e2e_latency"]
        new_instance._timestamps = timestamps
        new_instance.time_to_first_token = data["time_to_first_token"]
        new_instance.inter_token_latency = data["inter_token_latency"]
        new_instance.shape_and_decoded_outputs = data["shape_and_decoded_outputs"]
        new_instance.gpu_metrics = gpu_metrics
        return new_instance

    def get_throughput(self, total_generated_tokens: int) -> list[float]:
        return [total_generated_tokens / e2e_latency for e2e_latency in self.e2e_latency]

    def pprint(self, batch_size: int = 0, num_generated_tokens: int = 0, tabs: int = 0) -> None:
        measurements = {
            "E2E Latency": add_unit_to_duration(compute_basic_statistics(self.e2e_latency)),
            "Time to First Token": add_unit_to_duration(compute_basic_statistics(self.time_to_first_token)),
        }
        if len(self.inter_token_latency) > 0:
            measurements["Inter-Token Latency"] = add_unit_to_duration(
                compute_basic_statistics(self.inter_token_latency)
            )
        if batch_size > 0:
            throughput_stats = compute_basic_statistics(self.get_throughput(batch_size * num_generated_tokens))
            measurements["Throughput"] = {key: f"{value:.2f}tok/s" for key, value in throughput_stats.items()}
        dict_to_pprint = equalize_lengths_and_collate(measurements)
        pretty_print_dict(dict_to_pprint, tabs=tabs)
