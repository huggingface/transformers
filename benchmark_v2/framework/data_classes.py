from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .hardware_metrics import GPURawMetrics, HardwareInfo


def compute_basic_statistics(measurements: list[float]) -> dict[str, float]:
    return {
        "avg": np.mean(measurements),
        "std": np.std(measurements),
        "min": np.min(measurements),
        "med": np.median(measurements),
        "max": np.max(measurements),
        "p95": np.percentile(measurements, 95),
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


def equalize_lengths_and_collate(stats: list[dict[str, str]]) -> list[str]:
    keys = ["avg", "std", "min", "med", "max", "p95"]
    for key in keys:
        max_length = max(len(stat[key]) for stat in stats)
        for stat in stats:
            stat[key] = stat[key].ljust(max_length, " ")
    return [" ".join([f"{key}={stat[key]}" for key in keys]) for stat in stats]


def pretty_print_dict(data: dict[str, Any], tabs: int = 0) -> None:
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

    def __init__(self, model_id: str, commit_id: str, branch_name: str = "main", commit_message: str = "") -> None:
        self.model_id = model_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.branch_name = branch_name
        self.commit_id = commit_id
        self.commit_message = commit_message
        self.hardware_info = HardwareInfo()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "branch_name": self.branch_name,
            "commit_id": self.commit_id,
            "commit_message": self.commit_message,
            "hardware_info": self.hardware_info.to_dict(),
        }


class BenchmarkResult:
    """Result from a series of benchmark runs."""

    def __init__(self) -> None:
        self.e2e_latency = []
        self.token_generation_times = []  # time at which each token was generated (relative to start of the generation)
        self.shape_and_decoded_outputs = []
        self.gpu_metrics = []

    def accumulate(
        self,
        e2e_latency: float,
        token_generation_times: list[float],
        shape_and_decoded_output: str,
        gpu_metrics: GPURawMetrics | None,
    ) -> None:
        self.e2e_latency.append(e2e_latency)
        self.token_generation_times.append(token_generation_times)
        self.shape_and_decoded_outputs.append(shape_and_decoded_output)
        self.gpu_metrics.append(gpu_metrics)

    def to_dict(self) -> dict[str, None | int | float]:
        # Save GPU metrics as None if it contains only None values
        if all(gm is None for gm in self.gpu_metrics):
            gpu_metrics = None
        else:
            gpu_metrics = [gm.to_dict() for gm in self.gpu_metrics]
        return {
            "e2e_latency": self.e2e_latency,
            "token_generation_times": self.token_generation_times,
            "shape_and_decoded_outputs": self.shape_and_decoded_outputs,
            "gpu_metrics": gpu_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, None | int | float]) -> "BenchmarkResult":
        # Handle GPU metrics, which is saved as None if it contains only None values
        if data["gpu_metrics"] is None:
            gpu_metrics = [None for _ in range(len(data["e2e_latency"]))]
        else:
            gpu_metrics = [GPURawMetrics.from_dict(gm) for gm in data["gpu_metrics"]]
        # Create a new instance and accumulate the data
        new_instance = cls()
        for i in range(len(data["e2e_latency"])):
            new_instance.accumulate(
                e2e_latency=data["e2e_latency"][i],
                token_generation_times=data["token_generation_times"][i],
                shape_and_decoded_output=data["shape_and_decoded_outputs"][i],
                gpu_metrics=gpu_metrics[i],
            )
        return new_instance

    def get_measured_ttft(self) -> list[float]:
        return [dt[0] for dt in self.token_generation_times if len(dt) > 0]

    def get_measured_itl(self) -> list[float]:
        return [(dt[-1] - dt[0]) / (len(dt) - 1) for dt in self.token_generation_times if len(dt) > 1]

    def get_throughput(self, batch_size: int) -> float:
        return [
            batch_size * len(dt) / e2e_latency
            for e2e_latency, dt in zip(self.e2e_latency, self.token_generation_times)
        ]

    def pprint(self, batch_size: int = 0, tabs: int = 0) -> None:
        stats_to_collate = [
            add_unit_to_duration(compute_basic_statistics(self.e2e_latency)),
            add_unit_to_duration(compute_basic_statistics(self.get_measured_ttft())),
            add_unit_to_duration(compute_basic_statistics(self.get_measured_itl())),
        ]
        if batch_size > 0:
            throughput_stats = compute_basic_statistics(self.get_throughput(batch_size))
            stats_to_collate.append({key: f"{value:.2f}tok/s" for key, value in throughput_stats.items()})
        collated_stats = equalize_lengths_and_collate(stats_to_collate)
        dict_to_pprint = {
            "E2E Latency": collated_stats[0],
            "Time to First Token": collated_stats[1],
            "Inter-Token Latency": collated_stats[2],
        }
        if batch_size > 0:
            dict_to_pprint["Throughput"] = collated_stats[3]
        pretty_print_dict(dict_to_pprint, tabs=tabs)
