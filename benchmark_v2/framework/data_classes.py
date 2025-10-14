from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

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
    commit_id: str
    hardware_info: HardwareInfo

    def __init__(self, model_id: str, commit_id: str):
        self.model_id = model_id
        self.timestamp = datetime.utcnow().isoformat()
        self.commit_id = commit_id
        self.hardware_info = HardwareInfo()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "commit_id": self.commit_id,
            "hardware_info": self.hardware_info.to_dict(),
        }


class BenchmarkResult:
    """Result from a series of benchmark runs."""

    def __init__(self) -> None:
        self.e2e_latency = []
        self.token_generation_times = []  # time at which each token was generated (relative to start of the generation)
        self.decoded_outputs = []
        self.gpu_metrics = []

    def accumulate(
        self,
        e2e_latency: float,
        token_generation_times: list[float],
        decoded_output: str,
        gpu_metrics: Optional[GPURawMetrics],
    ) -> None:
        self.e2e_latency.append(e2e_latency)
        self.token_generation_times.append(token_generation_times)
        self.decoded_outputs.append(decoded_output)
        self.gpu_metrics.append(gpu_metrics)

    def to_dict(self) -> dict[str, Union[None, int, float]]:
        # Save GPU metrics as None if it contains only None values
        if all(gm is None for gm in self.gpu_metrics):
            gpu_metrics = None
        else:
            gpu_metrics = [gm.to_dict() for gm in self.gpu_metrics]
        return {
            "e2e_latency": self.e2e_latency,
            "token_generation_times": self.token_generation_times,
            "decoded_outputs": self.decoded_outputs,
            "gpu_metrics": gpu_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Union[None, int, float]]) -> "BenchmarkResult":
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
                decoded_output=data["decoded_output"][i],
                gpu_metrics=gpu_metrics[i],
            )
        return new_instance

    def get_measured_ttft(self) -> list[float]:
        return [dt[0] for dt in self.token_generation_times if len(dt) > 0]

    def get_measured_itl(self) -> list[float]:
        return [(dt[-1] - dt[0]) / (len(dt) - 1) for dt in self.token_generation_times if len(dt) > 1]

    def pprint(self, tabs: int = 0) -> None:
        collated_stats = equalize_lengths_and_collate(
            [
                add_unit_to_duration(compute_basic_statistics(self.e2e_latency)),
                add_unit_to_duration(compute_basic_statistics(self.get_measured_ttft())),
                add_unit_to_duration(compute_basic_statistics(self.get_measured_itl())),
            ]
        )
        pretty_print_dict(
            {
                "E2E Latency": collated_stats[0],
                "Time to First Token": collated_stats[1],
                "Inter-Token Latency": collated_stats[2],
            },
            tabs=tabs,
        )
