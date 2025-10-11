from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

from .hardware_metrics import GPURawMetrics, HardwareInfo


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
        self.wall_time_start = []
        self.e2e_latency = []
        self.dt_tokens = []
        self.decoded_outputs = []
        self.gpu_metrics = []

    def accumulate(
        self,
        wall_time_0: float,
        e2e_latency: float,
        dt_tokens: list[float],
        decoded_output: str,
        gpu_metrics: Optional[GPURawMetrics],
    ) -> None:
        self.wall_time_start.append(wall_time_0)
        self.e2e_latency.append(e2e_latency)
        self.dt_tokens.append(dt_tokens)
        self.decoded_outputs.append(decoded_output)
        self.gpu_metrics.append(gpu_metrics)

    def to_dict(self) -> dict[str, Union[None, int, float]]:
        # Save GPU metrics as None if it contains only None values
        if all(gm is None for gm in self.gpu_metrics):
            gpu_metrics = None
        else:
            gpu_metrics = [gm.to_dict() for gm in self.gpu_metrics]
        return {
            "wall_time_start": self.wall_time_start,
            "e2e_latency": self.e2e_latency,
            "dt_tokens": self.dt_tokens,
            "decoded_outputs": self.decoded_outputs,
            "gpu_metrics": gpu_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Union[None, int, float]]) -> "BenchmarkResult":
        # Handle GPU metrics, which is saved as None if it contains only None values
        if data["gpu_metrics"] is None:
            gpu_metrics = [None for _ in range(len(data["wall_time_start"]))]
        else:
            gpu_metrics = [GPURawMetrics.from_dict(gm) for gm in data["gpu_metrics"]]
        # Create a new instance and accumulate the data
        new_instance = cls()
        for i in range(len(data["wall_time_start"])):
            new_instance.accumulate(
                wall_time_start=data["wall_time_start"][i],
                e2e_latency=data["e2e_latency"][i],
                dt_tokens=data["dt_tokens"][i],
                decoded_output=data["decoded_output"][i],
                gpu_metrics=gpu_metrics[i],
            )
        return new_instance
