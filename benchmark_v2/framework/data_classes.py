from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

from .hardware_metrics import GPURawMetrics, HardwareInfo


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark scenario."""

    # Basic parameters
    name: str
    # Benchmark parameters
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    gpu_monitoring: bool = True
    # Generation parameters
    batch_size: int = 1
    sequence_length: int = 128
    num_tokens_to_generate: int = 128
    # Attention parameters
    attn_implementation: str = "eager"  # "eager", "sdpa", "flash_attention_2"
    use_cache: bool = True
    sdpa_backend: Optional[str] = None  # None, "math", "flash_attention", "efficient_attention", "cudnn_attention"
    # Compilation parameters
    compilation: bool = False
    compile_mode: Optional[str] = None  # None, "default", "reduce-overhead", "max-autotune"
    compile_options: dict[str, Any] = field(default_factory=dict)
    # Kernels parameters
    kernelize: bool = False

    # CONSTANTS (for now)
    device: str = "cuda"
    dtype: str = "torch.bfloat16"

    def to_dict(self) -> dict[str, Union[None, int, float, str]]:
        return {
            "name": self.name,
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "gpu_monitoring": self.gpu_monitoring,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "num_tokens_to_generate": self.num_tokens_to_generate,
            "attn_implementation": self.attn_implementation,
            "use_cache": self.use_cache,
            "sdpa_backend": self.sdpa_backend,
            "compilation": self.compilation,
            "compile_mode": self.compile_mode,
            "compile_options": self.compile_options,
            "kernelize": self.kernelize,
            "device": self.device,
            "dtype": str(self.dtype),
        }


@dataclass
class BenchmarkMetadata:
    """Metadata collected for each benchmark run."""

    model_id: str
    timestamp: str
    commit_id: str
    hardware_info: HardwareInfo
    config: BenchmarkConfig

    def __init__(self, model_id: str, commit_id: str, config: BenchmarkConfig):
        self.model_id = model_id
        self.timestamp = datetime.utcnow().isoformat()
        self.commit_id = commit_id
        self.hardware_info = HardwareInfo()
        self.config = config

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "commit_id": self.commit_id,
            "hardware_info": self.hardware_info.to_dict(),
            "config": self.config.to_dict(),
        }



class TimingResult:
    """Result from a timing measurement."""

    wall_time: float

    batch_size: int
    new_tokens: int

    time_to_first_token: Optional[float]
    tokens_per_second: float
    time_per_output_token: float

    gpu_metrics: Optional[GPURawMetrics]

    def __init__(
        self,
        wall_time_start: float,
        e2e_latency: float,
        t_tokens: list[float],
        batch_size: int,
        sequence_length: int,
        new_tokens: int,
        gpu_metrics: Optional[GPURawMetrics] = None
    ) -> None:
        self.wall_time_start = wall_time_start
        self.e2e_latency = e2e_latency
        self.t_tokens = t_tokens
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.new_tokens = new_tokens
        self.gpu_metrics = gpu_metrics

        self.time_to_first_token = self.t_tokens[0] - self.wall_time_start
        if len(self.t_tokens) > 1:
            self.inter_token_latency = (self.t_tokens[1] - self.t_tokens[-1]) / (len(self.t_tokens) - 1)
        else:
            self.inter_token_latency = None

    def to_dict(self) -> dict[str, Union[None, int, float]]:
        return {
            "wall_time_start": self.wall_time_start,
            "e2e_latency": self.e2e_latency,
            "t_tokens": self.t_tokens,
            "batch_size": self.batch_size,
            "new_tokens": self.new_tokens,
            "gpu_metrics": self.gpu_metrics.to_dict() if self.gpu_metrics is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Union[None, int, float]]) -> "TimingResult":
        return cls(
            wall_time_start=data["wall_time_start"],
            e2e_latency=data["e2e_latency"],
            t_tokens=data["t_tokens"],
            batch_size=data["batch_size"],
            new_tokens=data["new_tokens"],
            gpu_metrics=None if data["gpu_metrics"] is None else GPURawMetrics(data["gpu_metrics"]),
        )
