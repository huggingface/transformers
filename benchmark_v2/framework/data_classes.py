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
    cuda_time: float
    use_cuda_time: bool

    batch_size: int
    new_tokens: int

    time_to_first_token: Optional[float]
    tokens_per_second: float
    time_per_output_token: float

    gpu_metrics: Optional[GPURawMetrics]

    def __init__(
        self, wall_time: float, cuda_time: float, batch_size: int, new_tokens: int, use_cuda_time: bool = False, gpu_metrics: Optional[GPURawMetrics] = None
    ) -> None:
        self.wall_time = wall_time
        self.cuda_time = cuda_time
        self.batch_size = batch_size
        self.new_tokens = new_tokens
        self.use_cuda_time = use_cuda_time
        self.gpu_metrics = gpu_metrics

        self.latency = self.cuda_time if self.use_cuda_time else self.wall_time
        self.time_to_first_token = self.latency if self.new_tokens == 1 else None
        self.tokens_per_second = (self.batch_size * self.new_tokens) / self.latency
        self.time_per_output_token = self.latency / self.new_tokens

    def to_dict(self) -> dict[str, Union[None, int, float]]:
        return {
            "wall_time": self.wall_time,
            "cuda_time": self.cuda_time,
            "batch_size": self.batch_size,
            "new_tokens": self.new_tokens,
            "use_cuda_time": self.use_cuda_time,
            "gpu_metrics": self.gpu_metrics.to_dict() if self.gpu_metrics is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Union[None, int, float]]) -> "TimingResult":
        return cls(
            wall_time=data["wall_time"],
            cuda_time=data["cuda_time"],
            batch_size=data["batch_size"],
            new_tokens=data["new_tokens"],
            use_cuda_time=data["use_cuda_time"],
            gpu_metrics=None if data["gpu_metrics"] is None else GPURawMetrics(data["gpu_metrics"]),
        )
