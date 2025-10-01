import hashlib
import json
from typing import Any, Optional, Union


KERNELIZATION_AVAILABLE = False
try:
    from kernels import Mode, kernelize  # noqa: F401

    KERNELIZATION_AVAILABLE = True
except ImportError:
    pass


class BenchmarkConfig:
    """Configuration for a single benchmark scenario."""

    def __init__(
        self,
        warmup_iterations: int = 5,
        measurement_iterations: int = 20,
        gpu_monitoring: bool = True,
        batch_size: int = 1,
        sequence_length: int = 128,
        num_tokens_to_generate: int = 128,
        attn_implementation: str = "eager",
        sdpa_backend: Optional[str] = None,
        compile_mode: Optional[str] = None,
        compile_options: Optional[dict[str, Any]] = None,
        kernelize: bool = False,
        name: Optional[str] = None,
    ) -> None:
        # Benchmark parameters
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.gpu_monitoring = gpu_monitoring
        # Input parameters
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_tokens_to_generate = num_tokens_to_generate
        # Generation parameters
        self.attn_implementation = attn_implementation
        self.sdpa_backend = sdpa_backend
        # Optimization parameters
        self.compile_mode = compile_mode
        self.compile_options = compile_options if compile_options is not None else {}
        self.kernelize = kernelize
        self.name = name if name is not None else self.infer_name()

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def dtype(self) -> str:
        return "torch.bfloat16"

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict()).encode()).hexdigest()

    def infer_name(self) -> str:
        """Infer a human-readable name for the benchmark config."""
        attn_code = self.attn_implementation + (f"_{self.sdpa_backend}" if self.attn_implementation == "sdpa" else "")
        return "-".join([
            f"w{self.warmup_iterations}_i{self.measurement_iterations}",
            "monitored" if self.gpu_monitoring else "unmonitored",
            f"b{self.batch_size}_s{self.sequence_length}_n{self.num_tokens_to_generate}",
            attn_code,
            f"compiled_{self.compile_mode}" if self.compile_mode is not None else "uncompiled",
            "kernelized" if self.kernelize else "unkernelized",
        ])

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
            "sdpa_backend": self.sdpa_backend,
            "compile_mode": self.compile_mode,
            "compile_options": self.compile_options,
            "kernelize": self.kernelize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            warmup_iterations=data["warmup_iterations"],
            measurement_iterations=data["measurement_iterations"],
            gpu_monitoring=data["gpu_monitoring"],
            batch_size=data["batch_size"],
            sequence_length=data["sequence_length"],
            num_tokens_to_generate=data["num_tokens_to_generate"],
            attn_implementation=data["attn_implementation"],
            sdpa_backend=data["sdpa_backend"],
            compile_mode=data["compile_mode"],
            compile_options=data["compile_options"],
            kernelize=data["kernelize"],
            name=data.get("name"),
        )


def cross_generate_configs(
    attn_impl_and_sdpa_backend: list[tuple[str, Optional[str]]],
    compiled_mode: list[Optional[str]],
    kernelized: list[bool],
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_tokens_to_generate: int = 128,
    gpu_monitoring: bool = False,  # this slows down the benchmark by a lot so we disable it by default
) -> list[BenchmarkConfig]:
    # Create kwargs common to all configs
    kwargs = {
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_tokens_to_generate": num_tokens_to_generate,
        "gpu_monitoring": gpu_monitoring,
    }
    # Cross-generate all combinations of attn_implementation, compiled_mode, and kernelized
    configs = []
    for attn_implementation, sdpa_backend in list(set(attn_impl_and_sdpa_backend)):
        for cm in list(set(compiled_mode)):
            for kernelize_on in list(set(kernelized)):
                config = BenchmarkConfig(
                    attn_implementation=attn_implementation,
                    sdpa_backend=sdpa_backend,
                    compile_mode=cm,
                    kernelize=kernelize_on,
                    **kwargs,
                )
                configs.append(config)
    return configs

def generate_all_configs(
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_tokens_to_generate: int = 128,
    gpu_monitoring: bool = False,
) -> list[BenchmarkConfig]:
    all_attn_implementations = [
        ("flash_attention_2", None),
        ("eager", None),
        ("sdpa", "math"),
        ("sdpa", "flash_attention"),
    ]
    return cross_generate_configs(
        attn_impl_and_sdpa_backend=all_attn_implementations,
        compiled_mode=[None, "default", "reduce-overhead", "max-autotune"],
        kernelized=[False, KERNELIZATION_AVAILABLE],
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_tokens_to_generate=num_tokens_to_generate,
        gpu_monitoring=gpu_monitoring,
    )


def generate_useful_configs(
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_tokens_to_generate: int = 128,
    gpu_monitoring: bool = False,
) -> list[BenchmarkConfig]:
    all_attn_implementations = [
        ("flash_attention_2", None),
        ("eager", None),
        ("sdpa", "math"),
        ("sdpa", "flash_attention"),  # note: this one can fail with compile because of attn mask
    ]
    return cross_generate_configs(
        attn_impl_and_sdpa_backend=all_attn_implementations,
        compiled_mode=[None, "max-autotune"],
        kernelized=[False, KERNELIZATION_AVAILABLE],
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_tokens_to_generate=num_tokens_to_generate,
        gpu_monitoring=gpu_monitoring,
    )
