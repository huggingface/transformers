from dataclasses import dataclass, field
from typing import Any, Optional, Union


KERNELIZATION_AVAILABLE = False
try:
    from kernels import Mode, kernelize  # noqa: F401

    KERNELIZATION_AVAILABLE = True
except ImportError:
    pass


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
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            name=data["name"],
            warmup_iterations=data["warmup_iterations"],
            measurement_iterations=data["measurement_iterations"],
            gpu_monitoring=data["gpu_monitoring"],
            batch_size=data["batch_size"],
            sequence_length=data["sequence_length"],
            num_tokens_to_generate=data["num_tokens_to_generate"],
            attn_implementation=data["attn_implementation"],
            use_cache=data["use_cache"],
            sdpa_backend=data["sdpa_backend"],
            compilation=data["compilation"],
            compile_mode=data["compile_mode"],
            compile_options=data["compile_options"],
            kernelize=data["kernelize"],
            device=data["device"],
            dtype=data["dtype"],
        )


def cross_generate_configs(
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_tokens_to_generate: int = 128,
    use_cache: bool = True,  # no real interest in testing with cache disabled
    gpu_monitoring: bool = False,  # this slows down the benchmark by a lot so we disable it by default
) -> list[BenchmarkConfig]:
    kwargs = {
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_tokens_to_generate": num_tokens_to_generate,
        "use_cache": use_cache,
        "gpu_monitoring": gpu_monitoring,
    }
    configs = []
    for attn_implementation in [("flash_attention_2", None), ("eager", None), ("sdpa", "math"), ("sdpa", "flash_attention"), ("sdpa", "efficient_attention")]:
        for compiled_mode in [None, "default", "max-autotune", "reduce-overhead"]:
            for kernelized in {False, KERNELIZATION_AVAILABLE}:
                name = [
                    str(attn_implementation[0]),
                    str(attn_implementation[1]),
                    f"compiled_{compiled_mode}" if compiled_mode is not None else "uncompiled",
                    "kernelized" if kernelized else "vanilla",
                    "with_cache" if use_cache else "no_cache",
                ]
                config = BenchmarkConfig(
                    name = "_".join(name),
                    attn_implementation=attn_implementation[0],
                    sdpa_backend=attn_implementation[1],
                    compilation=compiled_mode is not None,
                    kernelize=kernelized,
                    compile_mode=compiled_mode,
                    **kwargs,
                )
                configs.append(config)
    return configs


def smart_generate_configs(
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_tokens_to_generate: int = 128,
    use_cache: bool = True,
    gpu_monitoring: bool = False,
) -> list[BenchmarkConfig]:
    kwargs = {
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_tokens_to_generate": num_tokens_to_generate,
    }
    configs = []
    for attn_implementation in [("eager", None), ("sdpa", None)]:
        for compiled_mode in [None, "max-autotune"]:
            for kernelized in {False, KERNELIZATION_AVAILABLE}:
                name = [
                    str(attn_implementation[0]),
                    str(attn_implementation[1]),
                    "compiled" if compiled_mode is not None else "uncompiled",
                    "kernelized" if kernelized else "vanilla",
                    "with_cache" if use_cache else "no_cache",
                ]
                config = BenchmarkConfig(
                    name = "_".join(name),
                    attn_implementation=attn_implementation[0],
                    sdpa_backend=attn_implementation[1],
                    compilation=compiled_mode is not None,
                    kernelize=kernelized,
                    compile_mode=compiled_mode,
                    gpu_monitoring=gpu_monitoring,
                    use_cache=use_cache,
                    **kwargs,
                )
                configs.append(config)
    return configs
