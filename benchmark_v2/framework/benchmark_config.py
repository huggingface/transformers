import hashlib
import itertools
import json
import logging
from typing import Any

from transformers.utils.import_utils import is_flash_attn_2_available


KERNELIZATION_AVAILABLE = False
try:
    from kernels import Mode, kernelize  # noqa: F401

    KERNELIZATION_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """Configuration for a single benchmark scenario."""

    all_attn_implementations = [
        ("flash_attention_2", None),
        ("eager", None),
        ("sdpa", "math"),
        ("sdpa", "flash_attention"),
        ("flex_attention", None),
    ]

    all_compiled_modes = [None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

    def __init__(
        self,
        warmup_iterations: int = 5,
        measurement_iterations: int = 20,
        gpu_monitoring: bool = True,  # NOTE: you may want to disable this at times as we have obsvered it could heavily slow down benchmarks on AMD
        continuous_batching: bool = False,
        batch_size: int = 1,
        sequence_length: int = 128,
        num_tokens_to_generate: int = 128,
        attn_implementation: str = "eager",
        sdpa_backend: str | None = None,
        compile_mode: str | None = None,
        compile_options: dict[str, Any] | None = None,
        kernelize: bool = False,
        name: str | None = None,
        skip_validity_check: bool = False,
    ) -> None:
        # Benchmark parameters
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.gpu_monitoring = gpu_monitoring
        self.continuous_batching = continuous_batching
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
        # Constant parameters
        self.dtype = "torch.bfloat16"
        self.device = "cuda"

        self.check_validity(skip_validity_check)
        self.name = name if name is not None else self.infer_name()

    def check_validity(self, skip_validity_check: bool = False) -> None:
        if skip_validity_check:
            return
        # Check FA is installed
        if self.attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning(
                "Flash attention does not support compile mode. Defaulting to SDPA w/ flash attention backend."
            )
            self.attn_implementation = "sdpa"
            self.sdpa_backend = "flash_attention"
        # Flash attention does not support compile mode, so we turn it off # FIXME: it would be better to support it
        is_fa = self.attn_implementation == "flash_attention_2"
        is_fa |= self.attn_implementation == "sdpa" and self.sdpa_backend == "flash_attention"
        if is_fa:
            logger.warning("Flash attention does not support compile mode. Turning off compile mode.")
            self.compile_mode = None
        # Handle SDPA backend if not determined by the config (needs to be done before skipping duplicates)
        if self.attn_implementation == "sdpa" and self.sdpa_backend is None:
            default_backend = "flash_attention"  # FIXME: torch has a _cur_sdpa_kernel_backends but it fails
            logger.warning(f"No SDPA backend provided, using {default_backend} instead.")
            self.sdpa_backend = default_backend
        if self.continuous_batching:
            if self.attn_implementation == "flex_attention":
                logger.error(
                    "disabling continuous batching because of invalid configuration: flex attention is not supported"
                )
                self.continuous_batching = False
            elif self.attn_implementation == "sdpa" and self.sdpa_backend is not None:
                logger.warning(
                    "when continuous batching is enabled, sdpa_backend must be None because of the attention mask, setting it to None"
                )
                self.sdpa_backend = "math"

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict()).encode()).hexdigest()

    def infer_name(self, compact: bool = True) -> str:
        """Infer a human-readable name for the benchmark config, either compact or verbose."""
        if compact:
            iter_str = f"w{self.warmup_iterations}_i{self.measurement_iterations}"
            gpu_monitor_str = "monitored" if self.gpu_monitoring else "unmonitored"
            dimensions_str = f"b{self.batch_size}_s{self.sequence_length}_n{self.num_tokens_to_generate}"
            attn_code = self.attn_implementation
            attn_code += f"_{self.sdpa_backend}" if self.attn_implementation == "sdpa" else ""
            compile_str = f"compiled_{self.compile_mode}" if self.compile_mode is not None else "uncompiled"
            kernelize_str = "kernelized" if self.kernelize else "unkernelized"
            continuous_batching_str = "cb" if self.continuous_batching else "generate"
            sep = "-"
        else:
            iter_str = f"{self.warmup_iterations} warmup, {self.measurement_iterations} iterations"
            gpu_monitor_str = ("with" if self.gpu_monitoring else "no") + " GPU monitoring"
            dimensions_str = f"batch size {self.batch_size}, sequence length {self.sequence_length}, {self.num_tokens_to_generate} generated tokens"
            attn_code = f"{self.attn_implementation} attention"
            attn_code += f" with {self.sdpa_backend} backend" if self.attn_implementation == "sdpa" else ""
            compile_str = "compiled" if self.compile_mode is not None else "not compiled"
            kernelize_str = "kernelized" if self.kernelize else "not kernelized"
            continuous_batching_str = "continuous batching" if self.continuous_batching else "regular generate"
            sep = ", "
        return sep.join(
            [iter_str, gpu_monitor_str, dimensions_str, attn_code, compile_str, kernelize_str, continuous_batching_str]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "gpu_monitoring": self.gpu_monitoring,
            "continuous_batching": self.continuous_batching,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "num_tokens_to_generate": self.num_tokens_to_generate,
            "attn_implementation": self.attn_implementation,
            "sdpa_backend": self.sdpa_backend,
            "compile_mode": self.compile_mode,
            "compile_options": self.compile_options | {},  # to avoid inplace modification of the original dict
            "kernelize": self.kernelize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], skip_validity_check: bool = False) -> "BenchmarkConfig":
        return cls(
            warmup_iterations=data.get("warmup_iterations", 5),
            measurement_iterations=data.get("measurement_iterations", 20),
            gpu_monitoring=data.get("gpu_monitoring", False),
            continuous_batching=data.get("continuous_batching", False),
            batch_size=data.get("batch_size", 1),
            sequence_length=data.get("sequence_length", 128),
            num_tokens_to_generate=data.get("num_tokens_to_generate", 128),
            attn_implementation=data.get("attn_implementation", "eager"),
            sdpa_backend=data.get("sdpa_backend"),
            compile_mode=data.get("compile_mode"),
            compile_options=data.get("compile_options"),
            kernelize=data.get("kernelize", False),
            name=data.get("name"),
            skip_validity_check=skip_validity_check,
        )


def adapt_configs(
    configs: list[BenchmarkConfig],
    warmup_iterations: int | list[int] = 5,
    measurement_iterations: int | list[int] = 20,
    batch_size: int | list[int] = 1,
    sequence_length: int | list[int] = 128,
    num_tokens_to_generate: int | list[int] = 128,
    gpu_monitoring: bool | list[bool] = True,
) -> list[BenchmarkConfig]:
    parameters = (
        x if isinstance(x, list) else [x]
        for x in [
            warmup_iterations,
            measurement_iterations,
            batch_size,
            sequence_length,
            num_tokens_to_generate,
            gpu_monitoring,
        ]
    )
    iterator = itertools.product(*parameters)

    adapted_configs = []
    for warmup_iters, measurement_iters, bs, seqlen, ntok, monitor in iterator:
        for config in configs:
            config = config.to_dict()
            config["warmup_iterations"] = warmup_iters
            config["measurement_iterations"] = measurement_iters
            config["batch_size"] = bs
            config["sequence_length"] = seqlen
            config["num_tokens_to_generate"] = ntok
            config["gpu_monitoring"] = monitor
            # Remove the old name so it gets re-inferred with the updated values
            config.pop("name", None)
            adapted_configs.append(BenchmarkConfig.from_dict(config))
    return adapted_configs


def get_config_by_level(level: int) -> list[BenchmarkConfig]:
    configs = []
    # Early return if level is greater than 3: we generate all combinations of configs, maybe even w/ all compile modes
    if level >= 3:
        for attn_implementation, sdpa_backend in BenchmarkConfig.all_attn_implementations:
            # Usually there is not much to gain by compiling with other modes, but we allow it for level 4
            compile_modes = BenchmarkConfig.all_compiled_modes if level >= 4 else [None, "default"]
            for cm in compile_modes:
                for kernelize_on in {False, KERNELIZATION_AVAILABLE}:
                    for cb_on in [False, True]:
                        configs.append(
                            BenchmarkConfig(
                                attn_implementation=attn_implementation,
                                sdpa_backend=sdpa_backend,
                                compile_mode=cm,
                                kernelize=kernelize_on,
                                continuous_batching=cb_on,
                            )
                        )
        return configs
    # Otherwise, we add the configs for the given level
    if level >= 0:
        configs.append(BenchmarkConfig(attn_implementation="flex_attention", compile_mode="default"))
    if level >= 1:
        configs.append(BenchmarkConfig(attn_implementation="flash_attention_2"))
        configs.append(BenchmarkConfig(attn_implementation="eager", compile_mode="default"))
        configs.append(BenchmarkConfig(attn_implementation="flash_attention_2", continuous_batching=True))
    if level >= 2:
        configs.append(BenchmarkConfig(attn_implementation="sdpa", compile_mode="default"))
        configs.append(BenchmarkConfig(attn_implementation="flex_attention", compile_mode="default", kernelize=True))
        configs.append(BenchmarkConfig(attn_implementation="flash_attention_2", kernelize=True))
        configs.append(BenchmarkConfig(attn_implementation="paged|sdpa", continuous_batching=True))
    return configs
