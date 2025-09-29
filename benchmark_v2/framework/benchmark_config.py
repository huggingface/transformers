from .data_classes import BenchmarkConfig


KERNELIZATION_AVAILABLE = False
try:
    from kernels import Mode, kernelize  # noqa: F401

    KERNELIZATION_AVAILABLE = True
except ImportError:
    pass


def cross_generate_configs(
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_tokens_to_generate: int = 128,
    gpu_monitoring: bool = False,
) -> list[BenchmarkConfig]:
    kwargs = {
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_tokens_to_generate": num_tokens_to_generate,
        "gpu_monitoring": gpu_monitoring,
    }
    configs = []
    for attn_implementation in [("eager", None), ("sdpa", "math"), ("sdpa", "flash_attention"), ("sdpa", "efficient_attention")]:
        for compiled_mode in [None, "default", "reduce-overhead", "max-autotune"]:
            for kernelized in {False, KERNELIZATION_AVAILABLE}:
                for use_cache in [True, False]:
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
                        use_cache=use_cache,
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
