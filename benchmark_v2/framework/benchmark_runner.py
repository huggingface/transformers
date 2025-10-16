import gc
import json
import logging
import os
import pathlib
import re
import time
from contextlib import nullcontext
from datetime import datetime
from queue import Queue
from typing import Any

import torch
from tqdm import trange

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CompileConfig,
    GenerationConfig,
    GenerationMixin,
)
from transformers.generation.streamers import BaseStreamer

from .benchmark_config import BenchmarkConfig
from .data_classes import BenchmarkMetadata, BenchmarkResult, GPURawMetrics, pretty_print_dict
from .hardware_metrics import GPUMonitor


try:
    from kernels import Mode, kernelize  # noqa: F401
except ImportError:
    kernelize = None
    Mode = None


DEFAULT_PROMPT = "\n".join([
    "The French Revolution was a period of political and societal change in France that began with the Estates General of 1789 and ended with the Coup of 18 Brumaire on 9 November 1799.",
    "Many of the revolution's ideas are considered fundamental principles of liberal democracy, and its values remain central to modern French political discourse.",
    "It was caused by a combination of social, political, and economic factors which the existing regime proved unable to manage.",
    "Financial crisis and widespread social distress led to the convocation of the Estates General in May 1789, its first meeting since 1614.",
    "The representatives of the Third Estate broke away and re-constituted themselves as a National Assembly in June.",
    "The Storming of the Bastille in Paris on 14 July led to a series of radical measures by the Assembly, including the abolition of feudalism, state control over the Catholic Church in France, and issuing the Declaration of the Rights of Man and of the Citizen.",
    "The next three years were dominated by a struggle for political control.",
    "King Louis XVI's attempted flight to Varennes in June 1791 further discredited the monarchy, and military defeats after the outbreak of the French Revolutionary Wars in April 1792 led to the insurrection of 10 August 1792.",
    "As a result, the monarchy was replaced by the French First Republic in September, followed by the execution of Louis XVI himself in January 1793.",
    "After another revolt in June 1793, the constitution was suspended, and political power passed from the National Convention to the Committee of Public Safety, dominated by radical Jacobins led by Maximilien Robespierre.",
    "About 16,000 people were sentenced by the Revolutionary Tribunal and executed in the Reign of Terror, which ended in July 1794 with the Thermidorian Reaction.",
    "Weakened by external threats and internal opposition, the Committee of Public Safety was replaced in November 1795 by the Directory.",
    "Its instability ended in the coup of 18 Brumaire and the establishment of the Consulate, with Napoleon Bonaparte as First Consul.",
])  # fmt: skip


def compact_json_numeric_arrays(data: dict):
    # Match arrays that contain only numbers (ints/floats), whitespace, commas, and newlines
    pattern = r"\[\s*\n\s*((?:\d+(?:\.\d+)?\s*,\s*)*\d+(?:\.\d+)?)\s*\n\s*\]"

    def replace_numeric_array(match):
        # Get the array content
        content = match.group(1)
        # Remove extra whitespace but keep commas
        compact_content = re.sub(r"\s+", " ", content).strip()
        return f"[{compact_content}]"

    return re.sub(pattern, replace_numeric_array, json.dumps(data, indent=4, default=str), flags=re.DOTALL)


def get_git_revision() -> str:
    base_path = pathlib.Path(__file__).parent.parent.parent
    git_dir = base_path / ".git"
    with (git_dir / "HEAD").open("r") as head:
        ref = head.readline().split(" ")[-1].strip()
    with (git_dir / ref).open("r") as git_hash:
        return git_hash.readline().strip()


def get_sdpa_backend(backend_name: str | None) -> torch.nn.attention.SDPBackend | None:
    """Get the SDPA backend enum from string name."""
    if backend_name is None:
        return None

    try:
        backend_map = {
            "math": torch.nn.attention.SDPBackend.MATH,
            "flash_attention": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            "efficient_attention": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "cudnn_attention": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        }
        return backend_map.get(backend_name.lower())
    except AttributeError:
        # torch.nn.attention.SDPBackend not available in older torch versions
        return None


def flush_memory():
    """Flush GPU memory and run garbage collection."""
    gc.collect()
    # Dynamo resets
    torch._dynamo.reset()
    torch._dynamo.reset_code_caches()
    if hasattr(torch._inductor, "codecache"):
        # Clear FX graph cache
        if hasattr(torch._inductor.codecache, "FxGraphCache"):
            torch._inductor.codecache.FxGraphCache.clear()
        # Clear PyCodeCache
        if hasattr(torch._inductor.codecache, "PyCodeCache"):
            torch._inductor.codecache.PyCodeCache.cache_clear()
        # Clear TritonFuture cache (for async compilation)
        if hasattr(torch._inductor.codecache, "TritonFuture"):
            if hasattr(torch._inductor.codecache.TritonFuture, "_compile_cache"):
                torch._inductor.codecache.TritonFuture._compile_cache.clear()
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()


class BenchmarkStreamer(BaseStreamer):
    def __init__(self, **kwargs) -> None:
        self.timestamps = []
        self.text_queue = Queue()

    def put(self, value):
        """Receives tokens and logs the timestamp of the generation."""
        self.timestamps.append(time.perf_counter())

    def end(self):
        self.timestamps.append(time.perf_counter())

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class BenchmarkRunner:
    """Main benchmark runner that coordinates benchmark execution."""

    def __init__(self, logger: logging.Logger, output_dir: str | None = None, commit_id: str | None = None) -> None:
        # Those stay constant for the whole run
        self.logger = logger
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_results")
        self.output_dir = output_dir
        self.commit_id = get_git_revision() if commit_id is None else commit_id
        os.makedirs(self.output_dir, exist_ok=True)
        self.profile_dir = None
        # Attributes that are reset for each model
        self._setup_for = ""
        # Attributes that are reset for each run
        self.model: GenerationMixin | None = None

    def cleanup(self) -> None:
        del self.model
        self.model = None
        flush_memory()

    def setup_one_run(self, model_id: str, config: BenchmarkConfig) -> None:
        # Some attributes only need to be set once per model
        if self._setup_for != model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # We set the EOS token to the padding token for open-ended generation
            self.tokenizer.eos_token = self.tokenizer.pad_token
            self._setup_for = model_id

        # Prepare inputs
        self.inputs = self.tokenizer(
            [DEFAULT_PROMPT for _ in range(config.batch_size)],
            return_tensors="pt",
            max_length=config.sequence_length,
            truncation=True,
            return_attention_mask=True,
        ).to(config.device)
        self.inputs["use_cache"] = True

        # Prepare generation config
        gen_config = GenerationConfig(
            do_sample=False, top_p=1.0, temperature=1.0, max_new_tokens=config.num_tokens_to_generate
        )

        # Prepare compile config
        if config.compile_mode is not None:
            gen_config.compile_config = CompileConfig(mode=config.compile_mode, options=config.compile_options)
            gen_config.cache_implementation = "static"

        # Load model
        self.logger.debug(f"Loading model {model_id} on device {config.device}...")
        dtype = getattr(torch, config.dtype.removeprefix("torch."))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=config.attn_implementation, generation_config=gen_config
        )
        self.model = self.model.eval().to(config.device)

        # Kernelize the model if needed
        if config.kernelize:
            self.model = kernelize(self.model, mode=Mode.INFERENCE)

    def run_one_benchmark(self, model_id: str, config: BenchmarkConfig, num_tokens_to_profile: int = 0) -> None:
        sdpa_ctx = nullcontext()
        if config.attn_implementation == "sdpa":
            sdpa_backend = get_sdpa_backend(config.sdpa_backend)
            sdpa_ctx = torch.nn.attention.sdpa_kernel(sdpa_backend)

        with sdpa_ctx, torch.no_grad():
            self.logger.info(f"Running benchmark scenario: {config.name}")

            # Quick validation: try one measurement first to see if this scenario works
            flush_memory()
            e2e_latency, token_generation_times, shape_and_decoded_output, gpu_metrics = self.time_generate(
                max_new_tokens=1, gpu_monitor=None
            )
            if e2e_latency < 0:
                self.logger.warning(f"Skipping config {config.name}: {e2e_latency = } (no GPU monitoring)")
                return None

            # Warmup runs
            self.logger.info(f"Warming up with {config.warmup_iterations} iterations...")
            for _ in trange(config.warmup_iterations):
                _ = self.time_generate(max_new_tokens=config.num_tokens_to_generate)
            self.logger.info("Warmup over.")

            # Measurement runs
            result = BenchmarkResult()
            self.logger.info(f"Benchmarking with {config.measurement_iterations} iterations.")
            for _ in trange(config.measurement_iterations):
                e2e_latency, token_generation_times, shape_and_decoded_output, gpu_metrics = self.time_generate(
                    max_new_tokens=config.num_tokens_to_generate,
                    gpu_monitor=(GPUMonitor(logger=self.logger) if config.gpu_monitoring else None),
                )
                result.accumulate(e2e_latency, token_generation_times, shape_and_decoded_output, gpu_metrics)
            self.logger.info("Benchmarking done. Cleaning up.")

            # Profile if needed
            if num_tokens_to_profile > 0:
                self.profile_generate(num_tokens_to_profile, config.name)

            return {
                "metadata": BenchmarkMetadata(model_id=model_id, commit_id=self.commit_id),
                "measurements": result,
                "config": config,
            }

    def time_generate(
        self,
        max_new_tokens: int,
        gpu_monitor: GPUMonitor | None = None,
    ) -> tuple[float, list[float], str, GPURawMetrics | None]:
        """Time the latency of a call to model.generate() with the given (inputs) and (max_new_tokens)."""
        # Prepare gpu monitoring if needed
        if gpu_monitor is not None:
            gpu_monitor.start()
        # Prepare streamer
        streamer = BenchmarkStreamer()
        # Generate and time
        wall_time_0 = time.perf_counter()
        outputs = self.model.generate(
            **self.inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )
        wall_time_1 = time.perf_counter()
        # Stop gpu monitoring if needed
        gpu_metrics = gpu_monitor.stop_and_collect() if gpu_monitor is not None else None
        # Check if generation had the right number of tokens
        input_tokens = self.inputs["input_ids"].size(-1)
        batch_size, output_tokens = outputs.shape
        new_tokens = output_tokens - input_tokens
        if new_tokens != max_new_tokens:
            raise RuntimeError(f"Generated {new_tokens} tokens, expected {max_new_tokens}")
        # Decode outputs
        decoded_output = self.tokenizer.decode(outputs[0, input_tokens:], skip_special_tokens=True)
        shape_and_decoded_output = f"{tuple(outputs.shape)} | {decoded_output}"
        # Compute intermediate quantities
        e2e_latency = wall_time_1 - wall_time_0
        token_generation_times = [t - wall_time_0 for t in streamer.timestamps[1:]]
        return e2e_latency, token_generation_times, shape_and_decoded_output, gpu_metrics

    def profile_generate(self, num_tokens_to_profile: int, config_name: str) -> None:
        """Profile the latency of a call to model.generate() with the given (inputs) and (max_new_tokens)."""
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        with profiler as prof:
            _ = self.model.generate(
                **self.inputs,
                max_new_tokens=num_tokens_to_profile,
            )
        if self.profile_dir is None:
            self.profile_dir = self.output_dir + "_profiles"
            os.makedirs(self.profile_dir, exist_ok=True)
        prof.export_chrome_trace(f"{self.profile_dir}/{config_name}.json")

    def run_benchmarks(
        self,
        model_id: str,
        benchmark_configs: list[BenchmarkConfig],
        num_tokens_to_profile: int = 0,
        pretty_print_summary: bool = True,
    ) -> dict[str, Any]:
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.perf_counter()

        n_configs = len(benchmark_configs)
        for i, config in enumerate(benchmark_configs):
            # Handle SDPA backend if not determined by the config (needs to be done before skipping duplicates)
            if config.attn_implementation == "sdpa" and config.sdpa_backend is None:
                default_backend = "flash_attention"  # FIXME: torch has a _cur_sdpa_kernel_backends but it fails
                self.logger.warning(f"No SDPA backend provided, using {default_backend} instead.")
                config.sdpa_backend = default_backend

            # Skip if already run
            if config.hash in all_results:
                self.logger.info(f"Skipping duplicate config {config.name} for model {model_id} ({i + 1}/{n_configs})")
                continue

            # Otherwise, run the benchmark
            self.setup_one_run(model_id, config)
            self.logger.info(
                f"Running benchmark of model {model_id} with scenario: {config.name} ({i + 1}/{n_configs})"
            )

            # Launch benchmark in a try/except block to avoid stopping the whole run if one benchmark fails
            try:
                results = self.run_one_benchmark(model_id, config, num_tokens_to_profile)
                if results is not None:
                    all_results[config.hash] = results

            except Exception as e:
                self.logger.error(f"Error running with scenario: {config.name}:\n{repr(e)}")
            # Cleanup model and save results
            self.cleanup()
            self.save_results(model_id, all_results, timestamp=timestamp)

        if pretty_print_summary:
            print()
            print("=" * 100)
            print(f"Finished benchmarks in {time.perf_counter() - start_time:.2f} seconds")
            print(f"Total number of benchmarks: {len(all_results)}")
            if len(all_results) > 0:
                print("First run metadata:")
                first_key = list(all_results.keys())[0]
                first_metadata = all_results[first_key]["metadata"].to_dict()
                hardware_info = first_metadata.pop("hardware_info")
                pretty_print_dict(first_metadata | hardware_info, tabs=1)
            for result in all_results.values():
                print("=" * 100)
                print(f"Config: {result['config'].infer_name(compact=False)}\n")
                result["measurements"].pprint(batch_size=result["config"].batch_size, tabs=1)
            print("=" * 100)

        return all_results

    def save_results(self, model_name: str, results: dict, timestamp: str = "") -> str:
        """Save benchmark results to JSON file."""
        # Create model-specific subdirectory
        model_name = model_name.replace("/", "_")
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if not timestamp else timestamp
        filename = f"{model_name}_benchmark_{timestamp}.json"
        filepath = os.path.join(model_dir, filename)

        # Convert results to dict
        converted_results = {}
        for cfg_hash in results.keys():
            converted_results[cfg_hash] = {
                "metadata": results[cfg_hash]["metadata"].to_dict(),
                "measurements": results[cfg_hash]["measurements"].to_dict(),
                "config": results[cfg_hash]["config"].to_dict(),
            }

        # Save to JSON file
        with open(filepath, "w") as f:
            f.write(compact_json_numeric_arrays(converted_results))

        self.logger.info(f"Results saved to {filepath}")
        return filepath
