import gc
import json
import logging
import os
import pathlib
import re
import tempfile
import time
from datetime import datetime
from queue import Queue
from typing import Any

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import trange

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
    is_torch_xpu_available,
)
from transformers.generation.streamers import BaseStreamer
from transformers.utils import is_torch_accelerator_available

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

PUSH_TO_HUB_TOKEN = os.getenv("PUSH_TO_HUB_TOKEN", None)


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


def flush_memory(flush_compile: bool = True) -> None:
    """Flush GPU memory and run garbage collection. If the flush_compile flag is set, we also clear the everything
    related to compile cache."""
    gc.collect()
    # If needed, flush everything related to torch.compile
    if flush_compile:
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
    # Clear device cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif is_torch_xpu_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    gc.collect()


class BenchmarkStreamer(BaseStreamer):
    def __init__(self, **kwargs) -> None:
        self.timeout = kwargs.pop("timeout", 10)
        self.timestamps = []
        self.text_queue = Queue()
        self.stop_signal = None

    def put(self, value):
        """Receives tokens and logs the timestamp of the generation."""
        self.timestamps.append(time.perf_counter())
        self.text_queue.put(value)

    def end(self):
        self.timestamps.append(time.perf_counter())
        self.text_queue.put(self.stop_signal)

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

    def __init__(
        self,
        logger: logging.Logger,
        output_dir: str | None = None,
        branch_name: str | None = None,
        commit_id: str | None = None,
        commit_message: str | None = None,
    ) -> None:
        # Those stay constant for the whole run
        self.logger = logger
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_results")
        self.output_dir = output_dir
        self.branch_name = branch_name
        self.commit_id = get_git_revision() if commit_id is None else commit_id
        self.commit_message = commit_message
        os.makedirs(self.output_dir, exist_ok=True)
        self.profile_dir = None
        # Attributes that are reset for each model
        self._setup_for = ""
        # Attributes that are reset for each run
        self.model: GenerationMixin | None = None
        self.device_type = torch.accelerator.current_accelerator().type if is_torch_accelerator_available() else "cuda"
        self.torch_accelerator_module = getattr(torch, self.device_type, torch.cuda)

    def cleanup(self) -> None:
        del self.model
        self.model = None
        flush_memory()

    def setup_benchmark(self, model_id: str, config: BenchmarkConfig) -> None:
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
        generation_config_kwargs = {
            "do_sample": False,
            "max_new_tokens": config.num_tokens_to_generate,
        }

        # Add compile config if found
        if config.compile_config is not None:
            generation_config_kwargs.update(compile_config=config.compile_config)
            # To trigger compile in generate, we need to set the cache to static
            if not config.continuous_batching:
                generation_config_kwargs.update(cache_implementation="static")

        generation_config = GenerationConfig(**generation_config_kwargs)

        # Load model
        self.logger.debug(f"Loading model {model_id} on device {config.device}...")
        dtype = getattr(torch, config.dtype.removeprefix("torch."))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=config.attn_implementation, generation_config=generation_config
        )
        self.model = self.model.eval().to(config.device)

        # Kernelize the model if needed
        if config.kernelize and kernelize is not None and Mode is not None:
            self.model = kernelize(self.model, mode=Mode.INFERENCE)

    def run_benchmark(self, config: BenchmarkConfig, num_tokens_to_profile: int = 0) -> BenchmarkResult | None:
        """Run a single benchmark with the given model ID and config."""
        with torch.no_grad():
            self.logger.info(f"Running benchmark scenario: {config.name}")
            self.logger.debug(f"Full config: {config.to_dict()}")

            # Quick validation: try one measurement first to see if this scenario works
            flush_memory()
            e2e_latency = self.time_generate(config, warmup=True)[0]
            if e2e_latency < 0:
                self.logger.warning(f"Skipping config {config.name}: {e2e_latency = }")
                return None

            # Warmup runs
            self.logger.info(f"Warming up with {config.warmup_iterations} iterations...")
            for _ in trange(config.warmup_iterations, desc="Warmup"):
                self.time_generate(config, warmup=True)
            self.logger.info("Warmup over.")

            # Measurement runs
            result = BenchmarkResult()
            self.logger.info(f"Benchmarking with {config.measurement_iterations} iterations.")
            for _ in trange(config.measurement_iterations, desc="Benchmarking"):
                e2e_latency, timestamps, shape_and_decoded_output, gpu_metrics = self.time_generate(
                    config, warmup=False
                )
                result.accumulate(e2e_latency, timestamps, shape_and_decoded_output, gpu_metrics)
            self.logger.info("Benchmarking done. Cleaning up.")

            # Profile if needed
            if num_tokens_to_profile > 0:
                self.profile_generate(num_tokens_to_profile, config.name)

            return result

    def time_generate(
        self, config: BenchmarkConfig, warmup: bool
    ) -> tuple[float, list[float], str, GPURawMetrics | None]:
        # Prepare gpu monitoring if needed
        if config.gpu_monitoring and not warmup:
            gpu_monitor = GPUMonitor(logger=self.logger)
            gpu_monitor.start()
        else:
            gpu_monitor = None

        # Generate and time
        if config.continuous_batching:
            inputs = self.inputs["input_ids"].tolist()
            wall_time_0 = time.perf_counter()
            outputs = self.model.generate_batch(inputs, allow_block_sharing=False, record_timestamps=True)
        else:
            streamer = BenchmarkStreamer()
            wall_time_0 = time.perf_counter()
            outputs = self.model.generate(**self.inputs, streamer=streamer)

        wall_time_1 = time.perf_counter()
        gpu_metrics = gpu_monitor.stop_and_collect() if gpu_monitor is not None else None

        # Retrieve timestamps and results in a way that allows similar post-processing
        input_tokens = self.inputs["input_ids"].size(-1)
        if config.continuous_batching:
            timestamps = [output.timestamps[:] for output in outputs.values()]
            results = torch.tensor([output.generated_tokens[:] for output in outputs.values()])
        else:
            timestamps = [streamer.timestamps[1:]]  # skip the first timestamp because it's the input tokens
            results = outputs[:, input_tokens:]
        outputs = None
        flush_memory(flush_compile=False)

        # Check if generation had the right number of tokens
        if results.size(-1) != config.num_tokens_to_generate:
            raise RuntimeError(f"Generated {results.size(-1)} tokens, expected {config.num_tokens_to_generate}")

        # Decode outputs
        decoded_output = self.tokenizer.decode(results[0], skip_special_tokens=True)
        shape_and_decoded_output = f"{tuple(results.shape)} | {decoded_output}"

        # Compute metrics
        e2e_latency = wall_time_1 - wall_time_0
        timestamps = torch.tensor(timestamps).sub(wall_time_0).tolist()
        self.logger.info(
            f"Time generate done in {e2e_latency:.2f} seconds. Memory usage: {self.torch_accelerator_module.memory_allocated() / 1024**2:.2f} MB"
        )
        return e2e_latency, timestamps, shape_and_decoded_output, gpu_metrics

    def profile_generate(self, num_tokens_to_profile: int, config_name: str) -> None:
        """Profile the latency of a call to model.generate() with the given (inputs) and (max_new_tokens)."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device_type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif self.device_type == "xpu":
            activities.append(torch.profiler.ProfilerActivity.XPU)

        profiler = torch.profiler.profile(
            activities=activities,
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

    @torch.inference_mode()
    def run_benchmarks(
        self,
        model_id: str,
        benchmark_configs: list[BenchmarkConfig],
        num_tokens_to_profile: int = 0,
        pretty_print_summary: bool = True,
        summarized: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """Run multiple benchmarks for the given model ID and list of benchmark configs."""
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.perf_counter()

        n_configs = len(benchmark_configs)
        for i, config in enumerate(benchmark_configs):
            # Skip if already run
            if config.hash in all_results:
                self.logger.info(f"Skipping duplicate config {config.name} for model {model_id} ({i + 1}/{n_configs})")
                continue

            # Otherwise, run the benchmark
            self.setup_benchmark(model_id, config)
            self.logger.info(
                f"Running benchmark of model {model_id} with scenario: {config.name} ({i + 1}/{n_configs})"
            )

            # Launch benchmark in a try/except block to avoid stopping the whole run if one benchmark fails
            try:
                result = self.run_benchmark(config, num_tokens_to_profile)
            except Exception as e:
                self.logger.error(f"Error running with scenario: {config.name}:\n{repr(e)}")
                result = None

            # Memoize
            all_results[config.hash] = {
                "metadata": BenchmarkMetadata(
                    model_id=model_id,
                    branch_name=self.branch_name,
                    commit_id=self.commit_id,
                    commit_message=self.commit_message,
                    success=result is not None,
                ),
                "measurements": result if result is not None else BenchmarkResult(),
                "config": config,
            }

            # Cleanup model and save results
            self.cleanup()
            self.save_results(model_id, all_results, timestamp=timestamp, summarized=summarized)

        if len(all_results) < 1:
            raise RuntimeError("No benchmark was run successfully")

        if pretty_print_summary:
            print()
            print("=" * 100)
            print(f"Finished benchmarks in {time.perf_counter() - start_time:.2f} seconds")
            print(f"Total number of benchmarks: {len(all_results)}")
            print("First run metadata:")
            first_key = list(all_results.keys())[0]
            first_metadata = all_results[first_key]["metadata"].to_dict()
            hardware_info = first_metadata.pop("hardware_info")
            pretty_print_dict(first_metadata | hardware_info, tabs=1)
            for result in all_results.values():
                print("=" * 100)
                print(f"Config: {result['config'].infer_name(compact=False)}\n")
                result["measurements"].pprint(
                    batch_size=result["config"].batch_size,
                    num_generated_tokens=result["config"].num_tokens_to_generate,
                    tabs=1,
                )
            print("=" * 100)

        return (timestamp, all_results)

    def save_results(self, model_name: str, results: dict, timestamp: str = "", summarized: bool = True) -> str:
        """Save benchmark results to JSON file."""
        # Create model-specific subdirectory
        model_name = model_name.replace("/", "_")
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_benchmark_{timestamp}.json"
        filepath = os.path.join(model_dir, filename)

        # Convert results to dict
        converted_results = {}
        for cfg_hash in results.keys():
            converted_results[cfg_hash] = {
                "metadata": results[cfg_hash]["metadata"].to_dict(),
                "measurements": results[cfg_hash]["measurements"].to_dict(summarized=summarized),
                "config": results[cfg_hash]["config"].to_dict(),
            }

        # Save to JSON file
        with open(filepath, "w") as f:
            f.write(compact_json_numeric_arrays(converted_results))

        self.logger.info(f"Results saved to {filepath}")
        return filepath

    def push_results_to_hub(self, dataset_id: str, results: dict[Any, Any], timestamp: str) -> None:
        if PUSH_TO_HUB_TOKEN is None:
            raise ValueError(
                "PUSH_TO_HUB_TOKEN is not set, cannot push results to the Hub. When setting dataset_id, please also set the PUSH_TO_HUB_TOKEN environment variable."
            )

        api = HfApi()
        n_results = len(results)
        for summarized in [False, True]:
            self.logger.info(f"Pushing {n_results} results to: {dataset_id} with {summarized = }")
            rows = []
            for cfg_hash, entry in results.items():
                row = {
                    "benchmark_config_hash": cfg_hash,
                    "config": entry["config"].to_dict(),
                    "measurements": entry["measurements"].to_dict(summarized=summarized),
                    "metadata": entry["metadata"].to_dict(),
                }
                rows.append(row)

            ds = Dataset.from_list(rows)
            with tempfile.TemporaryDirectory() as tmp:
                file_name = "summarized_results" if summarized else "full_results"
                jsonl_path = os.path.join(tmp, f"{file_name}.jsonl")
                with open(jsonl_path, "w") as f:
                    json_lines = []
                    for ex in ds:
                        json_lines.append(json.dumps(ex, ensure_ascii=False))
                    f.write("\n".join(json_lines))

                # NOTE: we expect the repository to already exist
                timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = file_name + "/" + f"benchmark_run_{timestamp}.jsonl"
                api.upload_file(
                    path_or_fileobj=jsonl_path,
                    path_in_repo=file_name,
                    repo_id=dataset_id,
                    repo_type="dataset",
                    token=PUSH_TO_HUB_TOKEN,
                )
                self.logger.info(f"Successfully uploaded results to: {dataset_id} with {summarized = }")
