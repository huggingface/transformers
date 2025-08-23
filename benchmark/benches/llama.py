# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from logging import Logger
import os
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Optional
import sys

# Add the parent directory to Python path to import benchmarks_entrypoint
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks_entrypoint import MetricsRecorder

import gpustat
import psutil
import psycopg2

# Optional heavy ML dependencies - only required when actually running the benchmark
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    GenerationConfig = None
    StaticCache = None

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "1"

# Only set torch precision if torch is available
if TRANSFORMERS_AVAILABLE:
    torch.set_float32_matmul_precision("high")


def collect_metrics(benchmark_id, continue_metric_collection, metrics_recorder):
    p = psutil.Process(os.getpid())
    while not continue_metric_collection.is_set():
        with p.oneshot():
            cpu_util = p.cpu_percent()
            mem_megabytes = p.memory_info().rss / (1024 * 1024)
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_util = gpu_stats[0]["utilization.gpu"]
        gpu_mem_megabytes = gpu_stats[0]["memory.used"]
        metrics_recorder.collect_device_measurements(
            benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes
        )
        sleep(0.01)


def run_benchmark(
    logger: Logger, repository: str, branch: str, commit_id: str, commit_msg: str, metrics_recorder=None, num_tokens_to_generate=100
):
    # Check if required ML dependencies are available
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers and torch are required to run the LLaMA benchmark. Please install them with:")
        logger.error("pip install torch transformers")
        logger.error("Skipping LLaMA benchmark due to missing dependencies.")
        return
    
    continue_metric_collection = Event()
    metrics_thread = None
    model_id = "meta-llama/Llama-2-7b-hf"
    
    # If no metrics_recorder is provided, create one for backward compatibility
    if metrics_recorder is None:
        try:
            metrics_recorder = MetricsRecorder(
                psycopg2.connect("dbname=metrics"), logger, repository, branch, commit_id, commit_msg, True
            )
            should_close_recorder = True
        except Exception as e:
            logger.error(f"Failed to create metrics recorder: {e}")
            return
    else:
        should_close_recorder = False
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_name = gpu_stats[0]["name"]
        benchmark_id = metrics_recorder.initialise_benchmark({"gpu_name": gpu_name, "model_id": model_id})
        logger.info(f"running benchmark #{benchmark_id} on {gpu_name} for {model_id}")
        metrics_thread = Thread(
            target=collect_metrics,
            args=[benchmark_id, continue_metric_collection, metrics_recorder],
        )
        metrics_thread.start()
        logger.info("started background thread to fetch device metrics")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence warnings when compiling

        device = "cuda"

        logger.info("downloading weights")
        # This is to avoid counting download in model load time measurement
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        gen_config = GenerationConfig(do_sample=False, top_p=1, temperature=1)
        logger.info("loading model")
        start = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, generation_config=gen_config
        ).eval()
        model.to(device)
        torch.cuda.synchronize()
        end = perf_counter()
        model_load_time = end - start
        logger.info(f"loaded model in: {model_load_time}s")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prompt = "Why dogs are so cute?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Specify the max length (including both the prompt and the response)
        # When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
        # with sequence length = `max_length`. The longer the more you will re-use it
        seq_length = inputs["input_ids"].shape[1]
        model.generation_config.max_length = seq_length + num_tokens_to_generate
        batch_size = inputs["input_ids"].shape[0]

        # Copied from the gpt-fast repo
        def multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
            q = torch.empty_like(probs_sort).exponential_(1)
            return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

        def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                pivot = v.select(-1, -1).unsqueeze(-1)
                logits = torch.where(logits < pivot, -float("Inf"), logits)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs

        def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
            probs = logits_to_probs(logits[0, -1], temperature, top_k)
            idx_next = multinomial_sample_one_no_sync(probs)
            return idx_next, probs

        # First eager forward pass
        logger.info("running first eager forward pass")
        start = perf_counter()
        outputs = model(**inputs)
        torch.cuda.synchronize()
        end = perf_counter()
        first_eager_fwd_pass_time = end - start
        logger.info(f"completed first eager forward pass in: {first_eager_fwd_pass_time}s")

        # Second eager forward pass (should be faster)
        logger.info("running second eager forward pass")
        start = perf_counter()
        outputs = model(**inputs)
        torch.cuda.synchronize()
        end = perf_counter()
        second_eager_fwd_pass_time = end - start
        logger.info(f"completed second eager forward pass in: {second_eager_fwd_pass_time}s")

        # First eager generation
        logger.info("running first eager generation")
        start = perf_counter()
        output = model.generate(**inputs)
        torch.cuda.synchronize()
        end = perf_counter()
        first_eager_generate_time = end - start
        logger.info(f"completed first eager generation in: {first_eager_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        # Second eager generation (should be faster)
        logger.info("running second eager generation")
        start = perf_counter()
        output = model.generate(**inputs)
        torch.cuda.synchronize()
        end = perf_counter()
        second_eager_generate_time = end - start
        logger.info(f"completed second eager generation in: {second_eager_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        logger.info("running generation timing loop")

        input_pos = torch.arange(0, seq_length, device=device)
        inputs = inputs["input_ids"]

        start = perf_counter()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits = model(inputs, position_ids=input_pos).logits
        next_token, probs = sample(logits, temperature=0.6, top_k=5)
        torch.cuda.synchronize()
        end = perf_counter()
        time_to_first_token = end - start

        input_pos = torch.tensor([seq_length], device=device, dtype=torch.int)
        next_token = next_token.clone()
        start = perf_counter()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits = model(next_token, position_ids=input_pos).logits
        next_token, probs = sample(logits, temperature=0.6, top_k=5)
        torch.cuda.synchronize()
        end = perf_counter()
        time_to_second_token = end - start

        input_pos = torch.tensor([seq_length + 1], device=device, dtype=torch.int)
        next_token = next_token.clone()
        start = perf_counter()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits = model(next_token, position_ids=input_pos).logits
        next_token, probs = sample(logits, temperature=0.6, top_k=5)
        torch.cuda.synchronize()
        end = perf_counter()
        time_to_third_token = end - start

        logger.info("running longer generation timing loop")

        total_time = 0
        for i in range(20):
            input_pos = torch.tensor([seq_length + 2 + i], device=device, dtype=torch.int)
            next_token = next_token.clone()
            start = perf_counter()
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                logits = model(next_token, position_ids=input_pos).logits
            next_token, probs = sample(logits, temperature=0.6, top_k=5)
            torch.cuda.synchronize()
            end = perf_counter()
            total_time += end - start

        mean_time_to_next_token = total_time / 20

        logger.info("running compilation benchmarks")

        # Now compile the model
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

        # StaticCache for generation
        with torch.device(device):
            model.setup_caches(max_batch_size=batch_size, max_seq_len=seq_length + num_tokens_to_generate)

        input_pos = torch.arange(0, seq_length, device=device)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]

        logger.info("compiling model")

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, generation_config=gen_config)
        model.to(device)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

        past_key_values = StaticCache(
            model.config,
            max_batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + 128,
        )
        # 1st call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values)
        end = perf_counter()
        first_compile_generate_time = end - start
        logger.info(f"completed first compile generation in: {first_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        past_key_values = StaticCache(
            model.config,
            max_batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + 128,
        )
        # 2nd call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values)
        end = perf_counter()
        second_compile_generate_time = end - start
        logger.info(f"completed second compile generation in: {second_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        past_key_values = StaticCache(
            model.config,
            max_batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + 128,
        )
        # 3rd call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values)
        end = perf_counter()
        third_compile_generate_time = end - start
        logger.info(f"completed third compile generation in: {third_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        past_key_values = StaticCache(
            model.config,
            max_batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + 128,
        )
        # 4th call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values)
        end = perf_counter()
        fourth_compile_generate_time = end - start
        logger.info(f"completed fourth compile generation in: {fourth_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        metrics_recorder.collect_model_measurements(
            benchmark_id,
            {
                "model_load_time": model_load_time,
                "first_eager_forward_pass_time_secs": first_eager_fwd_pass_time,
                "second_eager_forward_pass_time_secs": second_eager_fwd_pass_time,
                "first_eager_generate_time_secs": first_eager_generate_time,
                "second_eager_generate_time_secs": second_eager_generate_time,
                "time_to_first_token_secs": time_to_first_token,
                "time_to_second_token_secs": time_to_second_token,
                "time_to_third_token_secs": time_to_third_token,
                "time_to_next_token_mean_secs": mean_time_to_next_token,
                "first_compile_generate_time_secs": first_compile_generate_time,
                "second_compile_generate_time_secs": second_compile_generate_time,
                "third_compile_generate_time_secs": third_compile_generate_time,
                "fourth_compile_generate_time_secs": fourth_compile_generate_time,
            },
        )
    except Exception as e:
        logger.error(f"Caught exception: {e}")
    continue_metric_collection.set()
    if metrics_thread is not None:
        metrics_thread.join()
    
    # Only close the recorder if we created it locally
    if should_close_recorder:
        metrics_recorder.close() 