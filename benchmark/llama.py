import argparse
import json
import logging
import os
import sys
from statistics import mean
from threading import Event, Thread
from time import perf_counter, sleep

import gpustat
import psutil
import psycopg2
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.set_float32_matmul_precision("high")


def parse_arguments():
    """
    Parse command line arguments for the benchmarking CLI.
    """
    parser = argparse.ArgumentParser(description="CLI for benchmarking the huggingface/transformers.")

    parser.add_argument(
        "branch",
        type=str,
        help="The branch name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_id",
        type=str,
        help="The commit hash on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_msg",
        type=str,
        help="The commit message associated with the commit, truncated to 70 characters.",
    )

    args = parser.parse_args()

    return args.branch, args.commit_id, args.commit_msg


def collect_metrics(benchmark_id, continue_metric_collection):
    p = psutil.Process(os.getpid())
    conn = psycopg2.connect("dbname=metrics")
    cur = conn.cursor()
    while not continue_metric_collection.is_set():
        with p.oneshot():
            cpu_util = p.cpu_percent()
            mem_megabytes = p.memory_info().rss / (1024 * 1024)
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_util = gpu_stats[0]["utilization.gpu"]
        gpu_mem_megabytes = gpu_stats[0]["memory.used"]
        cur.execute(
            "INSERT INTO device_measurements (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes) VALUES (%s, %s, %s, %s, %s)",
            (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes),
        )
        sleep(0.01)
        conn.commit()
    conn.close()


def run_benchmark(branch: str, commit_id: str, commit_msg: str, num_tokens_to_generate=100):
    continue_metric_collection = Event()
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_name = gpu_stats[0]["name"]
    conn = psycopg2.connect("dbname=metrics")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO benchmarks (branch, commit_id, commit_message, gpu_name) VALUES (%s, %s, %s, %s) RETURNING benchmark_id",
        (branch, commit_id, commit_msg, gpu_name),
    )
    conn.commit()
    benchmark_id = cur.fetchone()[0]
    metrics_thread = Thread(target=collect_metrics, args=[benchmark_id, continue_metric_collection])
    metrics_thread.start()


    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence warnings when compiling

    device = "cuda"
    ckpt = "meta-llama/Llama-2-7b-hf"

    # This is to avoid counting download in model load time measurement
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
    gen_config = GenerationConfig(do_sample=False, top_p=1, temperature=1)
    start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, generation_config=gen_config).eval()
    model.to(device)
    end = perf_counter()
    model_load_time = end - start
    logger.info(f"loaded model in: {model_load_time}s")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    prompt = "Why dogs are so cute?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Specify the max length (including both the prompt and the response)
    # When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
    # with sequence length = `max_length`. The longer the more you will re-use it
    seq_length = inputs["input_ids"].shape[1]
    model.generation_config.max_length = seq_length + num_tokens_to_generate
    batch_size = inputs["input_ids"].shape[0]

    #########
    # Eager #
    #########
    with torch.no_grad():
        past_key_values = StaticCache(
            model.config,
            batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + num_tokens_to_generate,
        )
        cache_position = torch.arange(seq_length, device=device)
        start = perf_counter()
        model(
            **inputs,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )
        end = perf_counter()
        first_eager_fwd_pass_time = end - start
        logger.info(f"completed first eager fwd pass in: {first_eager_fwd_pass_time}s")
        start = perf_counter()
        output = model.generate(**inputs, do_sample=False)
        end = perf_counter()
        first_eager_generate_time = end - start
        logger.info(f"completed first eager generation in: {first_eager_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        past_key_values = StaticCache(
            model.config,
            batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + num_tokens_to_generate,
        )
        cache_position = torch.arange(seq_length, device=device)
        start = perf_counter()
        model(
            **inputs,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )
        end = perf_counter()
        second_eager_fwd_pass_time = end - start
        logger.info(f"completed second eager fwd pass in: {second_eager_fwd_pass_time}s")
        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        second_eager_generate_time = end - start
        logger.info(f"completed second eager generation in: {second_eager_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        torch.compiler.reset()

        ################
        # Forward pass #
        ################

        # `torch.compile(model, ...)` is not recommended as you compile callbacks
        # and full generate. We recommend compiling only the forward for now.
        # "reduce-overhead" will use cudagraphs.
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

        past_key_values = StaticCache(
            model.config,
            batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + num_tokens_to_generate,
        )
        cache_position = torch.arange(seq_length, device=device)
        start = perf_counter()
        logits = model(
            **inputs,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )[0]
        end = perf_counter()
        time_to_first_token = end - start
        logger.info(f"completed first compile generation in: {time_to_first_token}s")
        cache_position = torch.tensor([seq_length], device=device)
        next_token_times_secs = []
        for _ in range(1, num_tokens_to_generate):
            next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
            start = perf_counter()
            logits = model(
                next_token,
                position_ids=cache_position.unsqueeze(-1),
                cache_position=cache_position,
                return_dict=False,
                use_cache=True,
            )[0]
            end = perf_counter()
            next_token = torch.argmax(logits, dim=-1)
            next_token_times_secs.append(end[:,-1:] - start)

            logger.info(f"completed next compile generation in: {next_token_times_secs[-1]}s")
            cache_position += 1
        time_to_second_token = next_token_times_secs[0]
        time_to_third_token = next_token_times_secs[1]
        mean_time_to_next_token = mean(next_token_times_secs[2:])

        ####################
        # Generate compile #
        ####################
        torch.compiler.reset()
        # model.generate = torch.compile(model.generate, mode="reduce-overhead", fullgraph=True)

        past_key_values = StaticCache(
            model.config,
            batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + num_tokens_to_generate,
        )

        # 1st call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values, do_sample=False)
        end = perf_counter()
        first_compile_generate_time = end - start
        logger.info(f"completed first compile generation in: {first_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        # 2nd call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values, do_sample=False)
        end = perf_counter()
        second_compile_generate_time = end - start
        logger.info(f"completed second compile generation in: {second_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        # 3nd call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values, do_sample=False)
        end = perf_counter()
        third_compile_generate_time = end - start
        logger.info(f"completed second compile generation in: {third_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        # 4th call
        start = perf_counter()
        output = model.generate(**inputs, past_key_values=past_key_values, do_sample=False)
        end = perf_counter()
        fourth_compile_generate_time = end - start
        logger.info(f"completed second compile generation in: {fourth_compile_generate_time}s")
        logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

    cur.execute(
        """
        INSERT INTO model_measurements (
            benchmark_id,
            measurements,
        ) VALUES (%s, %s)
        """,
        (
            benchmark_id,
            json.dumps(
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
                }
            ),
        ),
    )
    conn.commit()
    conn.close()
    continue_metric_collection.set()
    metrics_thread.join()


if __name__ == "__main__":
    branch, commit_id, commit_msg = parse_arguments()
    run_benchmark(branch, commit_id, commit_msg, num_tokens_to_generate=20)
