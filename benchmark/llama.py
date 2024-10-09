import argparse
import os
from threading import Event, Thread
from time import perf_counter, sleep

import gpustat
import psutil
import psycopg2
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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


def run_benchmark(branch: str, commit_id: str, commit_msg: str):
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

    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence warnings when compiling

        device = "cuda:1"
        ckpt = "bert-base-uncased"

        # This is to avoid counting download in model load time measurement
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
        gen_config = GenerationConfig(do_sample=False, top_p=1, temperature=1)
        start = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, generation_config=gen_config)
        model.to(device)
        end = perf_counter()
        model_load_time = end - start

        tokenizer = AutoTokenizer.from_pretrained(ckpt)

        prompt = "Why dogs are so cute?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Specify the max length (including both the prompt and the response)
        # When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
        # with sequence length = `max_length`. The longer the more you will re-use it
        model.generation_config.max_length = 128

        # without `torch.compile`: each call takes ~ 5.0 seconds (on A100 80G + torch 2.3)
        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        first_eager_fwd_pass_time = end - start
        # TODO:
        # response = tokenizer.batch_decode(outputs)[0]
        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        second_eager_fwd_pass_time = end - start

        torch.compiler.reset()

        # `torch.compile(model, ...)` is not recommended as you compile callbacks
        # and full generate. We recommend compiling only the forward for now.
        # "reduce-overhead" will use cudagraphs.
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        model.generation_config.cache_implementation = "static"

        # with `torch.compile` (on A100 80G + torch 2.3)
        # 1st call: ~ 90 seconds
        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        first_compile_fwd_pass_time = end - start
        # TODO:
        # response = tokenizer.batch_decode(outputs)[0]

        # 2nd call: ~ 60 seconds
        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        second_compile_fwd_pass_time = end - start
        # TODO:
        # response = tokenizer.batch_decode(outputs)[0]

        # 3nd call: ~ 1.5 seconds
        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        third_compile_fwd_pass_time = end - start

        start = perf_counter()
        model.generate(**inputs, do_sample=False)
        end = perf_counter()
        fourth_compile_fwd_pass_time = end - start

        cur.execute(
            """
            INSERT INTO model_measurements (
                benchmark_id,
                model_load_time,
                first_eager_forward_pass_time_secs,
                second_eager_forward_pass_time_secs,
                first_compile_forward_pass_time_secs,
                second_compile_forward_pass_time_secs,
                third_compile_forward_pass_time_secs,
                fourth_compile_forward_pass_time_secs
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                benchmark_id,
                model_load_time,
                first_eager_fwd_pass_time,
                second_eager_fwd_pass_time,
                first_compile_fwd_pass_time,
                second_compile_fwd_pass_time,
                third_compile_fwd_pass_time,
                fourth_compile_fwd_pass_time,
            ),
        )
        conn.commit()
        conn.close()
        # TODO:
        # response = tokenizer.batch_decode(outputs)[0]
    except Exception as e:
        print(f"error: {e}")

    continue_metric_collection.set()
    metrics_thread.join()


if __name__ == "__main__":
    branch, commit_id, commit_msg = parse_arguments()
    run_benchmark(branch, commit_id, commit_msg)
