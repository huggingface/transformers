import cProfile
import json
import os
import pstats
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from transformers.distributed import DistributedConfig


kind = os.environ["MODEL_KIND"]
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
batch_size = int(os.environ.get("BATCH_SIZE", 1))
new_tokens = int(os.environ.get("NEW_TOKENS", 256))
warmups = int(os.environ.get("WARMUPS", 2))
repeats = int(os.environ.get("REPEATS", 5))
output_dir = os.environ.get("OUTPUT_DIR", "benchmark_results")
profile_enabled = os.environ.get("PROFILE", "0") == "1"
if kind == "qwen3_moe":
    model_id = "Qwen/Qwen3-30B-A3B"
    model_kwargs = {
        "dtype": torch.bfloat16,
    }
elif kind == "llama":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_kwargs = {
        "dtype": torch.bfloat16,
    }
elif kind == "gpt_oss_mxfp4":
    model_id = "openai/gpt-oss-20b"
    model_kwargs = {
        "dtype": torch.bfloat16,
        "quantization_config": Mxfp4Config(dequantize=False),
        "use_kernels": False,
    }
else:
    raise ValueError(kind)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(
        tp_size=world_size,
        enable_expert_parallel=kind != "llama",
    ),
    **model_kwargs,
).eval()
# region agent log
if os.environ.get("TRANSFORMERS_DEBUG_DENSE_LOCAL_TP") == "1":
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module._hf_tp_requires_local_tensors = True
# endregion
prompts = ["Explain tensor parallelism in detail."] * batch_size
inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
generation_kwargs = {
    "max_new_tokens": new_tokens,
    "min_new_tokens": new_tokens,
    "do_sample": False,
    "use_cache": True,
}


@torch.inference_mode()
def run():
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model.generate(**inputs, **generation_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    elapsed = torch.tensor(elapsed, device=model.device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    peak_memory = torch.tensor(torch.cuda.max_memory_allocated(), device=model.device, dtype=torch.float64)
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    generated = batch_size * (output.shape[1] - inputs.input_ids.shape[1])
    return elapsed.item(), generated, peak_memory.item()


# region agent log
@torch.inference_mode()
def profile_run():
    dist.barrier()
    torch.cuda.synchronize()
    python_profiler = cProfile.Profile()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as torch_profiler:
        python_profiler.enable()
        model.generate(**inputs, **generation_kwargs)
        python_profiler.disable()
        torch.cuda.synchronize()

    if rank != 0:
        return None

    def event_data(event):
        return {
            "name": event.key,
            "count": event.count,
            "self_cpu_ms": event.self_cpu_time_total / 1000,
            "self_device_ms": getattr(event, "self_device_time_total", 0.0) / 1000,
            "device_total_ms": getattr(event, "device_time_total", 0.0) / 1000,
        }

    events = list(torch_profiler.key_averages())
    top_ops = sorted(
        events,
        key=lambda event: event.self_cpu_time_total + getattr(event, "self_device_time_total", 0.0),
        reverse=True,
    )[:40]
    collective_terms = ("nccl", "all_reduce", "all_gather", "reduce_scatter", "redistribute", "c10d")
    collectives = [
        event_data(event)
        for event in events
        if any(term in event.key.lower() for term in collective_terms)
    ]

    stats = pstats.Stats(python_profiler)
    python_entries = []
    for (filename, line, function), (_, calls, self_time, cumulative_time, _) in stats.stats.items():
        python_entries.append(
            {
                "location": f"{filename}:{line}:{function}",
                "calls": calls,
                "self_cpu_s": self_time,
                "cumulative_cpu_s": cumulative_time,
            }
        )

    return {
        "dtensor_parameter_count": sum(isinstance(param, DTensor) for param in model.parameters()),
        "top_torch_ops": [event_data(event) for event in top_ops],
        "collective_ops": sorted(
            collectives,
            key=lambda event: event["self_cpu_ms"] + event["self_device_ms"],
            reverse=True,
        ),
        "top_python_functions": sorted(
            python_entries, key=lambda entry: entry["cumulative_cpu_s"], reverse=True
        )[:40],
    }
# endregion


for _ in range(warmups):
    run()
results = [run() for _ in range(repeats)]
profile_summary = profile_run() if profile_enabled else None
if rank == 0:
    latencies = [result[0] for result in results]
    generated = results[0][1]
    legacy_tp = os.environ.get("TRANSFORMERS_USE_LEGACY_TP", "0")
    tp_backend = "legacy" if legacy_tp == "1" else "dtensor"
    payload = {
        "model": model_id,
        "tp_backend": tp_backend,
        "legacy_tp": legacy_tp,
        "world_size": world_size,
        "batch_size": batch_size,
        "new_tokens": new_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "median_latency_s": statistics.median(latencies),
        "median_tokens_per_second": generated / statistics.median(latencies),
        "peak_memory_gib": max(r[2] for r in results) / 2**30,
        "all_latencies_s": latencies,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{kind}_{tp_backend}_tp{world_size}_bs{batch_size}_tokens{new_tokens}.json",
    )
    with open(output_path, "w") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark results to {output_path}")
    if profile_summary is not None:
        # region agent log
        with open(
            "/fsx/ferdinandmom/ferdinand-hf/transformers_pr/work/split_pr/b-pr-0-dual-path-tp/.cursor/debug-dd0465.log",
            "a",
        ) as debug_file:
            debug_file.write(
                json.dumps(
                    {
                        "sessionId": "dd0465",
                        "runId": tp_backend,
                        "hypothesisId": "L1,L2,L3,L4,L5",
                        "location": "bench_dtensor.py:profile_run",
                        "message": "Dense TP generation profile",
                        "data": profile_summary,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
        # endregion
dist.destroy_process_group()
