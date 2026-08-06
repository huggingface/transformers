import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, Mxfp4Config
from transformers.distributed import DistributedConfig


kind = os.environ["MODEL_KIND"]
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
output_dir = os.environ.get("OUTPUT_DIR", "benchmark_results")

if kind == "qwen3_moe":
    model_id = "Qwen/Qwen3-30B-A3B"
    model_kwargs = {"dtype": torch.bfloat16}
elif kind == "llama":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_kwargs = {"dtype": torch.bfloat16}
elif kind == "gpt_oss_mxfp4":
    model_id = "openai/gpt-oss-20b"
    model_kwargs = {
        "dtype": torch.bfloat16,
        "quantization_config": Mxfp4Config(dequantize=False),
        "use_kernels": False,
    }
else:
    raise ValueError(f"Unknown MODEL_KIND: {kind}")

torch.cuda.set_device(local_rank)
start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(
        tp_size=world_size,
        enable_expert_parallel=kind != "llama",
    ),
    **model_kwargs,
)
torch.cuda.synchronize()
local_load_time = time.perf_counter() - start

local_load_time_tensor = torch.tensor(local_load_time, device=f"cuda:{local_rank}", dtype=torch.float64)
rank_load_time_tensors = [torch.empty_like(local_load_time_tensor) for _ in range(world_size)]
dist.all_gather(rank_load_time_tensors, local_load_time_tensor)
all_rank_load_times = [rank_time.item() for rank_time in rank_load_time_tensors]
load_time = max(all_rank_load_times)

if rank == 0:
    legacy_tp = os.environ.get("TRANSFORMERS_USE_LEGACY_TP", "0")
    tp_backend = "legacy" if legacy_tp == "1" else "dtensor"

if rank == 0:
    payload = {
        "model": model_id,
        "tp_backend": tp_backend,
        "legacy_tp": legacy_tp,
        "world_size": world_size,
        "load_time_s": load_time,
        "median_rank_load_time_s": statistics.median(all_rank_load_times),
        "all_rank_load_times_s": all_rank_load_times,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{kind}_{tp_backend}_tp{world_size}_load.json")
    with open(output_path, "w") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(payload, indent=2))
    print(f"Saved loading benchmark results to {output_path}")

del model
dist.destroy_process_group()
