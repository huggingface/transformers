# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "accelerate", "kernels", "huggingface_hub[hf_xet]"]
# ///
"""Profile #50717's decode step on CUDA: host Python (cProfile) and device/ops (torch.profiler)."""

import cProfile
import glob
import pstats
import time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AutoModelForCausalLM, DynamicCache


path = snapshot_download("Qwen/Qwen3-30B-A3B", allow_patterns=["*.json", "*.safetensors"])
expert_bytes = 0
for file in glob.glob(f"{path}/*.safetensors"):
    with safe_open(file, "pt") as f:
        for key in f.keys():
            if ".experts." in key:
                expert_bytes += torch.Size(f.get_slice(key).get_shape()).numel() * 2
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, device_map="cuda", experts_offload={"cache_bytes": expert_bytes // 2}
)
ids = torch.randint(0, 1000, (1, 64), device="cuda")


def step(token, cache):
    return model(token, past_key_values=cache).logits[:, -1:].argmax(-1)


with torch.inference_mode():
    cache = DynamicCache(config=model.config)
    token = model(ids, past_key_values=cache).logits[:, -1:].argmax(-1)
    for _ in range(20):  # warm the cache
        token = step(token, cache)
    torch.cuda.synchronize()
    start = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(12):
        token = step(token, cache)
    torch.cuda.synchronize()
    profiler.disable()
    print(f"{(time.perf_counter() - start) / 12 * 1e3:.1f} ms / token under cProfile", flush=True)
    pstats.Stats(profiler).sort_stats("tottime").print_stats(30)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(3):
            token = step(token, cache)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30), flush=True)
