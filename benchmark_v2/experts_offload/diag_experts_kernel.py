# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "accelerate", "kernels"]
# ///
"""Where does a Qwen3-30B-A3B decode step go on this GPU? Random weights, 8 layers, per experts implementation."""

import time

import torch

from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache


config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
config.num_hidden_layers = 8
print(torch.cuda.get_device_name(), torch.__version__, flush=True)
for impl in ("eager", "grouped_mm", "batched_mm"):
    config._experts_implementation = impl
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    ids = torch.randint(0, 1000, (1, 32), device="cuda")
    with torch.inference_mode():
        cache = DynamicCache(config=config)
        token = model(ids, past_key_values=cache).logits[:, -1:].argmax(-1)
        times = []
        for _ in range(40):
            torch.cuda.synchronize()
            start = time.perf_counter()
            token = model(token, past_key_values=cache).logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        ms = 1e3 * sorted(times[5:])[len(times[5:]) // 2]
        print(f"{impl:12s} {ms:7.2f} ms / token for 8 layers -> {ms * 6:7.1f} ms for 48 layers", flush=True)
        if impl == "grouped_mm":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                token = model(token, past_key_values=cache).logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25), flush=True)
    del model
    torch.cuda.empty_cache()
