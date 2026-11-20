# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "accelerate", "kernels"]
# ///
"""Which copy breaks CUDA graph capture in Qwen3-MoE's compiled decode step? Random weights, 8 layers."""

import time
import traceback

import torch

from transformers import AutoConfig, AutoModelForCausalLM


torch._inductor.config.graph_partition = True
config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
config.num_hidden_layers = 8
for impl in ("grouped_mm", "batched_mm"):
    config._experts_implementation = impl
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    ids = torch.randint(0, 1000, (1, 32), device="cuda")
    for attempt in ("generate static", "compile forward"):
        try:
            with torch.inference_mode():
                if attempt == "generate static":
                    kwargs = {"cache_implementation": "static", "max_new_tokens": 24, "do_sample": False}
                    model.generate(ids, **kwargs)  # warm up + capture
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model.generate(ids, **kwargs)
                    torch.cuda.synchronize()
                    print(f"{impl} {attempt}: {(time.perf_counter() - start) / 24 * 1e3:.2f} ms/token", flush=True)
                else:
                    step = torch.compile(model.forward, mode="reduce-overhead")
                    for _ in range(3):
                        step(ids[:, :1])
                    print(f"{impl} {attempt}: ok", flush=True)
        except Exception:
            print(f"{impl} {attempt}: FAILED", flush=True)
            traceback.print_exc()
        torch._dynamo.reset()
    del model
