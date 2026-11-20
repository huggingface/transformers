"""Per-call cost of `generate` around the model: a warm compiled `generate` for 1 and for 9 new tokens (static cache),
median of 10 calls. Setup overhead = time(1 token) - one decode step."""

import statistics
import sys
import time

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig


repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
spec = sys.argv[2] if len(sys.argv) > 2 else "4g32c"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
if spec:
    from dense_quant import quantize_dense

    quantize_dense(model, spec)
config = CompileConfig()
config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": config, "do_sample": False}
ids = tok("Write a long story about a lighthouse keeper.", return_tensors="pt").input_ids.to("mps")


def timed(new_tokens):
    times = []
    with torch.inference_mode():
        for _ in range(12):
            torch.mps.synchronize()
            start = time.perf_counter()
            model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **kwargs)
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)
    return statistics.median(times[2:])


one, nine = timed(1), timed(9)
step = (nine - one) / 8
print(
    f"generate(1 token) {one * 1e3:.1f} ms | generate(9) {nine * 1e3:.1f} ms | decode step {step * 1e3:.2f} ms | "
    f"per-call overhead beyond one step ~{(one - step) * 1e3:.1f} ms (prefill of {ids.shape[1]} tokens included)",
    flush=True,
)
