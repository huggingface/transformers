"""Where the compiled MPS decode step goes: tok/s per 128-token window over 1024 tokens (the sustained drop), then the
top ops of a few compiled decode steps (torch.profiler, host-side)."""

import sys
import time

import torch
from dense_quant import quantize_dense

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig


repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
spec = sys.argv[2] if len(sys.argv) > 2 else "4g32c"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
quantize_dense(model, spec)
config = CompileConfig()
config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": config, "do_sample": False}
ids = tok.apply_chat_template(
    [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False,
).input_ids.to("mps")


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


with torch.inference_mode():
    model.generate(ids, max_new_tokens=32, **kwargs)  # compile
    stamps = Stamps()
    model.generate(ids, max_new_tokens=1024, min_new_tokens=1024, streamer=stamps, **kwargs)
steps = [b - a for a, b in zip(stamps.t[2:], stamps.t[3:])]
for w in range(0, len(steps), 128):
    window = steps[w : w + 128]
    print(
        f"tokens {w:4d}-{w + len(window) - 1:4d}: {len(window) / sum(window):6.2f} tok/s ({sum(window) / len(window) * 1e3:.2f} ms)"
    )

with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    model.generate(ids, max_new_tokens=8, min_new_tokens=8, **kwargs)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
