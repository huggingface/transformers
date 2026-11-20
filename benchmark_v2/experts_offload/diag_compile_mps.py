"""Why is torch.compile slower on MPS? eager dynamic cache vs eager static cache vs static cache + compile, with dynamo's
graph-break and recompile counters. Timing through `generate` + a streamer (same as the bench)."""

import collections
import sys
import time

import torch
import torch._dynamo

from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer


repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
spec = sys.argv[2] if len(sys.argv) > 2 else "4g32c"
print("torch", torch.__version__, flush=True)
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
quantize_dense(model, spec)
ids = tok.apply_chat_template(
    [{"role": "user", "content": "Explain how a CPU cache hierarchy works."}],
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


def run(name, **kwargs):
    for attempt in ("warmup", "timed"):
        stamps = Stamps()
        with torch.inference_mode():
            model.generate(ids, max_new_tokens=128, min_new_tokens=128, do_sample=False, streamer=stamps, **kwargs)
        torch.mps.synchronize()
    steps = sorted(b - a for a, b in zip(stamps.t[2:], stamps.t[3:]))
    print(
        f"{name:34s} {1 / steps[len(steps) // 2]:6.2f} tok/s (median step {steps[len(steps) // 2] * 1e3:.2f} ms)",
        flush=True,
    )


run("eager, dynamic cache")
torch._dynamo.utils.counters.clear()
run("static cache + compile (generate default)", cache_implementation="static")
breaks = torch._dynamo.utils.counters["graph_break"]
print(f"graph breaks: {sum(breaks.values())} ({len(breaks)} distinct)", flush=True)
for reason, count in collections.Counter(breaks).most_common(8):
    print(f"  {count:4d} x {reason.splitlines()[0][:160]}", flush=True)
print("recompiles / frames:", dict(torch._dynamo.utils.counters["stats"]), flush=True)

# What generate does not do on MPS: compile the forward ourselves (static shapes through the static cache)
for mode in ("default", "max-autotune-no-cudagraphs"):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    eager_forward = model.forward
    model.forward = torch.compile(eager_forward, mode=mode, dynamic=False)
    run(f"static cache + torch.compile({mode})", cache_implementation="static", disable_compile=True)
    breaks = torch._dynamo.utils.counters["graph_break"]
    print(f"  graph breaks: {sum(breaks.values())}, frames: {dict(torch._dynamo.utils.counters['stats'])}", flush=True)
    for reason, count in collections.Counter(breaks).most_common(6):
        print(f"  {count:4d} x {reason.splitlines()[0][:160]}", flush=True)
    model.forward = eager_forward
