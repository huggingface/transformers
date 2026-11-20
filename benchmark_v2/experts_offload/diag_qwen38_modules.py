"""GPU time per module of an eager Qwen3.8 decode step (synthetic weights, `bench_qwen38_synthetic.build`): a sync
before and after each timed module, summed over layers by module path with the layer index dropped."""

import re
import time
from collections import defaultdict

import torch
from bench_qwen38_synthetic import build


model, text = build()
times, starts = defaultdict(float), {}
timed = re.compile(r"(^model\.[^.]+$|layers\.\d+\.[^.]+$|layers\.\d+\.mlp\.[^.]+$|^lm_head$)")


def pre(name):
    def hook(*_):
        torch.mps.synchronize()
        starts[name] = time.perf_counter()

    return hook


def post(name):
    key = re.sub(r"layers\.\d+\.", "layers.*.", name)

    def hook(*_):
        torch.mps.synchronize()
        times[key] += time.perf_counter() - starts[name]

    return hook


ids = torch.randint(0, 200_000, (1, 64), device="mps")
with torch.inference_mode():
    out = model(ids, use_cache=True)
    cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
    for _ in range(3):
        out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
    torch.mps.synchronize()
    start, steps = time.perf_counter(), 5
    for _ in range(steps):
        out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
    torch.mps.synchronize()
    print(f"eager step (no hooks): {(time.perf_counter() - start) / steps * 1e3:.1f} ms", flush=True)
    handles = [
        h
        for name, module in model.named_modules()
        if timed.search(name)
        for h in (module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name)))
    ]
    for _ in range(steps):
        out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
for key, seconds in sorted(times.items(), key=lambda kv: -kv[1]):
    print(f"{seconds / steps * 1e3:8.2f} ms  {key}")
