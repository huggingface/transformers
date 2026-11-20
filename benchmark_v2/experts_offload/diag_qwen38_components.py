"""Compiled GPU cost of each Qwen3.8 decode component (synthetic weights, `bench_qwen38_synthetic.build`): capture every
module's inputs in a real static-cache decode step, then time each module alone, compiled, 200 calls back to back."""

import time

import torch
from bench_qwen38_synthetic import build

from transformers import StaticCache


model, text = build()
layers = model.model.layers
targets = {f"model.layers.{i}": layers[i] for i in (0, 3)}  # a linear-attention and a full-attention layer
targets |= {f"{p}.{n}": c for p, m in list(targets.items()) for n, c in m.named_children() if c is not None}
targets |= {"model.layers.0.mlp.experts": layers[0].mlp.experts, "lm_head": model.lm_head} | {f"model.{n}": c for n, c in model.model.named_children() if n != "layers"}
captured = {}


def capture(name):
    def hook(module, args, kwargs):
        captured[name] = (args, kwargs)

    return hook


handles = [m.register_forward_pre_hook(capture(n), with_kwargs=True) for n, m in targets.items()]
ids = torch.randint(0, 200_000, (1, 64), device="mps")
with torch.inference_mode():
    cache = StaticCache(config=model.config, max_cache_len=4096)  # room for the timed calls, which keep appending
    model.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False, past_key_values=cache)
for h in handles:
    h.remove()

rows = []
with torch.inference_mode():
    for name, module in targets.items():
        if name not in captured:
            continue
        args, kwargs = captured[name]
        fn = torch.compile(module, fullgraph=False, dynamic=False)
        for _ in range(3):
            fn(*args, **kwargs)
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            fn(*args, **kwargs)
        torch.mps.synchronize()
        rows.append(((time.perf_counter() - start) / 100 * 1e3, name, type(module).__name__))
        print(f"{rows[-1][0]:7.3f} ms  {name}  ({rows[-1][2]})", flush=True)
