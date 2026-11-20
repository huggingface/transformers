"""What verifying k tokens in one forward (speculative decoding) costs, Qwen3.8 synthetic on Metal. Proxy with the same
weight traffic: a compiled `generate` decode step over a batch of k sequences (dense weights read once, the union of
the k tokens' experts). Prints ms(k) and the accepted tokens per step a 100 tok/s target needs."""

import time

import torch
from bench_qwen38_synthetic import build

from transformers import CompileConfig


model, text = build()
config = CompileConfig()
config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": config, "do_sample": False}
base = None
with torch.inference_mode():
    for k in (1, 2, 4, 8):
        ids = torch.randint(0, 200_000, (k, 64), device="mps")
        model.generate(ids, max_new_tokens=8, min_new_tokens=8, **kwargs)
        stamps = []
        streamer = type("S", (), {"put": lambda s, _: stamps.append(time.perf_counter()), "end": lambda s: None})()
        model.generate(ids, max_new_tokens=64, min_new_tokens=64, streamer=streamer, **kwargs) if k == 1 else None
        if k > 1:  # streamers take batch 1: time the whole generate minus prefill instead
            torch.mps.synchronize(); t0 = time.perf_counter()
            model.generate(ids, max_new_tokens=1, min_new_tokens=1, **kwargs); torch.mps.synchronize()
            t1 = time.perf_counter()
            model.generate(ids, max_new_tokens=65, min_new_tokens=65, **kwargs); torch.mps.synchronize()
            ms = ((time.perf_counter() - t1) - (t1 - t0)) / 64 * 1e3
        else:
            steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
            ms = steps[len(steps) // 2] * 1e3
        base = base or ms
        print(f"verify {k} tokens: {ms:6.2f} ms ({ms / base:.2f}x one); 100 tok/s needs {ms / 10:.2f} accepted/step", flush=True)
