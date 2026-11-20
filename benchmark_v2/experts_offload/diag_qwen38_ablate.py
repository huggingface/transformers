"""Where the compiled Qwen3.8 decode step goes (synthetic weights): stub groups of linears / experts with zero outputs
(cache updates untouched, so compile stays valid) and time `generate` per variant. One model load, a fresh compile each."""

import re
import sys
import time

import torch
from bench_qwen38_synthetic import build

from transformers import CompileConfig
from transformers.integrations.metal_quantization import MetalExperts


model, text = build()
config = CompileConfig()
config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": config, "do_sample": False}
ids = torch.randint(0, 200_000, (1, 64), device="mps")
variants = {
    "baseline": None,
    "linear_attn proj": r"linear_attn\.(in_proj_\w+|out_proj)$",
    "self_attn proj+indexer": r"self_attn\.(\w_proj|indexer\.index_qk_proj)$",
    "hyper-connections": r"hyper_connection(_mixer)?\.\w+_weight\w*$",
    "shared expert": r"shared_expert(_gate|\.\w+_proj)$",
    "lm_head": r"^lm_head$",
    "experts": "experts",
    "all of the above": "all",
}
names = sys.argv[1:] or list(variants)


def stub(module):
    if isinstance(module, MetalExperts):
        return lambda h, i, w: torch.zeros_like(h)
    return lambda x: x.new_zeros(*x.shape[:-1], module.out_features)


def timed():
    torch._dynamo.reset()
    with torch.inference_mode():
        model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
        stamps = []
        streamer = type("S", (), {"put": lambda s, _: stamps.append(time.perf_counter()), "end": lambda s: None})()
        model.generate(ids, max_new_tokens=128, min_new_tokens=128, streamer=streamer, **kwargs)
    steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
    return steps[len(steps) // 2] * 1e3


base = None
for i, name in enumerate(names, 1):
    pattern, patched = variants[name], []
    for mod_name, module in model.named_modules():
        hit = pattern == "all" and any(re.search(p, mod_name) for p in list(variants.values())[1:-2] if p)
        hit |= pattern in ("experts", "all") and isinstance(module, MetalExperts)
        hit |= pattern not in (None, "experts", "all") and re.search(pattern, mod_name) is not None
        if hit and hasattr(module, "forward"):
            module.forward = stub(module)
            patched.append(module)
    ms = timed()
    base = ms if base is None else base
    print(f"[{i}/{len(names)}] {name:24s} {len(patched):4d} stubbed: {ms:6.2f} ms/token ({1e3 / ms:5.1f} tok/s), saves {base - ms:5.2f} ms", flush=True)
    for module in patched:
        del module.forward
