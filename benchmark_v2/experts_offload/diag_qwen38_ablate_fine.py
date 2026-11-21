"""Fine-grained cost of each kernel group in the compiled Qwen3.8 decode step (synthetic weights): stub one group with a
near-free stand-in (cache updates untouched, so compile stays valid), recompile, time 128 tokens. Saved ms = its cost."""

import re
import sys
import time

import torch
from bench_qwen38_synthetic import build

from transformers import CompileConfig
from transformers.integrations import metal_linear_attention as mla
from transformers.integrations import metal_qmv
from transformers.integrations import metal_quantization as mq
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


model, text = build()
config = CompileConfig()
config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": config, "do_sample": False}
ids = torch.randint(0, 200_000, (1, 64), device="mps")
linears = {n: m for n, m in model.named_modules() if isinstance(m, mq.MetalLinear)}


def zeros_linear(module):
    return lambda x: x.new_zeros(*x.shape[:-1], module.out_features)


def patch_linears(pattern):
    hits = [m for n, m in linears.items() if re.search(pattern, n)]
    for m in hits:
        m.forward = zeros_linear(m)
    return lambda: [m.__dict__.pop("forward", None) for m in hits]


def patch_attr(owner, name, value):
    old = getattr(owner, name)
    setattr(owner, name, value)
    return lambda: setattr(owner, name, old)


VARIANTS = {
    "baseline": lambda: (lambda: None),
    "linear_attn in_proj (fused qkv+z+b+a)": lambda: patch_linears(r"linear_attn\.in_proj_qkv\.fused$"),
    "linear_attn out_proj": lambda: patch_linears(r"linear_attn\.out_proj$"),
    "attention q/k/v/indexer (fused)": lambda: patch_linears(r"self_attn\.q_proj\.fused$"),
    "attention o_proj": lambda: patch_linears(r"self_attn\.o_proj$"),
    "hyper down+inject (fused)": lambda: patch_linears(r"input_mix_weight_down\.fused$"),
    "hyper up": lambda: patch_linears(r"input_mix_weight_up$"),
    "shared expert gate+up+gate (fused)": lambda: patch_linears(r"shared_expert\.gate_proj\.fused$"),
    "shared expert down": lambda: patch_linears(r"shared_expert\.down_proj$"),
    "lm_head": lambda: patch_linears(r"^lm_head$"),
    "experts gate/up (MLX gather_qmm)": lambda: patch_attr(
        mq.MetalExperts, "_qmm", lambda self, x, name, ids, s: x.new_zeros(x.shape[0], getattr(self, name).shape[-2])
    ),
    "experts down + routing sum": lambda: patch_attr(
        metal_qmv, "experts_down_sum", lambda act, down, *a: act.new_zeros(1, down.shape[-2])
    ),
    "gated delta rule kernel": lambda: patch_attr(
        mla, "_gdr", lambda q, k, v, g, beta, state, l2: torch.zeros_like(v)
    ),
    "causal conv1d step kernel": lambda: patch_attr(mla, "_conv_step", lambda x, state, w, b, silu: x.clone()),
    "sdpa (full attention)": lambda: patch_attr(
        ALL_ATTENTION_FUNCTIONS,
        "_global_mapping",
        {**ALL_ATTENTION_FUNCTIONS._global_mapping, "sdpa": lambda m, q, k, v, *a, **kw: (q.transpose(1, 2), None)},
    ),
}
names = sys.argv[1:] or list(VARIANTS)


def timed():
    torch._dynamo.reset()
    with torch.inference_mode():
        model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
        stamps = []
        streamer = type("S", (), {"put": lambda s, _: stamps.append(time.perf_counter()), "end": lambda s: None})()
        model.generate(ids, max_new_tokens=128, min_new_tokens=128, streamer=streamer, **kwargs)
    steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
    return steps[len(steps) // 2] * 1e3


base, start = None, time.perf_counter()
for i, name in enumerate(names, 1):
    undo = VARIANTS[name]()
    ms = timed()
    undo()
    base = ms if base is None else base
    elapsed = time.perf_counter() - start
    print(
        f"[{i}/{len(names)}] {name:40s} {ms:6.2f} ms/token  saves {base - ms:5.2f} ms"
        f"  | elapsed {elapsed / 60:.1f}m eta ~{elapsed / i * (len(names) - i) / 60:.1f}m",
        flush=True,
    )
