"""Where a decode step's time goes: the compiled step re-measured with one component replaced by a no-op at a time
(outputs are garbage, the timing is the point). Loads the real checkpoint once.

    PYTHONPATH=src:benchmark_v2/inkling2_metal python benchmark_v2/inkling2_metal/ablate_real.py [--dense4]
"""

import sys
import time

import torch

import speed_real  # noqa: F401  loads the model (and applies --dense4 / --group), defines `model`, `ids`, `kwargs`
from transformers.integrations.metal_quantization import MetalFp4Experts
from transformers.models.inkling2_loop import modeling_inkling2_loop as m


model, ids, kwargs = speed_real.model, speed_real.ids, speed_real.kwargs
originals = {}


def patch(owner, name, fn):
    originals[(owner, name)] = getattr(owner, name)
    setattr(owner, name, fn)


def restore():
    for (owner, name), fn in originals.items():
        setattr(owner, name, fn)
    originals.clear()


ABLATIONS = {
    "none": lambda: None,
    "experts": lambda: patch(MetalFp4Experts, "forward", lambda self, h, i, w: torch.zeros_like(h)),
    "router": lambda: patch(
        m.Inkling2LoopTopKRouter,
        "forward",
        lambda self, h: (None, h.new_ones(h.shape[0], 8, dtype=torch.float32), torch.arange(8, device=h.device)[None]),
    ),
    "attention": lambda: patch(
        m, "inkling2_loop_attention", lambda q, k, v, *a, **kw: (q.transpose(1, 2), None)
    ),
    "attn_res": lambda: patch(
        m, "inkling2_loop_attention_residual", lambda b, p, q, mod: p.float().unsqueeze(-2).expand(*p.shape[:-1], q.reshape(-1, p.shape[-1]).shape[0], -1)
    ),
    "message_gate": lambda: patch(m, "inkling2_loop_message_gate", lambda h, msg, mod: h.float()),
    "short_conv": lambda: patch(m.Inkling2LoopShortConvolution, "forward", lambda self, h, **kw: h),
    "lm_head": lambda: patch(type(model.lm_head), "forward", lambda self, x: x.new_zeros(*x.shape[:-1], 201024)),
}


def step_ms(tokens=48):
    stamps = speed_real.Stamps()
    with torch.inference_mode():
        model.generate(ids, max_new_tokens=8, min_new_tokens=8, **kwargs)  # compile + warm
        model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, streamer=stamps, **kwargs)
    steps = sorted(b - a for a, b in zip(stamps.t[2:], stamps.t[3:]))
    return steps[len(steps) // 2] * 1e3


start = time.perf_counter()
base = None
for i, (name, apply) in enumerate(ABLATIONS.items()):
    torch._dynamo.reset()
    apply()
    try:
        ms = step_ms()
        base = ms if base is None else base
        print(f"[{i + 1}/{len(ABLATIONS)}] without {name:13s} {ms:6.2f} ms/token (saves {base - ms:+6.2f}) "
              f"| elapsed {time.perf_counter() - start:.0f}s", flush=True)  # fmt: skip
    except Exception as error:  # noqa: BLE001
        print(f"[{i + 1}/{len(ABLATIONS)}] without {name}: failed ({type(error).__name__}: {str(error)[:120]})", flush=True)
    restore()
