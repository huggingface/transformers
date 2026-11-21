"""Qwen3.8-Flash-Next decode speed on Metal without its weights: the real config (dims, 48 layers, layer types,
top-10 routing, vocab) with random weights, 4-bit experts (Metal `gather_qmm`) + 4-bit always-on layers, through
`generate` with a static cache and torch.compile.

Two shrinks keep it in memory without changing a decode step's work: 64 experts instead of 512 (a token still runs
its top-10), and a small n-gram embedding vocab (a lookup costs the same). Experts streaming is not exercised.

    PYTHONPATH=src:. python bench_qwen38_synthetic.py [num_experts] [new_tokens]
"""

import os
import sys
import time

import torch
from dense_quant import quantizable, quantize_dense
from safetensors.torch import load_file, save_file

from transformers import AutoConfig, AutoModelForCausalLM, CompileConfig
from transformers.integrations.metal_quantization import MetalExperts, MetalLinear, fuse_sibling_linears


WEIGHTS = os.path.expanduser("~/.cache/qwen38-synthetic-{}e-all4bit.safetensors")


def skeleton(num_experts, linears=True):
    """Meta model with `MetalExperts` (4-bit g64) for the experts and, with `linears`, `MetalLinear` (4-bit g32) for
    the `nn.Linear`s `quantize_dense` converts."""
    cfg = AutoConfig.from_pretrained("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit")
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and hasattr(c, "quantization_config"):
            delattr(c, "quantization_config")
    text = getattr(cfg, "text_config", cfg)
    text.num_experts, text.ngram_vocab_size_base = num_experts, 128_000
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        for name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                if f"{name}.{child_name}".endswith("mlp.experts"):
                    module.register_module(child_name, MetalExperts(child, bits=4, group_size=64))
                elif linears and quantizable(child_name, child):
                    bias = child.bias is not None
                    module.register_module(
                        child_name,
                        MetalLinear(child.in_features, child.out_features, bias=bias, bits=4, group_size=32),
                    )
    return model.eval(), text


def build(num_experts=64):
    """Random weights at the real dims, made once and saved to `WEIGHTS`; later runs load them onto the skeleton."""
    path = WEIGHTS.format(num_experts)
    if not os.path.exists(path):
        model, _ = skeleton(num_experts, linears=False)
        model.to_empty(device="mps")
        torch.manual_seed(0)
        model.init_weights()  # dense params, norms, buffers (n-gram offsets)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dtype == torch.uint32:
                    param.copy_(torch.randint(0, 2**32, param.shape, dtype=torch.int64, device="mps").to(torch.uint32))
                elif name.endswith(("_scales", "_qbiases")):
                    param.data = torch.full_like(
                        param, 0.002 if name.endswith("_scales") else -0.015, dtype=torch.bfloat16
                    )
        quantize_dense(model, "4g32")
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, path)
        del model
        torch.mps.empty_cache()
    model, text = skeleton(num_experts)
    model.load_state_dict(load_file(path, device="mps"), strict=True, assign=True)
    for module in model.modules():  # non-persistent buffers are not in the file
        if module.__class__.__name__.endswith("RotaryEmbedding"):
            module.__init__(module.config)
            module.to("mps")
    assert not any(t.is_meta for t in (*model.parameters(), *model.buffers()))
    if os.environ.get("ATTN"):
        model.set_attn_implementation(os.environ["ATTN"])
    if "--no-fuse" not in sys.argv:
        print(f"fused {fuse_sibling_linears(model)} sibling projection groups", flush=True)
    print(f"loaded {path}, resident {torch.mps.current_allocated_memory() / 2**30:.1f} GiB", flush=True)
    return model, text


if __name__ == "__main__":
    num_experts = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    model, text = build(num_experts)
    # `--ablate=experts,linear_attn,...`: replace those modules by no-ops to measure what they cost in the compiled step
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    hc, hidden = text.hc_count, text.hidden_size
    noops = {
        "experts": (MetalExperts, lambda self, h, i, w: torch.zeros_like(h)),
        "shared_expert": (m.Qwen4ExpTextMLP, lambda self, x: torch.zeros_like(x)),
        "linear_attn": (m.Qwen4ExpTextGatedDeltaNet, lambda self, h, **k: h),
        "self_attn": (m.Qwen4ExpTextAttention, lambda self, h, *a, **k: (h, None)),
        "hyper": (
            m.Qwen4ExpTextGatedResidual,
            lambda self, h: (h.unflatten(-1, (hc, hidden)).mean(-2), h, h.new_ones(*h.shape[:-1], hc))
            if self.block_inject_weight is not None
            else h.unflatten(-1, (hc, hidden)).mean(-2),
        ),
    }
    for arg in sys.argv:
        for name in arg.removeprefix("--ablate=").split(",") if arg.startswith("--ablate=") else ():
            noops[name][0].forward = noops[name][1]
            print(f"ablated {name}", flush=True)
    config = CompileConfig()
    config._compile_all_devices = True
    kwargs = {"cache_implementation": "static", "compile_config": config, "do_sample": False}
    ids = torch.randint(0, 200_000, (1, 64), device="mps")

    class Stamps:
        def __init__(self):
            self.t = []

        def put(self, _):
            self.t.append(time.perf_counter())

        def end(self):
            pass

    with torch.inference_mode():
        for phase in ("compile", "warm"):
            start = time.perf_counter()
            model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
            print(f"{phase}: {time.perf_counter() - start:.1f}s", flush=True)
        stamps = Stamps()
        model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, streamer=stamps, **kwargs)
    steps = sorted(b - a for a, b in zip(stamps.t[2:], stamps.t[3:]))
    mean = sum(steps) / len(steps)
    print(
        f"Qwen3.8 synthetic ({num_experts} experts, top-{text.num_experts_per_tok}), {new_tokens} tokens: "
        f"median {1 / steps[len(steps) // 2]:.1f} tok/s, mean {1 / mean:.1f} tok/s ({mean * 1e3:.2f} ms/token)",
        flush=True,
    )

    if "--profile" in sys.argv:
        from torch._dynamo.utils import counters

        print("graph breaks:", dict(counters["graph_break"]) or 0, flush=True)
        # Each Metal command waits for completion, so an op's CPU time ~ its GPU time (+ a fixed commit cost)
        with (
            torch.inference_mode(),
            torch.mps.profiler.profile(mode="interval", wait_until_completed=True),
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof,
        ):
            model.generate(ids, max_new_tokens=9, min_new_tokens=9, **kwargs)
        table = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=40)
        print(table, flush=True)
