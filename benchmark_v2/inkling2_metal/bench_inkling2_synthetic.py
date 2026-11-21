"""Inkling-2-Mini decode speed on Metal without its weights: the real text config (24 layers, 6-18 looped 3x = 48
layer evaluations per token, 128 experts top-8, vocab 201k) with random weights, 4-bit experts (`MetalExperts`) and
4-bit dense linears (`MetalLinear`), through `generate` with a static cache and torch.compile.

    PYTHONPATH=src python benchmark_v2/inkling2_metal/bench_inkling2_synthetic.py [new_tokens] [--eager] [--profile]
        [--experts fp4]   # checkpoint-native experts (NVFP4 layers 0-19, MXFP8 20-23) instead of 4-bit affine
        [--dense bf16]    # keep the dense linears bf16 (the checkpoint's) instead of 4-bit affine

The weights are built once (dense initialized by the model then quantized, experts random packed nibbles) and saved to
`WEIGHTS`; later runs load them onto a meta skeleton.
"""

import os
import sys
import time

import inductor_mps_fixes
import torch
from safetensors.torch import load_file, save_file

from transformers import AutoConfig, AutoModelForCausalLM, CompileConfig
from transformers.integrations.metal_quantization import (
    MetalExperts,
    MetalFp4Experts,
    MetalLinear,
    _affine_quantize_tensor,
    fuse_sibling_linears,
)


REPO = "thinkingmachines/Inkling-2-Mini"
GROUP, EXPERT_GROUP = 32, 64


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


EXPERTS, DENSE = _arg("--experts", "affine4"), _arg("--dense", "affine4")
WEIGHTS = os.path.expanduser(f"~/.cache/inkling2-mini-synthetic-{EXPERTS}-{DENSE}.safetensors")
MXFP8_LAYERS = range(20, 24)  # the checkpoint's last 4 layers keep MXFP8 experts, the others NVFP4


def quantizable(name, linear):
    """bf16 `nn.Linear`s go 4-bit; the fp32 ones (router), the thin ones (message gates' rank-64 projection, which
    casts its input to the weight dtype, attention sinks) and odd shapes stay as they are."""
    return (
        DENSE == "affine4"
        and type(linear) is torch.nn.Linear
        and linear.weight.dtype == torch.bfloat16
        and linear.in_features % GROUP == 0
        and linear.out_features >= 128
    )


def skeleton(linears=True):
    config = AutoConfig.from_pretrained(REPO).text_config
    config.architectures = ["Inkling2LoopForCausalLM"]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        for name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                if child_name == "experts" and EXPERTS == "fp4":
                    fmt = "mxfp8" if int(name.split(".")[2]) in MXFP8_LAYERS else "nvfp4"
                    module.register_module(child_name, MetalFp4Experts(child, fmt))
                elif child_name == "experts":
                    module.register_module(child_name, MetalExperts(child, bits=4, group_size=EXPERT_GROUP))
                elif linears and quantizable(child_name, child):
                    module.register_module(
                        child_name, MetalLinear(child.in_features, child.out_features, bias=False, bits=4, group_size=GROUP)
                    )
    return model.eval(), config


def build():
    if not os.path.exists(WEIGHTS):
        model, _ = skeleton(linears=False)
        model.to_empty(device="mps")
        torch.manual_seed(0)
        model.init_weights()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dtype == torch.uint32:  # expert banks: random nibbles, constant scales / biases
                    param.copy_(torch.randint(0, 2**32, param.shape, dtype=torch.int64, device="mps").to(torch.uint32))
                elif ".experts." in name and param.dtype in (torch.uint8, torch.float8_e4m3fn):
                    # fp4 banks at a ~0.02 weight scale: NVFP4 codes * 0.25 * 0.028, MXFP8 N(0, 2) * 2^-7
                    if name.endswith("_scale"):  # NVFP4 e4m3 block scales, as bytes
                        param.data = torch.full(param.shape, 0.25, device="mps").to(torch.float8_e4m3fn).view(torch.uint8)
                    elif name.endswith("_scale_inv"):  # MXFP8 e8m0 block scales
                        param.data = torch.full(param.shape, 120, dtype=torch.uint8, device="mps")
                    elif param.dtype == torch.uint8:
                        param.data = torch.randint(0, 256, param.shape, dtype=torch.uint8, device="mps")
                    else:
                        param.data = (torch.randn(param.shape, device="mps") * 2).to(torch.float8_e4m3fn)
                elif name.endswith("_scale_2"):
                    param.data.fill_(0.028)
                elif name.endswith(("_scales", "_qbiases")):
                    param.data = torch.full_like(param, 0.002 if name.endswith("_scales") else -0.015, dtype=torch.bfloat16)
            count = 0
            for _, module in list(model.named_modules()):
                for child_name, child in list(module.named_children()):
                    if quantizable(child_name, child):
                        q = MetalLinear(child.in_features, child.out_features, bias=False, bits=4, group_size=GROUP)
                        packed, scales, biases = _affine_quantize_tensor(child.weight.float(), GROUP, 4)
                        q.weight = torch.nn.Parameter(packed, requires_grad=False)
                        q.scales = torch.nn.Parameter(scales.bfloat16(), requires_grad=False)
                        q.qbiases = torch.nn.Parameter(biases.bfloat16(), requires_grad=False)
                        setattr(module, child_name, q)
                        count += 1
        print(f"quantized {count} linears", flush=True)
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, WEIGHTS)
        del model
        torch.mps.empty_cache()
    model, config = skeleton()
    model.load_state_dict(load_file(WEIGHTS, device="mps"), strict=True, assign=True)
    for module in model.modules():  # non-persistent buffers are not in the file
        if module.__class__.__name__.endswith("RotaryEmbedding"):
            module.__init__(module.config)
            module.to("mps")
    missing = [n for n, t in (*model.named_parameters(), *model.named_buffers()) if t.is_meta]
    assert not missing, missing[:5]
    if "--no-fuse" not in sys.argv:
        print(f"fused {fuse_sibling_linears(model)} sibling projection groups", flush=True)
    print(f"loaded {WEIGHTS}, resident {torch.mps.current_allocated_memory() / 2**30:.1f} GiB", flush=True)
    return model, config


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


if __name__ == "__main__":
    new_tokens = int(next((a for a in sys.argv[1:] if a.isdigit()), 128))
    model, config = build()
    kwargs = {"cache_implementation": "static", "do_sample": False}
    inductor_mps_fixes.apply()
    if "--cpp-wrapper" in sys.argv:  # launch the compiled kernels from C++ instead of the Python wrapper
        torch._inductor.config.cpp_wrapper = True
    if "--eager" not in sys.argv:
        compile_config = CompileConfig()
        compile_config._compile_all_devices = True
        kwargs["compile_config"] = compile_config
    else:
        kwargs["disable_compile"] = True
    ids = torch.randint(0, 200_000, (1, 64), device="mps")
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
        f"Inkling-2-Mini synthetic (experts {EXPERTS}, dense {DENSE}), {new_tokens} tokens: median {1 / steps[len(steps) // 2]:.1f} tok/s, "
        f"mean {1 / mean:.1f} tok/s ({mean * 1e3:.2f} ms/token)",
        flush=True,
    )
    if "--profile" in sys.argv:
        from torch._inductor import metrics

        print(f"inductor generated kernels: {metrics.generated_kernel_count}", flush=True)
        # host time of one decode step vs its wall time: equal -> the GPU waits on the CPU
        with torch.inference_mode():
            stamps = Stamps()
            model.generate(ids, max_new_tokens=12, min_new_tokens=12, streamer=stamps, **kwargs)
        with (
            torch.inference_mode(),
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof,
        ):
            model.generate(ids, max_new_tokens=6, min_new_tokens=6, **kwargs)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25, max_name_column_width=70), flush=True)
