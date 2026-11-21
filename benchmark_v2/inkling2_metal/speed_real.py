"""Inkling-2-Mini decode speed on MPS from the real checkpoint (experts in their NVFP4 / MXFP8 banks), `generate` with
a static cache and torch.compile.   PYTHONPATH=src:benchmark_v2/inkling2_metal python .../speed_real.py [new_tokens]"""

import sys
import time

import inductor_mps_fixes
import torch

from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig


REPO = "thinkingmachines/Inkling-2-Mini"
inductor_mps_fixes.apply()
new_tokens = int(next((a for a in sys.argv[1:] if a.isdigit()), 128))
model = AutoModelForImageTextToText.from_pretrained(REPO, device_map="mps", dtype=torch.bfloat16).eval()


class Linear4(torch.nn.Module):
    """Experiment: a bf16 `nn.Linear` as 4-bit affine (MLX packing, groups of 64), batch-1 through `qmv4`."""

    def __init__(self, linear, group_size=64):
        super().__init__()
        from transformers.integrations.hub_kernels import metal_kernel
        from transformers.integrations.metal_quantization import _affine_dequantize_tensor, _affine_quantize_tensor

        self.qmv4, self.group_size = metal_kernel("metal-moe-decode").qmv4, group_size
        weight, scales, biases = _affine_quantize_tensor(linear.weight.float(), group_size, 4)
        self.weight, self.scales, self.biases = weight, scales.bfloat16(), biases.bfloat16()
        self.dense = _affine_dequantize_tensor(weight, scales, biases, group_size, 4).bfloat16()  # prefill only
        self.out_features = linear.out_features

    def forward(self, x):
        if x.numel() == x.shape[-1]:
            return self.qmv4(x.to(torch.bfloat16), self.weight, self.scales, self.biases, self.group_size).to(x.dtype)
        return torch.nn.functional.linear(x.to(torch.bfloat16), self.dense).to(x.dtype)


if "--dense4" in sys.argv:  # bf16 linears of the text model (and lm_head) -> 4-bit, the fp32 ones stay
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            text = name.startswith("model.language_model") or (name == "" and child_name == "lm_head")
            if type(child) is torch.nn.Linear and child.weight.dtype == torch.bfloat16 and text and "message_gate" not in name:
                if child.in_features % 64 == 0:
                    setattr(module, child_name, Linear4(child))
                    count += 1
    torch.mps.empty_cache()
    print(f"dense4: {count} linears", flush=True)


class Group4(torch.nn.Module):
    """The 4-bit siblings of one attention reading the same input: one qmv4 over their concatenated rows, each member
    slicing it; a member called with another input (k / v in the loop passes) runs its own rows only."""

    def __init__(self, members):
        super().__init__()
        first = members[0]
        self.qmv4, self.group_size = first.qmv4, first.group_size
        self.weight = torch.cat([m.weight for m in members])
        self.scales = torch.cat([m.scales for m in members])
        self.biases = torch.cat([m.biases for m in members])
        self.state = {}

    def run(self, x, start, end):
        if self.state.get("input") is not x:
            self.state["input"] = x
            self.state["output"] = self.qmv4(x, self.weight, self.scales, self.biases, self.group_size)
        return self.state["output"][..., start:end]


class Member4(torch.nn.Module):
    def __init__(self, group, linear, start, end):
        super().__init__()
        self.group, self.linear, self.start, self.end = group, linear, start, end

    def forward(self, x):
        if x.numel() != x.shape[-1]:
            return self.linear(x)
        x = x.to(torch.bfloat16)
        if self.group.state.get("input") is x or self.start == 0:
            return self.group.run(x, self.start, self.end)
        return self.linear(x)


if "--group" in sys.argv:
    groups = 0
    for name, module in list(model.named_modules()):
        if not name.endswith("self_attn") or not name.startswith("model.language_model"):
            continue
        paths = ["q_proj.linear", "o_proj.gate_proj", "relative_position_bias.linear", "sink_proj", "k_proj.linear", "v_proj.linear"]
        try:
            members = [(p, module.get_submodule(p)) for p in paths]
        except AttributeError:
            print(f"skip {name}: {[type(module.get_submodule(p.split('.')[0])).__name__ for p in paths]}", flush=True)
            continue
        if not all(isinstance(m, Linear4) for _, m in members):
            continue
        group, start = Group4([m for _, m in members]), 0
        for path, linear in members:
            parent, _, child = path.rpartition(".")
            owner = module.get_submodule(parent) if parent else module
            owner.register_module(child, Member4(group, linear, start, start + linear.out_features))
            start += linear.out_features
        groups += 1
    print(f"grouped {groups} attention projections", flush=True)

for module in model.modules():  # the relative bias table transposed once, not per decode step
    if type(module).__name__.endswith("RelativePositionBias"):
        module.proj_t = module.proj.detach().t().to(torch.bfloat16).contiguous()
tokenizer = AutoTokenizer.from_pretrained(REPO)
messages = [{"role": "user", "content": "Write a short story about a lighthouse keeper."}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = ids["input_ids"].to("mps")
compile_config = CompileConfig()
compile_config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": compile_config, "do_sample": False}


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


if __name__ == "__main__":
    with torch.inference_mode():
        for phase in ("compile", "warm"):
            start = time.perf_counter()
            model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
            print(f"{phase}: {time.perf_counter() - start:.1f}s", flush=True)
        stamps = Stamps()
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, streamer=stamps, **kwargs)
    steps = sorted(b - a for a, b in zip(stamps.t[2:], stamps.t[3:]))
    mean = sum(steps) / len(steps)
    print(f"Inkling-2-Mini (real, native fp4 experts), {new_tokens} tokens: median {1 / steps[len(steps) // 2]:.1f} tok/s, "
          f"mean {1 / mean:.1f} tok/s ({mean * 1e3:.2f} ms/token)", flush=True)  # fmt: skip
    print(repr(tokenizer.decode(out[0, ids.shape[1] :])[:300]), flush=True)
