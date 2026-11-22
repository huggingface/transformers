"""Hill-climb harness for Inkling-2-Mini decode on MPS (transformers generate, static cache, torch.compile).

    PYTHONPATH=src:benchmark_v2/inkling2_metal python benchmark_v2/inkling2_metal/hillclimb/hc_bench.py configs/<name>.json

Config: {"name": str, "dense": "exact" | "4bit", "env": {VAR: value}, "tokens": 128, "samples": 6}.
Refuses to run on battery or with the GPU busy before the run; records power and GPU state after, voiding the run if they
changed. Decode speed is the median of per-step times over `samples` generations (after compile + warm-up); one line
per run lands in ledger.jsonl next to this file.
"""

import json
import os
import statistics
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))


def on_ac():
    return "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout


def gpu_busy():
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    marker = '"Device Utilization %"='
    return int(out.split(marker)[1].split(",")[0].split("}")[0]) if marker in out else -1


SIBLINGS = [
    "q_proj.linear",
    "o_proj.gate_proj",
    "relative_position_bias.linear",
    "sink_proj",
]  # all read `hidden_states`


def group_siblings(model):
    """Lever HC_GROUP: each attention's projections reading the same input as one matmul over their concatenated rows
    (bf16 `nn.Linear` or 4-bit `MetalLinear`); the first member called on an input computes it, the others slice."""
    import torch
    from torch import nn

    from transformers.integrations.metal_quantization import MetalLinear

    class Member(nn.Module):
        def __init__(self, fused, state, start, end):
            super().__init__()
            self.fused, self.state, self.start, self.end = fused, state, start, end

        def forward(self, x):
            if self.state.get("input") is not x:
                self.state["input"], self.state["output"] = x, self.fused(x)
            return self.state["output"][..., self.start : self.end]

    groups = 0
    for name, attn in list(model.named_modules()):
        if not name.startswith("model.language_model") or not name.endswith("self_attn"):
            continue
        members = [attn.get_submodule(path) for path in SIBLINGS]
        kinds = {type(m) for m in members}
        if kinds == {MetalLinear} and len({(m.bits, m.group_size) for m in members}) == 1:
            first = members[0]
            fused = MetalLinear(first.in_features, sum(m.out_features for m in members), bits=first.bits,
                                group_size=first.group_size)  # fmt: skip
            for param in ("weight", "scales", "qbiases"):
                tensor = torch.cat([getattr(m, param) for m in members])
                setattr(fused, param, nn.Parameter(tensor, requires_grad=False))
        elif kinds == {nn.Linear} and all(m.bias is None for m in members):
            fused = nn.Linear(members[0].in_features, sum(m.out_features for m in members), bias=False)
            fused.weight = nn.Parameter(torch.cat([m.weight for m in members]), requires_grad=False)
        else:
            continue
        fused = fused.to(members[0].weight.device)
        state, start = {}, 0
        for path, member in zip(SIBLINGS, members):
            parent, _, child = path.rpartition(".")
            owner = attn.get_submodule(parent) if parent else attn
            owner.register_module(child, Member(fused, state, start, start + member.out_features))
            start += member.out_features
        groups += 1
    return groups


def main(path):
    cfg = json.load(open(path))
    os.environ.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    if not on_ac():
        sys.exit("refusing to run on battery")
    busy_before = gpu_busy()
    if busy_before > 20:
        sys.exit(f"refusing to run: GPU {busy_before}% busy")

    import inductor_mps_fixes
    import torch

    from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig, MetalConfig

    inductor_mps_fixes.apply()
    repo = "thinkingmachines/Inkling-2-Mini"
    kwargs = {"device_map": "mps", "dtype": torch.bfloat16}
    if cfg.get("dense", "4bit") == "4bit":
        metal = MetalConfig(bits=4, group_size=64)
        kwargs["quantization_config"] = {"model.visual.*": None, "*_message_gate": None, "lm_head": metal, "*": metal}
    model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs).eval()
    for module in model.modules():
        if type(module).__name__.endswith("RelativePositionBias"):
            module.proj_t = module.proj.detach().t().to(torch.bfloat16).contiguous()
    if os.environ.get("HC_GROUP") == "1":
        print(f"grouped {group_siblings(model)} attention projection groups", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    prompt = [{"role": "user", "content": "Write a short story about a lighthouse keeper."}]
    ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = ids["input_ids"].to("mps")
    compile_config = CompileConfig()
    compile_config._compile_all_devices = True
    gen = {"cache_implementation": "static", "compile_config": compile_config, "do_sample": False}
    tokens, samples = cfg.get("tokens", 128), cfg.get("samples", 6)

    stamps = []

    class Stamps:
        def put(self, _):
            stamps.append(time.perf_counter())

        def end(self):
            pass

    run_medians, text = [], ""
    with torch.inference_mode():
        for _ in range(2):  # compile + warm
            model.generate(ids, max_new_tokens=16, min_new_tokens=16, **gen)
        for i in range(samples):
            stamps.clear()
            out = model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, streamer=Stamps(), **gen)
            steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
            run_medians.append(1 / steps[len(steps) // 2])
            print(f"[{i + 1}/{samples}] {run_medians[-1]:.1f} tok/s", flush=True)
            text = tokenizer.decode(out[0, ids.shape[1] :])
    busy_after, ac_after = gpu_busy(), on_ac()
    words = text.split()
    looping = len(words) > 20 and len(set(words[-20:])) < 5
    # thermal throttling shows as a steady decay over the samples: void such a run
    throttled = run_medians[-1] < 0.95 * run_medians[0]
    record = {
        "config": cfg["name"], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "decode_tok_s": statistics.median(run_medians),
        "samples": [round(v, 2) for v in run_medians], "gpu_busy_before": busy_before, "gpu_busy_after_pct": busy_after,
        "ac": ac_after, "throttled": throttled, "void": not ac_after or throttled, "looping_output": looping, "text_head": text[:120],
    }  # fmt: skip
    with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
