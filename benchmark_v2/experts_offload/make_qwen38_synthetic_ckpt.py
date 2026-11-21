"""A loadable Qwen3.8 checkpoint with all 512 experts and random weights, for the experts-streaming cache table on a Mac
(no real weights here): the text model in transformers' native format, dense weights bf16 (taken from the 64-expert
synthetic build, initialized by the model), experts as 4-bit Metal banks (g64), one safetensors file per layer.

    PYTHONPATH=src:. python make_qwen38_synthetic_ckpt.py [out_dir]       # ~80 GB written, one layer in memory at a time

Then e.g. `bench_experts_offload.py --model <out_dir> --configs ours:0.5+compile --quantize-dense 4g32`.
"""

import json
import os
import sys
import time

import torch
from bench_qwen38_synthetic import skeleton
from safetensors.torch import save_file


out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/qwen38-synthetic-512e")
os.makedirs(out, exist_ok=True)
EXPERTS, GROUP = 512, 64

# Dense weights: the 64-expert skeleton without dense quantization, materialized and initialized by the model
model, text = skeleton(64, linears=False)
model.to_empty(device="mps")
torch.manual_seed(0)
model.init_weights()
dense = {
    k: v for k, v in model.state_dict().items() if ".mlp.experts." not in k and not k.endswith(".mlp.gate.weight")
}
hidden, inter = text.hidden_size, text.moe_intermediate_size
del model
torch.mps.empty_cache()

text.num_experts = EXPERTS
text.quantization_config = {"*.mlp.experts": {"quant_method": "metal", "bits": 4, "group_size": GROUP}}
text.architectures = ["Qwen4ExpForCausalLM"]
text.save_pretrained(out)


def bank(rows, columns):
    """Random packed 4-bit bank [EXPERTS, rows, columns / 8] + constant bf16 scales / biases."""
    packed = torch.randint(-(2**31), 2**31, (EXPERTS, rows, columns // 8), dtype=torch.int32).view(torch.uint32)
    scales = torch.full((EXPERTS, rows, columns // GROUP), 0.002, dtype=torch.bfloat16)
    return packed, scales, torch.full_like(scales, -0.015)


weight_map, start = {}, time.perf_counter()
layers = text.num_hidden_layers
for i in range(layers):
    prefix = f"model.layers.{i}."
    shard = {k: v.cpu().contiguous() for k, v in dense.items() if k.startswith(prefix)}
    for name, (rows, columns) in {"gate_up_proj": (2 * inter, hidden), "down_proj": (hidden, inter)}.items():
        packed, scales, biases = bank(rows, columns)
        shard |= {f"{prefix}mlp.experts.{name}": packed, f"{prefix}mlp.experts.{name}_scales": scales}
        shard[f"{prefix}mlp.experts.{name}_qbiases"] = biases
    shard[f"{prefix}mlp.gate.weight"] = torch.randn(EXPERTS, hidden, dtype=torch.bfloat16) * 0.02
    file = f"layer-{i:02d}.safetensors"
    save_file(shard, os.path.join(out, file), metadata={"format": "pt"})
    weight_map |= dict.fromkeys(shard, file)
    elapsed = time.perf_counter() - start
    print(
        f"[{i + 1}/{layers}] {file} | elapsed {elapsed:.0f}s eta ~{elapsed / (i + 1) * (layers - i - 1):.0f}s",
        flush=True,
    )
rest = {k: v.cpu().contiguous() for k, v in dense.items() if not k.startswith("model.layers.")}
save_file(rest, os.path.join(out, "rest.safetensors"), metadata={"format": "pt"})
weight_map |= dict.fromkeys(rest, "rest.safetensors")
with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f)
print(f"done: {out}", flush=True)
