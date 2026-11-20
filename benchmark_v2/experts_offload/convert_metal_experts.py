# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "huggingface_hub[hf_xet]", "safetensors>=0.9.0rc1"]
# ///
"""Rewrite an FP8 MoE checkpoint with Metal affine-quantized, stacked expert banks for streaming on a Mac.

Experts `layers.L.mlp.experts.E.{gate,up,down}_proj.weight` (FP8, 128x128 block scales) become, per layer,
`layers.L.mlp.experts.{gate_up_proj,down_proj}` (uint32, packed) with `_scales` / `_qbiases`: the layout of
`MetalExperts` (`integrations/metal_quantization.py`), one contiguous row per expert so a miss is one read.
Everything else is copied unchanged and the config gets a quantization plan: Metal for the experts, the
checkpoint's own config for the rest.

  hf jobs uv run --flavor a100-large -s HF_TOKEN convert_metal_experts.py Qwen/Qwen3.8-Flash-Next-FP8 ArthurZ/...
"""

import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


source, target = sys.argv[1], sys.argv[2]
bits, group_size = int(os.environ.get("BITS", 4)), int(os.environ.get("GROUP_SIZE", 64))
device = "cuda" if torch.cuda.is_available() else "cpu"
out = "/tmp/out"
os.makedirs(out, exist_ok=True)


def log(msg):
    print(msg, flush=True)


os.system("df -h /tmp /root 2>/dev/null")
api = HfApi()
api.create_repo(target, private=True, exist_ok=True)


def upload(name):
    """Upload one finished file and delete it: a crash keeps what is done, the disk stays bounded."""
    api.upload_file(path_or_fileobj=f"{out}/{name}", path_in_repo=name, repo_id=target)
    os.remove(f"{out}/{name}")


start = time.perf_counter()
path = snapshot_download(source)
log(f"downloaded {source} in {time.perf_counter() - start:.0f}s")
index = json.load(open(f"{path}/model.safetensors.index.json", encoding="utf-8"))["weight_map"]
expert_key = re.compile(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight(_scale_inv)?$")


def dequant_fp8(weight, scale_inv, block=128):
    """FP8 weight `[N, K]` with block scales `[ceil(N/128), ceil(K/128)]` -> bf16."""
    scale = scale_inv.float().repeat_interleave(block, 0)[: weight.shape[0]].repeat_interleave(block, 1)
    return (weight.float() * scale[:, : weight.shape[1]]).to(torch.bfloat16)


def quantize(weight):
    """`[E, N, K]` -> packed uint32 `[E, N, K * bits / 32]`, scales and biases `[E, N, K / group_size]` (bf16)."""
    num_experts, rows, columns = weight.shape
    w = weight.float().reshape(-1, columns // group_size, group_size)
    w_min, w_max = w.amin(-1), w.amax(-1)
    scales = ((w_max - w_min) / (2**bits - 1)).clamp(min=1e-8)
    q = ((w - w_min[..., None]) / scales[..., None]).round().clamp(0, 2**bits - 1).to(torch.int32)
    q = q.reshape(-1, columns)
    per_int = 32 // bits
    packed = torch.zeros(q.shape[0], columns // per_int, dtype=torch.int32, device=q.device)
    for i in range(per_int):
        packed |= q[:, i::per_int] << (bits * i)
    shape = (num_experts, rows, -1)
    return packed.view(torch.uint32).reshape(shape), scales.bfloat16().reshape(shape), w_min.bfloat16().reshape(shape)


# 1. Group expert keys by layer
layers = defaultdict(dict)  # prefix -> {(expert, proj, is_scale): file}
for key, file in index.items():
    match = expert_key.match(key)
    if match:
        layers[match[1]][(int(match[2]), match[3], bool(match[4]))] = (key, file)
log(f"{len(layers)} expert layers, {len({e for e, _, _ in next(iter(layers.values()))})} experts each")

# 2. Per layer: dequantize, stack, quantize, write one shard
handles = {}


def tensor(key, file):
    if file not in handles:
        handles[file] = safe_open(f"{path}/{file}", "pt", device=device)
    return handles[file].get_tensor(key)


new_index = {}
started = time.perf_counter()
for done, (prefix, entries) in enumerate(
    sorted(layers.items(), key=lambda kv: int(re.search(r"layers\.(\d+)", kv[0])[1])), 1
):
    num_experts = max(e for e, _, _ in entries) + 1

    def bank(proj):
        return torch.stack(
            [
                dequant_fp8(tensor(*entries[(e, proj, False)]), tensor(*entries[(e, proj, True)]))
                if (e, proj, True) in entries
                else tensor(*entries[(e, proj, False)]).to(torch.bfloat16)  # not FP8: as is
                for e in range(num_experts)
            ]
        )

    gate_up = torch.cat([bank("gate_proj"), bank("up_proj")], dim=1)  # gate rows, then up rows
    tensors = {}
    for name, weight in (("gate_up_proj", gate_up), ("down_proj", bank("down_proj"))):
        packed, scales, biases = quantize(weight)
        tensors[f"{prefix}.{name}"] = packed.cpu().contiguous()
        tensors[f"{prefix}.{name}_scales"] = scales.cpu().contiguous()
        tensors[f"{prefix}.{name}_qbiases"] = biases.cpu().contiguous()
    shard = f"experts-{done:05d}.safetensors"
    save_file(tensors, f"{out}/{shard}", metadata={"format": "pt"})
    upload(shard)
    new_index.update(dict.fromkeys(tensors, shard))
    elapsed = time.perf_counter() - started
    eta = elapsed / done * (len(layers) - done)
    log(f"[{done}/{len(layers)}] {prefix} | elapsed {elapsed:.0f}s | eta ~{eta:.0f}s")

# 3. Copy every other tensor (dense layers keep their quantization), shard by shard
expert_keys = {key for entries in layers.values() for key, _ in entries.values()}
by_file = defaultdict(list)
for key, file in index.items():
    if key not in expert_keys:
        by_file[file].append(key)
for i, (file, keys) in enumerate(sorted(by_file.items()), 1):
    with safe_open(f"{path}/{file}", "pt") as f:
        save_file({k: f.get_tensor(k) for k in keys}, f"{out}/{file}", metadata={"format": "pt"})
    upload(file)
    new_index.update(dict.fromkeys(keys, file))
    log(f"[{i}/{len(by_file)}] copied {file} ({len(keys)} tensors)")

json.dump(
    {"metadata": {}, "weight_map": new_index},
    open(f"{out}/model.safetensors.index.json", "w", encoding="utf-8"),
    indent=2,
)

# 4. Config: a quantization plan, Metal for the experts and the checkpoint's own config for the rest
config = json.load(open(f"{path}/config.json", encoding="utf-8"))
holder = config.get("text_config", config) if "quantization_config" not in config else config
original = holder.pop("quantization_config", None)
metal = {"quant_method": "metal", "bits": bits, "group_size": group_size}
holder["quantization_config"] = {"*.mlp.experts": metal, **({"*": original} if original else {})}
json.dump(config, open(f"{out}/config.json", "w", encoding="utf-8"), indent=2)
for name in os.listdir(path):
    if not name.endswith(".safetensors") and name not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(f"{path}/{name}", f"{out}/{name}")

api.upload_folder(folder_path=out, repo_id=target)  # index, config, tokenizer files
log(f"uploaded to {target} in {(time.perf_counter() - start) / 60:.0f}m total")
