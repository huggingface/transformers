# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert an RWKV-7 checkpoint to the `Rwkv7ForCausalLM` layout.

Two input flavours:

* **native**: the upstream `.pth` from `BlinkDL/RWKV-LM`. Because this port keeps
  the reference parameter names, the conversion is a prefix rename
  (`blocks.` -> `rwkv7.blocks.`) with no per-tensor mapping table.
* **fla**: the `safetensors` layout published on the Hub by the `flash-linear-attention`
  port, which renames the projections and stores each LoRA as a pair of
  `nn.Linear`s. Those linears are the transpose of the reference's raw factors, so
  the mapping below transposes them back. Verified against
  `rwkv7-g1d-0.1b` / `rwkv7-0.1b-fla`: 31/31 tensors agree on shape.

Two details in the `fla` direction are easy to get wrong and are handled here:

1. `fla` squeezes some per-channel parameters to `(C,)` (`k_k`, `k_a`, `ffn.x_k`)
   but keeps others at `(1, 1, C)` (`x_r` … `x_g`). Everything is reshaped to the
   reference's `(1, 1, C)`.
2. `fla` drops the value-residual LoRA on layer 0 entirely, while a native
   checkpoint carries it (unused, since layer 0 produces `v_first` rather than mixing
   towards it). Converting from `fla` therefore leaves layer 0's `v0/v1/v2` at
   their initial values, which is correct precisely because they are never read.

Usage:
    python convert_rwkv7_checkpoint_to_hf.py \
        --checkpoint RWKV-x070-World-0.1B.pth --flavour native \
        --config config.json --output_dir ./rwkv7-0.1b-hf
"""

import argparse
import json
import os
import re

import torch
from safetensors.torch import save_file

from transformers.models.rwkv7 import Rwkv7Config, Rwkv7ForCausalLM


# fla suffix -> (reference suffix, transpose?)
_FLA_SUFFIX_MAP = {
    "r_proj.weight": ("receptance.weight", False),
    "k_proj.weight": ("key.weight", False),
    "v_proj.weight": ("value.weight", False),
    "o_proj.weight": ("output.weight", False),
    "g_norm.weight": ("ln_x.weight", False),
    "g_norm.bias": ("ln_x.bias", False),
}
for _chain in ("w", "a", "g", "v"):
    _FLA_SUFFIX_MAP[f"{_chain}_lora.lora.0.weight"] = (f"{_chain}1", True)
    _FLA_SUFFIX_MAP[f"{_chain}_lora.lora.2.weight"] = (f"{_chain}2", True)
    _FLA_SUFFIX_MAP[f"{_chain}_lora.lora.2.bias"] = (f"{_chain}0", False)

# parameters the reference keeps at (1, 1, C)
_CHANNEL_PARAMS = {"x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "k_k", "k_a", "w0", "a0", "v0"}


def _convert_native(state_dict):
    out = {}
    for key, tensor in state_dict.items():
        if key == "emb.weight":
            out["rwkv7.emb.weight"] = tensor
        elif key == "head.weight":
            out["head.weight"] = tensor
        elif key.startswith("ln_out."):
            out["rwkv7." + key] = tensor
        elif key.startswith("blocks."):
            out["rwkv7." + key] = tensor
        else:
            raise KeyError(f"unrecognised native key {key!r}")
    return out


def _convert_fla(state_dict):
    out = {}
    for key, tensor in state_dict.items():
        if key == "model.embeddings.weight":
            out["rwkv7.emb.weight"] = tensor
            continue
        if key == "lm_head.weight":
            out["head.weight"] = tensor
            continue
        if key.startswith("model.norm."):
            out["rwkv7.ln_out." + key.split(".")[-1]] = tensor
            continue
        match = re.match(r"model\.layers\.(\d+)\.(.+)", key)
        if match is None:
            raise KeyError(f"unrecognised fla key {key!r}")
        layer, rest = match.group(1), match.group(2)
        prefix = f"rwkv7.blocks.{layer}."

        if rest.startswith("pre_norm."):
            out[prefix + "ln0." + rest.split(".")[-1]] = tensor
            continue
        if rest.startswith("attn_norm."):
            out[prefix + "ln1." + rest.split(".")[-1]] = tensor
            continue
        if rest.startswith("ffn_norm."):
            out[prefix + "ln2." + rest.split(".")[-1]] = tensor
            continue
        if rest.startswith("ffn."):
            name = rest[len("ffn.") :]
            if name == "x_k":
                tensor = tensor.reshape(1, 1, -1)
            out[prefix + "ffn." + name] = tensor
            continue
        if not rest.startswith("attn."):
            raise KeyError(f"unrecognised fla key {key!r}")

        name = rest[len("attn.") :]
        if name in _FLA_SUFFIX_MAP:
            target, transpose = _FLA_SUFFIX_MAP[name]
            if transpose:
                tensor = tensor.t().contiguous()
            if target in _CHANNEL_PARAMS:
                tensor = tensor.reshape(1, 1, -1)
            out[prefix + "att." + target] = tensor
            continue
        # plain per-channel parameters carried over unchanged (modulo shape)
        if name in _CHANNEL_PARAMS:
            out[prefix + "att." + name] = tensor.reshape(1, 1, -1)
        elif name == "r_k":
            out[prefix + "att.r_k"] = tensor
        else:
            raise KeyError(f"unrecognised fla attn parameter {name!r} (from {key!r})")
    return out


def convert(checkpoint, flavour, config_path, output_dir, dtype="float32"):
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

    converted = _convert_native(state_dict) if flavour == "native" else _convert_fla(state_dict)

    if config_path is not None:
        with open(config_path) as handle:
            config = Rwkv7Config(**json.load(handle))
    else:
        config = _infer_config(converted)

    # Check the key set against a meta-device skeleton rather than a real model:
    # a 7B checkpoint would otherwise need the full weights twice over in RAM.
    with torch.device("meta"):
        skeleton = Rwkv7ForCausalLM(config)
    expected = set(skeleton.state_dict())
    unexpected = sorted(set(converted) - expected)
    # layer 0's value-residual LoRA is legitimately absent from `fla` checkpoints
    missing = sorted(k for k in expected - set(converted) if not re.search(r"blocks\.0\.att\.v[012]$", k))
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    # Shapes too, not just names. The skeleton is right here and its shapes are already
    # read three lines below to fill the absent tensors, so comparing them costs
    # nothing -- and without it a config that disagrees with the checkpoint converts
    # with zero reported mismatches and produces a model that loads and generates
    # noise. A hand-written `config.json` that gets one width wrong is the ordinary way
    # to reach that, and the widths are exactly what a hand-written config gets wrong.
    reference = skeleton.state_dict()
    wrong_shape = {
        key: (tuple(converted[key].shape), tuple(reference[key].shape))
        for key in sorted(set(converted) & expected)
        if tuple(converted[key].shape) != tuple(reference[key].shape)
    }
    if wrong_shape:
        listed = "\n".join(f"  {k}: checkpoint {got}, config implies {want}" for k, (got, want) in wrong_shape.items())
        raise RuntimeError(
            f"{len(wrong_shape)} tensor(s) do not match the shapes this config implies:\n{listed}\n"
            "The config and the checkpoint describe different models; re-check --config, "
            "or omit it and let the shapes be inferred."
        )
    for key in expected - set(converted):  # fill the legitimately-absent ones
        converted[key] = torch.zeros(skeleton.state_dict()[key].shape)

    target = getattr(torch, dtype)
    converted = {k: v.to(target).contiguous() for k, v in converted.items()}

    os.makedirs(output_dir, exist_ok=True)
    save_file(converted, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["Rwkv7ForCausalLM"]
    config.dtype = dtype
    config.save_pretrained(output_dir)
    total = sum(v.numel() for v in converted.values())
    print(f"wrote {output_dir} ({total / 1e6:.1f}M params, {dtype})")


def _infer_config(converted):
    """Derive the config from tensor shapes when none is supplied."""
    vocab, hidden = converted["rwkv7.emb.weight"].shape
    layers = 1 + max(int(m.group(1)) for k in converted if (m := re.match(r"rwkv7\.blocks\.(\d+)\.", k)))
    heads, head_dim = converted["rwkv7.blocks.0.att.r_k"].shape
    # `fla` checkpoints carry no value-residual LoRA on layer 0 (it is never read
    # there), so read its rank from the first layer that has one.
    v_rank_key = next(k for k in (f"rwkv7.blocks.{i}.att.v1" for i in range(layers)) if k in converted)
    return Rwkv7Config(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_heads=heads,
        head_dim=head_dim,
        decay_low_rank_dim=converted["rwkv7.blocks.0.att.w1"].shape[1],
        a_low_rank_dim=converted["rwkv7.blocks.0.att.a1"].shape[1],
        v_low_rank_dim=converted[v_rank_key].shape[1],
        gate_low_rank_dim=converted["rwkv7.blocks.0.att.g1"].shape[1],
        intermediate_size=converted["rwkv7.blocks.0.ffn.key.weight"].shape[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help=".pth (native) or .safetensors (fla)")
    parser.add_argument("--flavour", choices=["native", "fla"], default="native")
    parser.add_argument("--config", default=None, help="optional config.json; inferred otherwise")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    convert(args.checkpoint, args.flavour, args.config, args.output_dir, args.dtype)
