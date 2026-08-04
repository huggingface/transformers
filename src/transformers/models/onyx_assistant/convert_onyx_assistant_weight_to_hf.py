# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Convert an Onyx Drafter checkpoint to a Hugging Face transformers checkpoint.

Usage:
    python src/transformers/models/onyx/convert_onyx_weights_to_hf.py \\
        --checkpoint_path        pytorch_weights/dflash/head.pt \\
        --output_path            onyx-assistant-hf-converted
"""

import argparse
from pathlib import Path

import torch

from transformers import OnyxAssistantConfig, OnyxAssistantModel

def build_config():
    return OnyxAssistantConfig(
        hidden_size=6656,
        intermediate_size=19968,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 500_000.0},
        max_position_embeddings=131072,
        sliding_window=2048,
        layer_types=["sliding_attention"] * 5,
        attention_dropout=0,
        hidden_act="silu",
        bos_token_id=200000,
        eos_token_id=200001,
        pad_token_id=200018,
        block_size=16,
        mask_token_id=201818,
        target_layer_ids=[1, 13, 25, 37, 49],
    )


PER_LAYER_RENAMES = {
    "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.mlp.fc2_weight": "mlp.down_proj.weight",
    "feed_forward.mlp.layer_norm_weight": "post_attention_layernorm.weight",
    "attention.input_layernorm.weight": "input_layernorm.weight",
}

OTHER_RENAMES = {
    "hidden_norm.weight": "encoder.output_norm_enc.weight",
    "norm.weight": "norm.weight",
    "W_c.weight": "encoder.fc.weight",
}


def permute_rope(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    dim0 = tensor.shape[0]

    half_head = dim0 // n_heads // 2
    head_shape = (half_head, 2)
    if tensor.ndim == 2:
        tensor = tensor.view(n_heads, *head_shape, tensor.shape[1])
        tensor = tensor.transpose(1, 2).reshape(dim0, tensor.shape[-1])
    elif tensor.ndim == 1:
        tensor = tensor.view(n_heads, *head_shape)
        tensor = tensor.transpose(1, 2).reshape(dim0)
    return tensor


def convert_state_dict(source: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    n_layers = config.num_hidden_layers
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    ffn_dim = config.intermediate_size

    out: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        # split qkv proj
        qkv = source.pop(f"layers.{i}.attention.wqkv.weight")
        pieces = qkv.split([q_dim, kv_dim, kv_dim], dim=0)
        q_proj = pieces[0]
        k_proj = pieces[1]
        out[f"layers.{i}.self_attn.v_proj.weight"] = pieces[2]

        # permute qk proj and norms for block-split format rope
        out[f"layers.{i}.self_attn.q_proj.weight"] = permute_rope(q_proj, n_heads=config.num_attention_heads)
        out[f"layers.{i}.self_attn.k_proj.weight"] = permute_rope(k_proj, n_heads=config.num_key_value_heads)
        out[f"layers.{i}.self_attn.q_norm.weight"] = permute_rope(
            source.pop(f"layers.{i}.attention.q_norm.weight"), n_heads=1
        )
        out[f"layers.{i}.self_attn.k_norm.weight"] = permute_rope(
            source.pop(f"layers.{i}.attention.k_norm.weight"), n_heads=1
        )

        # split MLP gate-up proj
        fc1 = source.pop(f"layers.{i}.feed_forward.mlp.fc1_weight")
        gate_w, up_w = fc1.split([ffn_dim, ffn_dim], dim=0)
        out[f"layers.{i}.mlp.gate_proj.weight"] = gate_w
        out[f"layers.{i}.mlp.up_proj.weight"] = up_w

        for suffix, target_suffix in PER_LAYER_RENAMES.items():
            out[f"layers.{i}.{target_suffix}"] = source.pop(f"layers.{i}.{suffix}")

    for src_name, target_name in OTHER_RENAMES.items():
        out[target_name] = source.pop(src_name)

    if source:
        raise RuntimeError(f"Unconsumed LLM keys after conversion: {source}")

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to the drafter model, e.g. pytorch_weights/dflash/head.pt",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Directory to write the HF checkpoint into.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("bfloat16", "float16", "float32"),
        help="Dtype for the emitted model weights.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    config = build_config()

    print(f"Loading model shard from {args.checkpoint_path}...")
    loaded_state = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    state_dict: dict[str, torch.Tensor] = convert_state_dict(loaded_state, config)

    print(f"Materialising {OnyxAssistantModel.__name__} on meta device...")
    with torch.device("meta"):
        model = OnyxAssistantModel(config)
    model = model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    if missing:
        raise RuntimeError(f"Missing keys after load: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys after load: {unexpected}")
    model = model.to(dtype)

    print(f"Saving model to {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)

    print("Done.")


if __name__ == "__main__":
    main()
