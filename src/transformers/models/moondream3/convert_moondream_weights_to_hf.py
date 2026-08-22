#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import json
import re
from pathlib import Path
from typing import Dict

from safetensors.torch import load_file
from safetensors.torch import save_file


# Key mapping from original Moondream to HF Moondream3
OLD_KEY_TO_NEW_KEY_MAPPING = [
    # Text model
    (r"model\.text\.wte", "model.text_model.embed_tokens.weight"),
    (r"model\.text\.post_ln\.(weight|bias)", r"model.text_model.norm.\1"),
    (r"model\.text\.lm_head\.(weight|bias)", r"lm_head.\1"),
    (
        r"model\.text\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)",
        r"model.text_model.layers.\1.self_attn.qkv.\2",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.attn\.proj\.(weight|bias)",
        r"model.text_model.layers.\1.self_attn.o_proj.\2",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.attn\.tau\.wq",
        r"model.text_model.layers.\1.self_attn.tau_wq.weight",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.attn\.tau\.wv",
        r"model.text_model.layers.\1.self_attn.tau_wv.weight",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.attn\.tau\.alpha",
        r"model.text_model.layers.\1.self_attn.tau_alpha",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.ln\.(weight|bias)",
        r"model.text_model.layers.\1.input_layernorm.\2",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)",
        r"model.text_model.layers.\1.mlp.up_proj.\2",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)",
        r"model.text_model.layers.\1.mlp.down_proj.\2",
    ),
    (
        r"model\.text\.blocks\.(\d+)\.mlp\.router\.(weight|bias)",
        r"model.text_model.layers.\1.mlp.gate.\2",
    ),
    # Vision model
    (
        r"model\.vision\.patch_emb\.(weight|bias)",
        r"model.vision_model.embeddings.projection.\1",
    ),
    (r"model\.vision\.pos_emb", "model.vision_model.embeddings.position_embeddings"),
    (r"model\.vision\.post_ln\.(weight|bias)", r"model.vision_model.post_layernorm.\1"),
    (
        r"model\.vision\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)",
        r"model.vision_model.layers.\1.self_attn.qkv.\2",
    ),
    (
        r"model\.vision\.blocks\.(\d+)\.attn\.proj\.(weight|bias)",
        r"model.vision_model.layers.\1.self_attn.o_proj.\2",
    ),
    (
        r"model\.vision\.blocks\.(\d+)\.ln1\.(weight|bias)",
        r"model.vision_model.layers.\1.input_layernorm.\2",
    ),
    (
        r"model\.vision\.blocks\.(\d+)\.ln2\.(weight|bias)",
        r"model.vision_model.layers.\1.post_attention_layernorm.\2",
    ),
    (
        r"model\.vision\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)",
        r"model.vision_model.layers.\1.mlp.up_proj.\2",
    ),
    (
        r"model\.vision\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)",
        r"model.vision_model.layers.\1.mlp.down_proj.\2",
    ),
    # Vision projection
    (
        r"model\.vision\.proj_mlp\.fc1\.(weight|bias)",
        r"model.vision_model.vision_projection.up_proj.\1",
    ),
    (
        r"model\.vision\.proj_mlp\.fc2\.(weight|bias)",
        r"model.vision_model.vision_projection.down_proj.\1",
    ),
    # Region model
    (
        r"model\.region\.coord_encoder\.(weight|bias)",
        r"model.region_encoder.coord_encoder.\1",
    ),
    (
        r"model\.region\.coord_decoder\.(weight|bias)",
        r"model.region_decoder.coord_decoder.\1",
    ),
    (
        r"model\.region\.size_encoder\.(weight|bias)",
        r"model.region_encoder.size_encoder.\1",
    ),
    (
        r"model\.region\.size_decoder\.(weight|bias)",
        r"model.region_decoder.size_decoder.\1",
    ),
    (r"model\.region\.coord_features", "model.region_encoder.coord_freq"),
    (r"model\.region\.size_features", "model.region_encoder.size_freq"),
]


def rename_key(old_key: str) -> str:
    """Convert original key name to HF key name."""
    for pattern, new_key in OLD_KEY_TO_NEW_KEY_MAPPING:
        if re.match(pattern, old_key):
            return re.sub(pattern, new_key, old_key)
    return old_key


def convert_state_dict(original_state_dict: Dict) -> Dict:
    """Convert original state dict to HF format."""
    converted_state_dict = {}
    converted_keys = []
    for old_key, tensor in original_state_dict.items():
        new_key = rename_key(old_key)

        # Handle QKV weight splitting for attention
        if "attn.qkv.weight" in old_key or "attn.qkv.bias" in old_key:
            # Split QKV into separate Q, K, V matrices
            layer_match = re.search(r"blocks\.(\d+)", old_key)
            if layer_match:
                layer_idx = int(layer_match.group(1))

                # Determine if this is text or vision model
                if "model.text.blocks" in old_key:
                    n_heads = 32
                    n_kv_heads = 32
                    head_dim = 64  # 2048 / 32
                    base_key = f"model.text_model.layers.{layer_idx}.self_attn"
                else:  # vision
                    n_heads = 16
                    n_kv_heads = 16
                    head_dim = 72  # 1152 / 16
                    base_key = f"model.vision_model.layers.{layer_idx}.self_attn"

                # Split tensor
                q_dim = n_heads * head_dim
                kv_dim = n_kv_heads * head_dim

                if "weight" in old_key:
                    q_weight = tensor[:q_dim]
                    k_weight = tensor[q_dim : q_dim + kv_dim]
                    v_weight = tensor[q_dim + kv_dim :]

                    converted_state_dict[f"{base_key}.q_proj.weight"] = q_weight
                    converted_state_dict[f"{base_key}.k_proj.weight"] = k_weight
                    converted_state_dict[f"{base_key}.v_proj.weight"] = v_weight
                    converted_keys.append(old_key)
                else:  # bias
                    q_bias = tensor[:q_dim]
                    k_bias = tensor[q_dim : q_dim + kv_dim]
                    v_bias = tensor[q_dim + kv_dim :]

                    converted_state_dict[f"{base_key}.q_proj.bias"] = q_bias
                    converted_state_dict[f"{base_key}.k_proj.bias"] = k_bias
                    converted_state_dict[f"{base_key}.v_proj.bias"] = v_bias
                    converted_keys.append(old_key)
        # Handle MoE expert weight splitting
        elif (
            "mlp.fc1.weight" in old_key or "mlp.fc2.weight" in old_key
        ) and not "proj_mlp" in old_key:
            layer_match = re.search(r"blocks\.(\d+)", old_key)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                # Only process MoE layers (4+ in this model)
                if layer_idx >= 4 and "model.text." in old_key:
                    n_experts = 64  # From config

                    if "fc1.weight" in old_key:
                        # Shape: (n_experts, 2 * d_ffn, d_model) → split into individual experts
                        for expert_idx in range(n_experts):
                            expert_weight = tensor[
                                expert_idx
                            ]  # Shape: (2 * d_ffn, d_model)
                            # For GeGLU, split into gate and up projections
                            up_weight = expert_weight[
                                : expert_weight.shape[0] // 2
                            ]  # First half
                            gate_weight = expert_weight[
                                expert_weight.shape[0] // 2 :
                            ]  # Second half

                            converted_state_dict[
                                f"model.text_model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                            ] = gate_weight
                            converted_state_dict[
                                f"model.text_model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                            ] = up_weight
                    elif "fc2.weight" in old_key:
                        # Shape: (n_experts, d_model, d_ffn) → split into individual experts
                        for expert_idx in range(n_experts):
                            expert_weight = tensor[
                                expert_idx
                            ]  # Shape: (d_model, d_ffn)
                            converted_state_dict[
                                f"model.text_model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                            ] = expert_weight
                else:
                    # Dense MLP for layers < 4
                    converted_state_dict[new_key] = tensor
        else:
            converted_state_dict[new_key] = tensor
    return converted_state_dict


def convert_moondream_weights_to_hf(
    original_model_path: str,
    output_file: str,
):
    """Convert Moondream weights to HuggingFace format."""

    # Load original state dict
    print(f"Loading original model from {original_model_path}")

    # Find safetensors files
    model_path = Path(original_model_path)
    if model_path.is_file() and model_path.suffix == ".safetensors":
        # Single file
        original_state_dict = load_file(str(model_path))
    elif model_path.is_dir():
        # Directory - look for index file or single model file
        index_path = model_path / "model.safetensors.index.json"
        single_file_path = model_path / "model.safetensors"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)

            original_state_dict = {}
            for filename in set(index["weight_map"].values()):
                file_path = model_path / filename
                if file_path.exists():
                    state_dict = load_file(str(file_path))
                    for k, v in state_dict.items():
                        original_state_dict[k] = v
                else:
                    print(f"Warning: {file_path} not found")
        elif single_file_path.exists():
            original_state_dict = load_file(str(single_file_path))
        else:
            raise FileNotFoundError(
                f"Could not find model files in {original_model_path}"
            )
    else:
        raise FileNotFoundError(f"Could not find model files in {original_model_path}")

    print(f"Loaded {len(original_state_dict)} tensors")

    # Convert state dict
    print("Converting state dict...")
    converted_state_dict = convert_state_dict(original_state_dict)

    print(f"Converted {len(converted_state_dict)} tensors")

    # Save converted weights
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving converted weights to {output_path}")
    save_file(converted_state_dict, str(output_path))

    print(f"Converted weights saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Moondream weights to HuggingFace format"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to original Moondream model directory or safetensors file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save converted HuggingFace safetensors file",
    )

    args = parser.parse_args()

    convert_moondream_weights_to_hf(
        args.input_path,
        args.output_file,
    )


if __name__ == "__main__":
    main()
