# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

"""
This script is intended to convert weights provided in the ZhipuAI TGI format into Transformers-compatible weights.
It has currently only been tested successfully on the image/video question answering model GLM-4V-9B,
which is based on the GLM-4-9B-0414 model.
Run the script with:
```
python convert_glm4v_weights_to_hf.py --model_dir /model/glm4v/tgi/ --save_dir glm-4v-9b-0414
```
"""

import torch
import os
import json
import argparse
import re
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import Glm4vConfig


def convert_weight(model_path, save_path):
    def write_json(text, path):
        with open(path, "w") as f:
            json.dump(text, f)

    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path, "r"))

    dic = {}
    for filename in os.listdir(model_path):
        if filename.endswith(".bin") or filename.endswith(".pt") or filename.endswith("safetensors"):
            print(f"Loading file: {filename}")
            if filename.endswith("safetensors"):
                with safe_open(os.path.join(model_path, filename), framework="pt") as f:
                    state_dict = {k: f.get_tensor(k) for k in f.keys()}
            else:
                state_dict = torch.load(os.path.join(model_path, filename), map_location="cpu")

            for key in state_dict.keys():
                dic[key] = state_dict[key]

    print(f"Loaded {len(dic)} weights")

    # Track which weights have been processed
    processed_weights = set()

    param_count = 0
    index_dict = {"weight_map": {}}

    os.makedirs(save_path, exist_ok=True)

    for key in dic:
        if dic[key].dtype != torch.bfloat16:
            dic[key] = dic[key].to(torch.bfloat16)

    new_dic = {}
    base_model_mappings = {
        "model.word_embeddings.weight": "model.embed_tokens.weight",
        "model.final_layernorm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for src_key, tgt_key in base_model_mappings.items():
        if src_key in dic:
            new_dic[tgt_key] = dic[src_key].clone()
            processed_weights.add(src_key)
            print(f"Mapping: {src_key} -> {tgt_key} (BF16), shape: {dic[src_key].shape}")

    num_layers = config.get("num_layers", 40)
    for i in range(num_layers):
        norm_keys = [
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
        ]

        for key in norm_keys:
            if key in dic:
                new_dic[key] = dic[key].clone()
                processed_weights.add(key)
                print(f"Keeping: {key} (BF16), shape: {dic[key].shape}")

        attn_mapping = {
            f"model.layers.{i}.self_attention.query.weight": f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attention.key.weight": f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attention.value.weight": f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attention.query.bias": f"model.layers.{i}.self_attn.q_proj.bias",
            f"model.layers.{i}.self_attention.key.bias": f"model.layers.{i}.self_attn.k_proj.bias",
            f"model.layers.{i}.self_attention.value.bias": f"model.layers.{i}.self_attn.v_proj.bias",
            f"model.layers.{i}.self_attention.dense.weight": f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.attention_sandwich_layernorm.weight": f"model.layers.{i}.post_self_attn_layernorm.weight",
            f"model.layers.{i}.mlp_sandwich_layernorm.weight": f"model.layers.{i}.post_mlp_layernorm.weight",
        }

        for src_key, tgt_key in attn_mapping.items():
            if src_key in dic:
                new_dic[tgt_key] = dic[src_key].clone()
                processed_weights.add(src_key)
                print(f"Mapping: {src_key} -> {tgt_key} (BF16), shape: {dic[src_key].shape}")

        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        down_key = f"model.layers.{i}.mlp.down_proj.weight"

        if gate_key in dic and up_key in dic:
            combined_weight = torch.cat([dic[gate_key], dic[up_key]], dim=0)
            new_dic[f"model.layers.{i}.mlp.gate_up_proj.weight"] = combined_weight
            processed_weights.add(gate_key)
            processed_weights.add(up_key)
            print(f"Merging MLP: {gate_key} + {up_key} -> gate_up_proj.weight (BF16), shape: {combined_weight.shape}")

        if down_key in dic:
            new_dic[down_key] = dic[down_key].clone()
            processed_weights.add(down_key)
            print(f"Keeping: {down_key} (BF16), shape: {dic[down_key].shape}")

    visual_keys = [k for k in dic.keys() if k.startswith("visual.")]
    vision_keys = [k for k in dic.keys() if k.startswith("vision_model.")]

    for key in visual_keys:
        if not "blocks." in key and not "transformer.layers." in key:
            new_dic[key] = dic[key].clone()
            processed_weights.add(key)
            print(f"Keeping vision model weight: {key} (BF16), shape: {dic[key].shape}")

    special_visual_mappings = {
        "vision_model.post_layernorm.weight": "visual.post_layernorm.weight",
        "vision_model.post_conv_layernorm.weight": "visual.post_conv_layernorm.weight",
        "vision_model.conv.weight": "visual.downsample.weight",
        "vision_model.conv.bias": "visual.downsample.bias",
        "vision_model.position_embeddings.weight": "visual.embeddings.position_embedding.weight",
        "vision_model.conv3d.weight": "visual.patch_embed.proj.weight",
        "vision_model.conv3d.bias": "visual.patch_embed.proj.bias",
        "vision_model.linear_proj.norm1.weight": "visual.merger.post_projection_norm.weight",
        "vision_model.linear_proj.norm1.bias": "visual.merger.post_projection_norm.bias",
        "vision_model.linear_proj.linear_proj.weight": "visual.merger.proj.weight",
        "vision_model.linear_proj.gate_proj.weight": "visual.merger.gate_proj.weight",
        "vision_model.linear_proj.dense_h_to_4h.weight": "visual.merger.up_proj.weight",
        "vision_model.linear_proj.dense_4h_to_h.weight": "visual.merger.down_proj.weight",
    }

    for src_key, tgt_key in special_visual_mappings.items():
        if src_key in dic:
            new_dic[tgt_key] = dic[src_key].clone()
            processed_weights.add(src_key)
            print(f"Special mapping: {src_key} -> {tgt_key} (BF16), shape: {dic[src_key].shape}")

    vision_layer_pattern = re.compile(r"vision_model\.transformer\.layers\.(\d+)\..*")
    qkv_weights = {}
    layer_indices = set()
    for key in vision_keys:
        match = vision_layer_pattern.match(key)
        if match:
            layer_indices.add(match.group(1))

    for layer_idx in layer_indices:
        for key in vision_keys:
            attn_match = re.match(
                f"vision_model\\.transformer\\.layers\\.{layer_idx}\\.attention\\.(query|key|value)\\.weight", key
            )
            if attn_match:
                qkv_type = attn_match.group(1)
                if layer_idx not in qkv_weights:
                    qkv_weights[layer_idx] = {}
                qkv_weights[layer_idx][qkv_type] = dic[key].clone()
                processed_weights.add(key)

        vision_mapping = {
            f"vision_model.transformer.layers.{layer_idx}.input_layernorm.weight": f"visual.blocks.{layer_idx}.norm1.weight",
            f"vision_model.transformer.layers.{layer_idx}.post_attention_layernorm.weight": f"visual.blocks.{layer_idx}.norm2.weight",
            f"vision_model.transformer.layers.{layer_idx}.attention.dense.weight": f"visual.blocks.{layer_idx}.attn.proj.weight",
            f"vision_model.transformer.layers.{layer_idx}.mlp.gate_proj.weight": f"visual.blocks.{layer_idx}.mlp.gate_proj.weight",
            f"vision_model.transformer.layers.{layer_idx}.mlp.dense_h_to_4h.weight": f"visual.blocks.{layer_idx}.mlp.up_proj.weight",
            f"vision_model.transformer.layers.{layer_idx}.mlp.fc2.weight": f"visual.blocks.{layer_idx}.mlp.down_proj.weight",
        }

        for src_key, tgt_key in vision_mapping.items():
            if src_key in dic:
                new_dic[tgt_key] = dic[src_key].clone()
                processed_weights.add(src_key)
                print(f"Mapping: {src_key} -> {tgt_key} (BF16), shape: {dic[src_key].shape}")

    for key in vision_keys:
        if not vision_layer_pattern.match(key) and not any(key == src for src, _ in special_visual_mappings.items()):
            if re.match(r"vision_model\.transformer\.layers\.\d+\.attention\.(query|key|value)\.weight", key):
                continue

            new_key = key.replace("vision_model.", "visual.")
            new_dic[new_key] = dic[key].clone()
            processed_weights.add(key)
            print(f"Mapping: {key} -> {new_key} (BF16), shape: {dic[key].shape}")

    for layer_idx, weights in qkv_weights.items():
        if "query" in weights and "key" in weights and "value" in weights:
            combined = torch.cat([weights["query"], weights["key"], weights["value"]], dim=0)
            new_key = f"visual.blocks.{layer_idx}.attn.qkv.weight"
            new_dic[new_key] = combined
            print(f"Combining QKV weights: layer {layer_idx} -> {new_key} (BF16), shape: {combined.shape}")
        else:
            print(f"Warning: QKV weights for layer {layer_idx} are incomplete, cannot combine")

    unmapped_weights = set(dic.keys()) - processed_weights
    if unmapped_weights:
        print(f"Warning: {len(unmapped_weights)} weights were not mapped:")
        for i, key in enumerate(sorted(unmapped_weights)):
            if i < 10:  # Only show first 10 to avoid flooding console
                print(f"  - {key}")
            elif i == 10:
                print(f"  - ... and {len(unmapped_weights) - 10} more")
    else:
        print("All weights were successfully mapped")

    # Use safetensors to save in chunks
    CHUNK_SIZE = 5 * 1024 * 1024 * 1024  # 5GB chunks
    total_size_bytes = sum(tensor.numel() * 2 for tensor in new_dic.values())  # BF16 is 2 bytes per parameter
    estimated_chunks = max(1, (total_size_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE)

    print(f"Estimated total size: {total_size_bytes / (1024**3):.2f} GB, expected chunks: {estimated_chunks}")

    # Define a function to sort keys by the requested order
    def sort_key_by_layer(key):
        if key.startswith("model.layers."):
            layer_match = re.match(r"model\.layers\.(\d+)\.", key)
            if layer_match:
                layer_num = int(layer_match.group(1))
                return (0, layer_num, key)

        elif key.startswith("visual.blocks."):
            block_match = re.match(r"visual\.blocks\.(\d+)\.", key)
            if block_match:
                block_num = int(block_match.group(1))
                return (1, block_num, key)

        else:
            return (2, 0, key)

    # Group weights into chunks (5GB per chunk)
    sorted_items = sorted(new_dic.items(), key=lambda x: sort_key_by_layer(x[0]))
    chunks = {}
    current_chunk = {}
    current_size = 0
    chunk_idx = 1

    for key, tensor in sorted_items:
        tensor_size = tensor.numel() * 2  # BF16 is 2 bytes

        # If tensor is larger than chunk size, save individually
        if tensor_size > CHUNK_SIZE:
            chunk_name = f"{chunk_idx:05d}"
            chunks[chunk_name] = {key: tensor}
            index_dict["weight_map"][key] = f"model-{chunk_name}-of-{estimated_chunks:05d}.safetensors"
            param_count += tensor_size
            chunk_idx += 1
            print(f"Large tensor {key} will be saved separately")
            continue

        # If adding this tensor would exceed chunk size, start a new chunk
        if current_size + tensor_size > CHUNK_SIZE and current_chunk:
            chunk_name = f"{chunk_idx:05d}"
            chunks[chunk_name] = current_chunk
            chunk_idx += 1
            current_chunk = {}
            current_size = 0

        # Add tensor to current chunk
        current_chunk[key] = tensor
        current_size += tensor_size
        param_count += tensor_size

    # Save the last chunk if not empty
    if current_chunk:
        chunk_name = f"{chunk_idx:05d}"
        chunks[chunk_name] = current_chunk

    # Adjust chunk count if actual count differs from estimate
    actual_chunks = len(chunks)
    if actual_chunks != estimated_chunks:
        print(f"Actual chunk count ({actual_chunks}) differs from estimate ({estimated_chunks})")
        estimated_chunks = actual_chunks

    # Save chunks to files
    for chunk_name, weights in chunks.items():
        filename = f"model-{chunk_name}-of-{estimated_chunks:05d}.safetensors"
        filepath = os.path.join(save_path, filename)
        save_file(weights, filepath)

        # Update weight map
        for key in weights:
            index_dict["weight_map"][key] = filename

        print(f"Saved chunk {chunk_name} to {filename}, containing {len(weights)} weights")

    # Save index file
    index_dict["metadata"] = {"total_size": param_count}
    write_json(index_dict, os.path.join(save_path, "model.safetensors.index.json"))
    print(f"Saved index file to model.safetensors.index.json")

    # Copy and update config
    if os.path.exists(config_path):
        config = Glm4vConfig()
        config_data = config.to_dict()
        if "torch_dtype" not in config_data:
            config_data["torch_dtype"] = "bfloat16"

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

    print(f"Conversion complete, model saved to {save_path} in chunked mode (5GB/chunk) (BF16 format)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to TGI model directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save converted transformers model")

    args = parser.parse_args()
    convert_weight(args.model_dir, args.save_dir)
