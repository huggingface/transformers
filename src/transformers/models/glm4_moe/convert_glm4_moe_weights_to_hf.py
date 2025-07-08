#!/usr/bin/env python
"""
Internal utility to convert a proprietary GLM-4-MoE checkpoint into Hugging Face format.

Steps
1. Load every *.safetensors file in input_dir and collect all tensors.
2. Split tensors into ≤5 GiB shards named model-00001-of-000NN.safetensors, etc.
3. Write model.safetensors.index.json with tensor-to-shard mapping.
4. Write config.json for Glm4MoeForCausalLM.
5. Print progress throughout.

Usage
    python convert_to_hf.py <input_dir> <output_dir>
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


MAX_SHARD_BYTES = 5 * 1024**3

CONFIG = {
    "architectures": ["Glm4MoeForCausalLM"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "pad_token_id": 151329,
    "eos_token_id": [151329, 151336, 151338],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "decoder_sparse_step": 1,
    "initializer_range": 0.02,
    "intermediate_size": 10944,
    "max_position_embeddings": 131072,
    "model_type": "glm4_moe",
    "moe_intermediate_size": 1408,
    "norm_topk_prob": True,
    "num_attention_heads": 96,
    "n_group": 1,
    "topk_group": 1,
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "routed_scaling_factor": 1.0,
    "topk_method": "noaux_tc",
    "moe_router_dtype": "float32",
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 1,
    "num_hidden_layers": 46,
    "num_key_value_heads": 8,
    "partial_rotary_factor": 0.5,
    "output_router_logits": False,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 10000,
    "router_aux_loss_coef": 0.001,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.54.0dev",
    "use_cache": True,
    "vocab_size": 151552,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Location of the local folder copied from the Hub.")
    parser.add_argument("output_dir", type=str, help="Location to write HF model and tokenizer")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tensors, total_bytes = [], 0
    for file in sorted(input_path.glob("*.safetensors")):
        print(f"Loading {file.name}")
        data = load_file(file)
        tensors.extend(data.items())
        total_bytes += sum(t.numel() * t.element_size() for t in data.values())

    print(f"Total tensors: {len(tensors)}   Total size: {total_bytes / 1024**3:.2f} GiB")

    shards, current, used = [], {}, 0
    for name, tensor in tensors:
        size = tensor.numel() * tensor.element_size()
        if used and used + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, used = {}, 0
        current[name] = tensor
        used += size
    shards.append(current)

    total_shards = len(shards)
    weight_map = {}
    for idx, shard in enumerate(shards, 1):
        fname = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
        print(f"Saving {fname}")
        save_file(shard, output_path / fname)
        weight_map.update(dict.fromkeys(shard, fname))

    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)
    print("Wrote model.safetensors.index.json")

    with open(output_path / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print("Wrote config.json")

    print("Conversion complete.")


if __name__ == "__main__":
    main()
