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
import argparse
import json
import os

import shutill
import torch
from safetensors.torch import load_file, save_file

from transformers import MiMoV2FlashConfig, MiMoV2FlashForCausalLM


def convert_mimo_checkpoint(input_dir, output_dir):
    print(f"Loading weight from {input_dir}...")

    # Load config
    config_path = os.path.join(input_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"config.json not found in {input_dir}")

    with open(config_path, "r") as f:
        original_config = json.load(f)

    # Initialize HF Config
    # This ensures we have a valid configuration object based on the file
    config = MiMoV2FlashConfig(**original_config)
    print(
        f"Config loaded. Hidden size: {config.hidden_size}, Layers: {config.num_hidden_layers}"
    )

    # Load state dict (Handling Sharded Safetensors)
    index_file = os.path.join(input_dir, "model.safetensors.index.json")
    state_dict = {}

    if os.path.exists(index_file):
        print(f"Found sharded index file: {index_file}")
        with open(index_file, "r") as f:
            index_data = json.load(f)

        # Get list of unique files containing weights
        shard_files = set(index_data["weight_map"].values())

        for shard in shard_files:
            shard_path = os.path.join(input_dir, shard)
            print(f"Loading shard: {shard}...")
            if not os.path.exists(shard_path):
                print(
                    f"WARNING: Shard {shard} missing. Skipping (model might be incomplete)."
                )
                continue

            shard_weights = load_files(shard_path)
            state_dict.update(shard_weights)
    else:
        # Fallback for single file models
        single_file = os.path.join(input_dir, "model.safetensors")
        bin_file = os.path.join(input_dir, "pytorch_model.bin")

        if os.path.exists(single_file):
            print("Found single model.safetensors file.")
            state_dict = load_file(single_file)
        elif os.path.exists(bin_file):
            print("Found single pytorch_model.bin file.")
            state_dict = torch.load(bin_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {input_dir}")

    print(f"Total keys loaded: {len(state_dict)}")

    # Rename Keys
    # We map the keys from the remote code structure to our official structure.
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Ensure standard prefix
        # Sometimes keys start with "layers." instead of "model.layers."
        if not new_key.startswith("model.") and not new_key.startswith("lm_head."):
            if (
                key.startswith("layers.")
                or key.startswith("norm.")
                or key.startswith("embed_tokens.")
            ):
                new_key = f"model.{key}"
            elif key.startswith("output."):
                new_key = key.replace("output.", "lm_head.")

        # Standardize Attention and MLP names
        # Common pattern: attention.wq -> self_attn.q_proj
        new_key = new_key.replace(".attention.wq.", ".self_attn.q_proj.")
        new_key = new_key.replace(".attention.wk.", ".self_attn.k_proj.")
        new_key = new_key.replace(".attention.wv.", ".self_attn.v_proj.")
        new_key = new_key.replace(".attention.wo.", ".self_attn.o_proj.")

        # Feed Forward
        new_key = new_key.replace(".feed_forward.w1.", ".mlp.gate_proj.")
        new_key = new_key.replace(".feed_forward.w2.", ".mlp.down_proj.")
        new_key = new_key.replace(".feed_forward.w3.", ".mlp.up_proj.")

        # Layer Norms
        new_key = new_key.replace("attention_norm", "input_layernorm")
        new_key = new_key.replace("ffn_norm", "post_attention_layernorm")

        # MoE Specific Mappings
        # Based on typical MoE implementations
        new_key = new_key.replace("block_sparse_moe.gate", "mlp.router.gate")
        new_key = new_key.replace("block_sparse_moe.experts", "mlp.experts")

        # Handle the linear naming if they use w1/w2/w3 inside experts
        if "experts" in new_key:
            new_key = new_key.replace(".w1.", ".gate_proj.")
            new_key = new_key.replace(".w2.", ".down_proj.")
            new_key = new_key.replace(".w3.", ".up_proj.")

        new_state_dict[new_key] = value

    print("Keys converted.")

    # Save to Output
    print(f"Saving HF model to {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # We instantiate the model class to verify the structure matches
        # strict=False allows us to save even if some keys didn't map perfectly (user can debug)
        model = MiMoV2FlashForCausalLM(config)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print(f"  Missing keys: {len(missing)}")
        if len(missing) > 0:
            print(f"  Example missing: {missing[:3]}")

        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) > 0:
            print(f"  Example unexpected: {unexpected[:3]}")

        # Save weights
        model.save_pretrained(output_dir)

        # Save tokenizer files if present in input
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]
    for fname in tokenizer_files:
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            shutil.copy(fpath, output_dir)

    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the original/remote model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the converted HF model",
    )
    args = parser.parse_args()
    convert_mimo_checkpoint(args.input_dir, args.output_dir)
