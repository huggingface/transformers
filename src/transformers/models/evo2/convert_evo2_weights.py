# Copyright 2024 The HuggingFace Inc. team.
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
import os

import torch
from huggingface_hub import hf_hub_download

from transformers import Evo2Config, Evo2ForCausalLM


def convert_original_weights_to_transformers(original_weights):
    """Convert weights from original Evo2 format to transformers format."""
    
    # Create config based on the original model architecture (Evo2-1b-base)
    # vocab_size=512, hidden_size=1920, 25 layers (21 hyena + 4 attention every 7th layer starting from 3)
    layer_types = []
    for i in range(25):
        if i % 7 == 3:
            layer_types.append("attention")
        else:
            layer_types.append("hyena")

    config = Evo2Config(
        vocab_size=512,
        hidden_size=1920,
        intermediate_size=5120,
        num_hidden_layers=25,
        num_attention_heads=15,  # 1920 / 128
        num_key_value_heads=15,
        layer_types=layer_types,
        hyena_filters=128,  # Number of filter groups
        hyena_order=3,  # 5760 / 1920 = 3
        hyena_kernel_size=3,  # Short filter kernel size
        tie_word_embeddings=True,
    )

    # Initialize new state dict
    new_state_dict = {}

    # Convert embeddings
    new_state_dict["model.embed_tokens.weight"] = original_weights["embedding_layer.weight"]
    new_state_dict["lm_head.weight"] = original_weights["unembed.weight"]

    # Convert each layer
    for layer_idx in range(25):
        layer_type = layer_types[layer_idx]
        orig_prefix = f"blocks.{layer_idx}"
        new_prefix = f"model.layers.{layer_idx}.block"

        # Common components: norms and MLP
        new_state_dict[f"model.layers.{layer_idx}.block.input_layernorm.weight"] = original_weights[
            f"{orig_prefix}.pre_norm.scale"
        ]
        new_state_dict[f"model.layers.{layer_idx}.block.post_attention_layernorm.weight"] = original_weights[
            f"{orig_prefix}.post_norm.scale"
        ]

        # MLP layers
        # Original: l1 (gate), l2 (up), l3 (down)
        new_state_dict[f"{new_prefix}.mlp.gate_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l1.weight"]
        new_state_dict[f"{new_prefix}.mlp.up_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l2.weight"]
        new_state_dict[f"{new_prefix}.mlp.down_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l3.weight"]

        if layer_type == "attention":
            # Convert attention layer
            # Original uses Wqkv (combined), we need separate q_proj, k_proj, v_proj
            wqkv = original_weights[f"{orig_prefix}.inner_mha_cls.Wqkv.weight"]
            hidden_size = config.hidden_size
            
            # Split Wqkv into q, k, v
            q, k, v = torch.split(wqkv, hidden_size, dim=0)
            new_state_dict[f"model.layers.{layer_idx}.block.attention.q_proj.weight"] = q
            new_state_dict[f"model.layers.{layer_idx}.block.attention.k_proj.weight"] = k
            new_state_dict[f"model.layers.{layer_idx}.block.attention.v_proj.weight"] = v

            # Output projection
            new_state_dict[f"model.layers.{layer_idx}.block.attention.o_proj.weight"] = original_weights[
                f"{orig_prefix}.inner_mha_cls.out_proj.weight"
            ]

            # Load rotary embedding inv_freq from original weights
            if f"{orig_prefix}.inner_mha_cls.rotary_emb.inv_freq" in original_weights:
                new_state_dict[f"model.layers.{layer_idx}.block.attention.rotary_emb.inv_freq"] = original_weights[
                    f"{orig_prefix}.inner_mha_cls.rotary_emb.inv_freq"
                ]

        else:
            # Convert hyena filter layer
            new_state_dict[f"model.layers.{layer_idx}.block.filter.projections.weight"] = original_weights[
                f"{orig_prefix}.projections.weight"
            ]
            new_state_dict[f"model.layers.{layer_idx}.block.filter.short_filter_weight"] = original_weights[
                f"{orig_prefix}.filter.short_filter_weight"
            ]
            new_state_dict[f"model.layers.{layer_idx}.block.filter.out_filter_dense.weight"] = original_weights[
                f"{orig_prefix}.out_filter_dense.weight"
            ]
            new_state_dict[f"model.layers.{layer_idx}.block.filter.out_filter_dense.bias"] = original_weights[
                f"{orig_prefix}.out_filter_dense.bias"
            ]

            # Long filter parameters (FIR or IIR)
            # These are not standard nn.Parameters in our implementation but we can load them into the state dict
            # and then manually assign them in the model if needed, or just save them as part of the state dict
            # since we registered them as buffers/parameters in the model (or should have).
            # In our implementation, they are initialized as None. We need to make sure they are loaded.
            
            if f"{orig_prefix}.filter.h" in original_weights:
                new_state_dict[f"model.layers.{layer_idx}.block.filter.h"] = original_weights[
                    f"{orig_prefix}.filter.h"
                ]
            if f"{orig_prefix}.filter.D" in original_weights:
                new_state_dict[f"model.layers.{layer_idx}.block.filter.D"] = original_weights[
                    f"{orig_prefix}.filter.D"
                ]
            if f"{orig_prefix}.filter.log_poles" in original_weights:
                new_state_dict[f"model.layers.{layer_idx}.block.filter.log_poles"] = original_weights[
                    f"{orig_prefix}.filter.log_poles"
                ]
            if f"{orig_prefix}.filter.residues" in original_weights:
                new_state_dict[f"model.layers.{layer_idx}.block.filter.residues"] = original_weights[
                    f"{orig_prefix}.filter.residues"
                ]

    # Final norm
    new_state_dict["model.norm.weight"] = original_weights["norm.scale"]

    return new_state_dict, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="arcinstitute/evo2_1b_base",
        help="Hub model id",
    )
    parser.add_argument(
        "--output_dir",
        default="evo2_converted",
        help="The output directory to save the converted model",
    )
    args = parser.parse_args()

    print(f"Downloading weights for {args.model_id}...")
    weights_path = hf_hub_download(args.model_id, "evo2_1b_base.pt")
    original_weights = torch.load(weights_path, map_location="cpu", weights_only=False)

    print("Converting weights...")
    new_state_dict, config = convert_original_weights_to_transformers(original_weights)

    print("Loading into Evo2ForCausalLM...")
    model = Evo2ForCausalLM(config)
    
    # Load state dict (strict=False because Hyena layers have optional parameters that might be missing if unused)
    # But we want to make sure we load everything we have.
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Missing keys: {len(missing_keys)}")
    if len(missing_keys) > 0:
        print(missing_keys[:10])
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if len(unexpected_keys) > 0:
        print(unexpected_keys[:10])

    # Manually assign filter parameters (h, D, log_poles, residues) if they were not loaded by load_state_dict
    # because they were None in the model init.
    # Actually, since we put them in new_state_dict, load_state_dict might complain if the model attributes are None.
    # We might need to initialize them in the model first or just assign them directly.
    
    for layer_idx in range(config.num_hidden_layers):
        if config.layer_types[layer_idx] == "hyena":
            filter_module = model.model.layers[layer_idx].block.filter
            orig_prefix = f"blocks.{layer_idx}.filter"
            
            if f"{orig_prefix}.h" in original_weights:
                filter_module.h = nn.Parameter(original_weights[f"{orig_prefix}.h"])
            if f"{orig_prefix}.D" in original_weights:
                filter_module.D = nn.Parameter(original_weights[f"{orig_prefix}.D"])
            if f"{orig_prefix}.log_poles" in original_weights:
                filter_module.log_poles = nn.Parameter(original_weights[f"{orig_prefix}.log_poles"])
            if f"{orig_prefix}.residues" in original_weights:
                filter_module.residues = nn.Parameter(original_weights[f"{orig_prefix}.residues"])

    print(f"Saving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
