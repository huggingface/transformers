# Copyright 2023 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

import torch

from transformers import (
    MixtralConfig,
    MixtralForCausalLM,
)


"""
Sample usage:

```
python src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py \
    --input_dir /path/to/downloaded/mixtral/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import MixtralForCausalLM

model = MixtralForCausalLM.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, safe_serialization=True):
    os.makedirs(model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = 1

    # For some reason this is a string in the params.json
    sliding_window = int(params["sliding_window"])
    n_layers = params["num_hidden_layers"]
    n_heads = params["num_attention_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["hidden_size"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    max_position_embeddings = 4096 * 8
    num_local_experts = params["num_local_experts"]
    ffn_dim = params["intermediate_size"]

    vocab_size = params["vocab_size"]

    if "num_key_value_heads" in params:
        num_key_value_heads = params["num_key_value_heads"]  # for GQA / MQA
        num_local_key_value_heads = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_local_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    loaded = [
        torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pt"), map_location="cpu") for i in range(8)
    ]

    merged_state_dict = {}
    for state_dict in loaded:
        merged_state_dict.update(state_dict)

    state_dict = {}

    for layer_i in range(n_layers):
        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

        state_dict.update(
            {
                f"model.layers.{layer_i}.input_layernorm.weight": merged_state_dict[
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": merged_state_dict[
                    f"layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
        )

        state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
            merged_state_dict[f"layers.{layer_i}.attention.wq.weight"]
            .view(n_heads_per_shard, dims_per_head, dim)
            .reshape(dim, dim)
        )
        state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
            merged_state_dict[f"layers.{layer_i}.attention.wk.weight"]
            .view(num_local_key_value_heads, dims_per_head, dim)
            .reshape(key_value_dim, dim),
            num_key_value_heads,
            key_value_dim,
            dim,
        )
        state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = (
            merged_state_dict[f"layers.{layer_i}.attention.wv.weight"]
            .view(num_local_key_value_heads, dims_per_head, dim)
            .reshape(key_value_dim, dim)
        )

        state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = merged_state_dict[
            f"layers.{layer_i}.attention.wo.weight"
        ]

        w1 = merged_state_dict[f"layers.{layer_i}.block_sparse_moe.w1"]
        w2 = merged_state_dict[f"layers.{layer_i}.block_sparse_moe.w2"]
        w3 = merged_state_dict[f"layers.{layer_i}.block_sparse_moe.w3"]

        experts_w1 = [
            w1[ffn_dim * expert_idx : ffn_dim * (expert_idx + 1), :].contiguous().clone()
            for expert_idx in range(num_local_experts)
        ]

        for idx, expert_block in enumerate(experts_w1):
            expert_key = f"model.layers.{layer_i}.block_sparse_moe.experts.{idx}.w1"
            state_dict[expert_key + ".weight"] = expert_block.clone()

        experts_w2 = [
            w2[ffn_dim * expert_idx : ffn_dim * (expert_idx + 1), :].contiguous().clone()
            for expert_idx in range(num_local_experts)
        ]

        for idx, expert_block in enumerate(experts_w2):
            expert_key = f"model.layers.{layer_i}.block_sparse_moe.experts.{idx}.w2"
            state_dict[expert_key + ".weight"] = expert_block.T.clone().contiguous()

        experts_w3 = [
            w3[ffn_dim * expert_idx : ffn_dim * (expert_idx + 1), :].contiguous().clone()
            for expert_idx in range(num_local_experts)
        ]

        for idx, expert_block in enumerate(experts_w3):
            expert_key = f"model.layers.{layer_i}.block_sparse_moe.experts.{idx}.w3"
            state_dict[expert_key + ".weight"] = expert_block.clone()

        state_dict[f"model.layers.{layer_i}.block_sparse_moe.gate.weight"] = merged_state_dict[
            f"layers.{layer_i}.block_sparse_moe.gate.weight"
        ]

    state_dict.update(
        {
            "model.norm.weight": merged_state_dict["norm.weight"],
            "model.embed_tokens.weight": merged_state_dict["tok_embeddings.weight"],
            "lm_head.weight": merged_state_dict["output.weight"],
        }
    )

    config = MixtralConfig(
        hidden_size=dim,
        intermediate_size=ffn_dim,
        num_attention_heads=params["num_attention_heads"],
        num_hidden_layers=params["num_hidden_layers"],
        rms_norm_eps=params["rms_norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        sliding_window=sliding_window,
        num_local_experts=num_local_experts,
    )

    print("Loading the checkpoint in a Mixtral model.")
    with torch.device("meta"):
        model = MixtralForCausalLM(config)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    print("Saving in the Transformers format.")

    model.load_state_dict(state_dict, strict=True, assign=True)

    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"{n} has not been loaded!"

    model.save_pretrained(model_path, safe_serialization=safe_serialization)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Mixtral weights, which contains tokenizer.model and model folders",
        required=True,
    )
    parser.add_argument(
        "--model_size",
        choices=["7B"],
        help="'f' models correspond to the finetuned versions, and are specific to the Mixtral official release. For more details on Mixtral, checkout the original repo: https://huggingface.co/mistral-ai",
        default="7B",
    )
    parser.add_argument("--output_dir", help="Location to write HF model", required=True)
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
