# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
import gc
import json
import os
import shutil
import warnings

import torch

from transformers import ChameleonConfig, ChameleonForCausalLM


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/chameleon/convert_chameleon_weights_to_hf.py \
    --input_dir /path/to/downloaded/chameleon/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import ChameleonForCausalLM, LlamaTokenizer

model = ChameleonForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

NUM_SHARDS = {
    "7B": 1,
    "30B": 4,
}

VOCAB_SIZE = 65536


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, safe_serialization=True, chameleon_version=1):
    # for backward compatibility, before you needed the repo to be called `my_repo/model_size`
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    if os.path.isfile(os.path.join(input_base_path, "consolidate_params.json")):
        params = {**params, **read_json(os.path.join(input_base_path, "consolidate_params.json"))}
    num_shards = NUM_SHARDS[model_size]
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    qk_layernorm = params["qk_normalization"]
    swin_norm = params["swin_norm"]
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        # Depending on the Chameleon version, the default max_position_embeddings has different values.
        if chameleon_version == 1:
            max_position_embeddings = 4096
        else:
            raise NotImplementedError(
                f"Version {chameleon_version} of chameleon is not supported yet. "
                "Current supported versions of chameleon are [1]."
            )

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if num_shards == 1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = None
        for possible_name in ["consolidated.pth", "consolidated.00.pth"]:
            possible_path = os.path.join(input_base_path, possible_name)
            if os.path.exists(possible_path):
                loaded = torch.load(possible_path, map_location="cpu")
                break
        assert loaded is not None
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if num_shards == 1:
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": loaded[f"layers.{layer_i}.attention.wq.weight"],
                f"model.layers.{layer_i}.self_attn.k_proj.weight": loaded[f"layers.{layer_i}.attention.wk.weight"],
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
            if qk_layernorm:
                state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"] = loaded[
                    f"layers.{layer_i}.attention.q_normalization.weight"
                ]
                state_dict[f"model.layers.{layer_i}.self_attn.q_norm.bias"] = loaded[
                    f"layers.{layer_i}.attention.q_normalization.bias"
                ]
                state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"] = loaded[
                    f"layers.{layer_i}.attention.k_normalization.weight"
                ]
                state_dict[f"model.layers.{layer_i}.self_attn.k_norm.bias"] = loaded[
                    f"layers.{layer_i}.attention.k_normalization.bias"
                ]

        else:
            # Sharded
            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": torch.stack(
                    [l[f"layers.{layer_i}.attention_norm.weight"] for l in loaded]
                ).mean(dim=0),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": torch.stack(
                    [l[f"layers.{layer_i}.ffn_norm.weight"] for l in loaded]
                ).mean(dim=0),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            if qk_layernorm:
                state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"] = torch.stack(
                    [l[f"layers.{layer_i}.attention.q_normalization.weight"] for l in loaded]
                ).mean(dim=0)
                state_dict[f"model.layers.{layer_i}.self_attn.q_norm.bias"] = torch.stack(
                    [l[f"layers.{layer_i}.attention.q_normalization.bias"] for l in loaded]
                ).mean(dim=0)
                state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"] = torch.stack(
                    [l[f"layers.{layer_i}.attention.k_normalization.weight"] for l in loaded]
                ).mean(dim=0)
                state_dict[f"model.layers.{layer_i}.self_attn.k_norm.bias"] = torch.stack(
                    [l[f"layers.{layer_i}.attention.k_normalization.bias"] for l in loaded]
                ).mean(dim=0)
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards == 1:
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "model.norm.weight": torch.stack([loaded[i]["norm.weight"] for i in range(num_shards)]).mean(dim=0),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    config = ChameleonConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=VOCAB_SIZE,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        qk_layernorm=qk_layernorm,
        swin_norm=swin_norm,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Chameleon model.")
    model = ChameleonForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Chameleon weights",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "30B"],
        help="'f' models correspond to the finetuned versions, and are specific to the Chameleon official release. For more details on Chameleon, checkout the original repo: https://huggingface.co/meta-chameleon",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # Different Chameleon versions used different default values for max_position_embeddings, hence the need to be able to specify which version is being used.
    parser.add_argument(
        "--chameleon_version",
        choices=[1],
        default=1,
        type=int,
        help="Version of the Chameleon model to convert",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        safe_serialization=args.safe_serialization,
        chameleon_version=args.chameleon_version,
    )


if __name__ == "__main__":
    main()
