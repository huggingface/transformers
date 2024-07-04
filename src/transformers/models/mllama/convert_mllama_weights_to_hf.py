# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from transformers import LlamaConfig, LlamaForCausalLM, CLIPVisionConfig, LlamaTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None


NUM_SHARDS = {
    "7B": 1,
    "8B": 1,
    "8Bf": 1,
    "7Bf": 1,
    "13B": 2,
    "13Bf": 2,
    "34B": 4,
    "30B": 4,
    "65B": 8,
    "70B": 8,
    "70Bf": 8,
}


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(
    model_path,
    input_base_path,
    model_size,
    safe_serialization=True,
    llama_version=1,
    vocab_size=None,
):
    # for backward compatibility, before you needed the repo to be called `my_repo/model_size`
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0 and llama_version != 3:
        max_position_embeddings = 16384
    else:
        # Depending on the Llama version, the default max_position_embeddings has different values.
        if llama_version == 1:
            max_position_embeddings = 2048
        elif llama_version == 2:
            max_position_embeddings = 4096
        elif llama_version == 3:
            max_position_embeddings = 8192

    vocab_size = vocab_size if vocab_size is not None else 32000
    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if num_shards == 1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_language_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if num_shards == 1:
            # Unsharded
            state_dict = {
                f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded.pop(f"text_model.layers.{layer_i}.attention.wq.weight"), n_heads=n_heads
                ),
                f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded.pop(f"text_model.layers.{layer_i}.attention.wk.weight"),
                    n_heads=num_key_value_heads,
                    dim1=dim // num_local_key_value_heads,
                ),
                f"model.language_model.layers.{layer_i}.self_attn.v_proj.weight": loaded.pop(f"text_model.layers.{layer_i}.attention.wv.weight"),
                f"model.language_model.layers.{layer_i}.self_attn.o_proj.weight": loaded.pop(f"text_model.layers.{layer_i}.attention.wo.weight"),
                f"model.language_model.layers.{layer_i}.mlp.gate_proj.weight": loaded.pop(f"text_model.layers.{layer_i}.feed_forward.w1.weight"),
                f"model.language_model.layers.{layer_i}.mlp.down_proj.weight": loaded.pop(f"text_model.layers.{layer_i}.feed_forward.w2.weight"),
                f"model.language_model.layers.{layer_i}.mlp.up_proj.weight": loaded.pop(f"text_model.layers.{layer_i}.feed_forward.w3.weight"),
                f"model.language_model.layers.{layer_i}.input_layernorm.weight": loaded.pop(f"text_model.layers.{layer_i}.attention_norm.weight"),
                f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight": loaded.pop(f"text_model.layers.{layer_i}.ffn_norm.weight"),
            }
        else:
            # Sharded
            # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
            # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
            # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

            state_dict = {
                f"model.language_model.layers.{layer_i}.input_layernorm.weight": loaded[0].pop(
                    f"text_model.layers.{layer_i}.attention_norm.weight"
                ).clone(),
                f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0].pop(
                    f"text_model.layers.{layer_i}.ffn_norm.weight"
                ).clone(),
            }
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i].pop(f"text_model.layers.{layer_i}.attention.wq.weight").view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim),
                n_heads=n_heads,
            )
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i].pop(f"text_model.layers.{layer_i}.attention.wk.weight").view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim),
                num_key_value_heads,
                key_value_dim,
                dim,
            )
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i].pop(f"text_model.layers.{layer_i}.attention.wv.weight").view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            state_dict[f"model.language_model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.attention.wo.weight") for i in range(num_shards)], dim=1
            )
            state_dict[f"model.language_model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.feed_forward.w1.weight") for i in range(num_shards)], dim=0
            )
            state_dict[f"model.language_model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.feed_forward.w2.weight") for i in range(num_shards)], dim=1
            )
            state_dict[f"model.language_model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.feed_forward.w3.weight") for i in range(num_shards)], dim=0
            )

        state_dict[f"model.language_model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_language_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards == 1:
        # Unsharded
        state_dict = {
            "model.language_model.embed_tokens.weight": loaded.pop("text_model.tok_embeddings.weight"),
            "model.language_model.norm.weight": loaded.pop("text_model.norm.weight"),
            "lm_head.weight": loaded.pop("text_model.output.weight"),
        }
    else:
        concat_dim = 0 if llama_version == 3 else 1
        state_dict = {
            "model.language_model.norm.weight": loaded[0].pop("text_model.norm.weight"),
            "model.language_model.embed_tokens.weight": torch.cat(
                [loaded[i].pop("text_model.tok_embeddings.weight") for i in range(num_shards)], dim=concat_dim
            ),
            "lm_head.weight": torch.cat([loaded[i].pop("text_model.output.weight") for i in range(num_shards)], dim=0),
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
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=128000 if llama_version == 3 else 1,
        eos_token_id=128001 if llama_version == 3 else 2,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    language_model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del language_model.config._name_or_path
    language_model.config.torch_dtype = torch.float16  # not sure about this.
    print("Saving in the Transformers format.")
    language_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


write_model(
    model_path="/home/pablo/mllama_hf/test",
    input_base_path="/home/pablo/weights/Meta-Llama-3.1-87B-Vision-Dummy-20240624190000",
    safe_serialization=True,
    model_size="70B",
    llama_version=3,
    vocab_size=128256,
)