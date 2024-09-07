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

import gc
import json
import os
import shutil
import warnings

import torch

from transformers import MllamaConfig, MllamaForConditionalGeneration, MllamaImageProcessor, PreTrainedTokenizerFast
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
    "90B": 8,
    "11B": 1,
}


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def split_wqkv_to_query_key_value(wqkv, n_heads, n_kv_heads):
    total_n_heads = n_heads + n_kv_heads * 2
    head_dim = wqkv.shape[0] // total_n_heads
    dim1 = head_dim * n_heads
    dim2 = dim1 + head_dim * n_kv_heads
    dim3 = dim1 + head_dim * n_kv_heads * 2

    query = wqkv[:dim1]
    key = wqkv[dim1:dim2]
    value = wqkv[dim2:dim3]

    return query, key, value


def apply_load_hooks_(state_dict):
    keys = list(state_dict.keys())

    for param_name in keys:
        # text model cross attention layers
        if "text_model.cross_attention_layers." in param_name:
            # CrossAttention load_hook
            if "inner_attention.q_norm.weight" in param_name:
                q_weight = state_dict.pop(param_name)
                new_param_name = param_name.replace("inner_attention.q_norm.weight", "q_norm.weight")
                state_dict[new_param_name] = q_weight
            if "inner_attention.k_norm.weight" in param_name:
                k_weight = state_dict.pop(param_name)
                new_param_name = param_name.replace("inner_attention.k_norm.weight", "k_norm.weight")
                state_dict[new_param_name] = k_weight
            if "wkv.weight" in param_name:
                wk, wv = state_dict.pop(param_name).chunk(2)
                new_param_name = param_name.replace("wkv.weight", "wk.weight")
                state_dict[new_param_name] = wk
                new_param_name = param_name.replace("wkv.weight", "wv.weight")
                state_dict[new_param_name] = wv

            # CrossAttentionTransformerBlock load_hook
            if "gate_attn" in param_name:
                attn_gate = state_dict.pop(param_name)
                new_param_name = param_name.replace("gate_attn", "gate_attn")
                if attn_gate.dim() == 1:
                    attn_gate = attn_gate[0].view(1)
                if attn_gate.dim() == 3:
                    attn_gate = attn_gate.view(1)
                state_dict[new_param_name] = attn_gate
            if "gate_ffwd" in param_name:
                ffn_gate = state_dict.pop(param_name)
                new_param_name = param_name.replace("gate_ffwd", "gate_ffwd")
                if ffn_gate.dim() == 1:
                    ffn_gate = ffn_gate[0].view(1)
                if ffn_gate.dim() == 3:
                    ffn_gate = ffn_gate.view(1)
                state_dict[new_param_name] = ffn_gate
            if "feed_forward.mlp.layer_norm_weight" in param_name:
                new_param_name = param_name.replace("feed_forward.mlp.layer_norm_weight", "ffn_norm.weight")
                state_dict[new_param_name] = state_dict.pop(param_name)
            if "attention.wq.layer_norm_weight" in param_name:
                new_param_name = param_name.replace("attention.wq.layer_norm_weight", "attention_norm.weight")
                state_dict[new_param_name] = state_dict.pop(param_name)

            # FeedForward load_hook
            if "mlp.fc1_weight" in param_name:
                fc1_weight, fc3_weight = state_dict.pop(param_name).chunk(2)
                state_dict[param_name.replace("mlp.fc1_weight", "w1.weight")] = fc1_weight
                state_dict[param_name.replace("mlp.fc1_weight", "w3.weight")] = fc3_weight
            if "mlp.fc2_weight" in param_name:
                fc2_weight = state_dict.pop(param_name)
                state_dict[param_name.replace("mlp.fc2_weight", "w2.weight")] = fc2_weight


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

    # language parameters
    n_layers = params["n_layers"]  # language model self-attention layers
    n_layers_cross_attention = params[
        "vision_num_cross_attention_layers"
    ]  # language model cross-attention layers; 90B - 20, 11B - 8
    n_heads = params["n_heads"]  # 64 for 90b (70b llama), 32 for 11b mllama
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    patch_size = 14
    num_channels = 3

    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # vision parameters
    n_layers_vision_transformer = 32  # vision model 1st transformer layers
    n_layers_global_transformer = 8  # global transformer vision layers
    n_heads_vision = 16
    n_vision_heads_per_shard = n_heads_vision // num_shards
    vision_hidden_dim = 1280  # width of vision transformers
    vision_dims_per_head = vision_hidden_dim // n_heads_vision
    mlp_ratio = 4  # vision_hidden_dim * mlp_ratio is mlp dim

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
        loaded = torch.load(os.path.join(input_base_path, "consolidated.pth"), map_location="cpu", weights_only=True)
        apply_load_hooks_(loaded)
        loaded = [loaded]
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]

    # weights are loaded. Now, we convert them into a state_dict.

    # first, language model weights.

    # we start with self-attention layers.

    param_count = 0
    index_dict = {"weight_map": {}}
    print("Converting language model self-attention layers.")
    for layer_i in range(n_layers):
        filename = f"pytorch_language_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

        state_dict = {}
        if model_size == "11B":
            state_dict[f"model.language_model.layers.{layer_i}.input_layernorm.weight"] = (
                loaded[0].pop(f"text_model.layers.{layer_i}.attention.wqkv.layer_norm_weight").clone()
            )
            state_dict[f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight"] = (
                loaded[0].pop(f"text_model.layers.{layer_i}.feed_forward.mlp.layer_norm_weight").clone()
            )
        else:
            state_dict[f"model.language_model.layers.{layer_i}.input_layernorm.weight"] = (
                loaded[0].pop(f"text_model.layers.{layer_i}.attention_norm.weight").clone()
            )
            state_dict[f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight"] = (
                loaded[0].pop(f"text_model.layers.{layer_i}.ffn_norm.weight").clone()
            )

        if model_size == "11B":
            wqkv = loaded[0].pop(f"text_model.layers.{layer_i}.attention.wqkv.weight")
            query, key, value = split_wqkv_to_query_key_value(wqkv, n_heads, num_key_value_heads)
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight"] = query
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight"] = key
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.v_proj.weight"] = value
        else:
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"text_model.layers.{layer_i}.attention.wq.weight")
                        .view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim),
                n_heads=n_heads,
            )
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"text_model.layers.{layer_i}.attention.wk.weight")
                        .view(num_local_key_value_heads, dims_per_head, dim)
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
                    loaded[i]
                    .pop(f"text_model.layers.{layer_i}.attention.wv.weight")
                    .view(num_local_key_value_heads, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

        state_dict[f"model.language_model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
            [loaded[i].pop(f"text_model.layers.{layer_i}.attention.wo.weight") for i in range(num_shards)], dim=1
        )

        if model_size == "11B":
            state_dict[f"model.language_model.layers.{layer_i}.mlp.down_proj.weight"] = loaded[0].pop(
                f"text_model.layers.{layer_i}.feed_forward.mlp.fc2_weight"
            )
            w1w3 = loaded[0].pop(f"text_model.layers.{layer_i}.feed_forward.mlp.fc1_weight")
            w1, w3 = w1w3.chunk(2)
            state_dict[f"model.language_model.layers.{layer_i}.mlp.gate_proj.weight"] = w1
            state_dict[f"model.language_model.layers.{layer_i}.mlp.up_proj.weight"] = w3
        else:
            state_dict[f"model.language_model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.feed_forward.w1.weight") for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"model.language_model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.feed_forward.w2.weight") for i in range(num_shards)],
                dim=1,
            )
            state_dict[f"model.language_model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i].pop(f"text_model.layers.{layer_i}.feed_forward.w3.weight") for i in range(num_shards)],
                dim=0,
            )

        #! no inv_freqs in modeling code?
        state_dict[f"model.language_model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        print(f"Saving {filename} in {tmp_model_path}...")
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # save embedding layer and norm

    filename = f"pytorch_language_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    concat_dim = 0 if llama_version == 3 else 1
    state_dict = {
        "model.language_model.norm.weight": loaded[0].pop("text_model.norm.weight"),
        "model.language_model.embed_tokens.weight": torch.cat(
            [loaded[i].pop("text_model.tok_embeddings.weight") for i in range(num_shards)], dim=concat_dim
        ),
        "model.language_model.learnable_embedding.weight": torch.cat(
            [loaded[i].pop("text_model.learnable_embedding.weight") for i in range(num_shards)], dim=concat_dim
        ),
        "lm_head.weight": torch.cat([loaded[i].pop("text_model.output.weight") for i in range(num_shards)], dim=0),
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    print(f"Saving {filename} in {tmp_model_path}...")
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Then, cross-attention layers from the language model
    # TODO instead of cross-attention layers, could we bundle self + cross attention into the same layers?
    # that might be a better abstraction
    print("Converting cross-attention layers.")
    for xattn_layer_i in range(n_layers_cross_attention):
        cross_attentions_filename = (
            f"pytorch_language_model_xattn-{xattn_layer_i + 1}-of-{n_layers_cross_attention + 1}.bin"
        )

        # norms

        state_dict = {
            f"model.language_model.cross_attention_layers.{xattn_layer_i}.input_layernorm.weight": loaded[0]
            .pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention_norm.weight")
            .clone(),
            f"model.language_model.cross_attention_layers.{xattn_layer_i}.post_attention_layernorm.weight": loaded[0]
            .pop(f"text_model.cross_attention_layers.{xattn_layer_i}.ffn_norm.weight")
            .clone(),
        }

        # projections

        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.mlp.gate_proj.weight"] = torch.cat(
            [
                loaded[i].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.feed_forward.w1.weight")
                for i in range(num_shards)
            ],
            dim=0,
        )
        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.mlp.down_proj.weight"] = torch.cat(
            [
                loaded[i].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.feed_forward.w2.weight")
                for i in range(num_shards)
            ],
            dim=1,
        )
        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.mlp.up_proj.weight"] = torch.cat(
            [
                loaded[i].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.feed_forward.w3.weight")
                for i in range(num_shards)
            ],
            dim=0,
        )

        # attention weights
        if model_size == "11B":
            state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.q_proj.weight"] = (
                loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wq.weight").clone()
            )
            state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.k_proj.weight"] = (
                loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wk.weight").clone()
            )
            state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.v_proj.weight"] = (
                loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wv.weight").clone()
            )
        else:
            state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.q_proj.weight"] = (
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wq.weight")
                        .view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.k_proj.weight"] = (
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wk.weight")
                        .view(num_local_key_value_heads, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim)
            )
            state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.v_proj.weight"] = (
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wv.weight")
                        .view(num_local_key_value_heads, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim)
            )

        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.o_proj.weight"] = torch.cat(
            [
                loaded[i].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.wo.weight")
                for i in range(num_shards)
            ],
            dim=1,
        )

        # gate attn (to mimic the loading hook from the authors)
        attn_gate = loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.gate_attn")
        if attn_gate.dim() == 1:
            attn_gate = attn_gate[0].view(1)
        if attn_gate.dim() == 3:
            attn_gate = attn_gate.view(1)

        ffn_gate = loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.gate_ffwd")

        if ffn_gate.dim() == 1:
            ffn_gate = ffn_gate[0].view(1)
        if ffn_gate.dim() == 3:
            ffn_gate = ffn_gate.view(1)
        # model.language_model.cross_attention_layers.0.gate_attn
        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.gate_attn"] = attn_gate.clone()

        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.ffn_gate"] = ffn_gate.clone()

        if model_size == "11B":
            q_weight = loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.q_norm.weight")
            k_weight = loaded[0].pop(f"text_model.cross_attention_layers.{xattn_layer_i}.attention.k_norm.weight")
        else:
            # q and k normalization weights (for cross-attention stability in training) are not sharded
            q_weight = loaded[0].pop(
                f"text_model.cross_attention_layers.{xattn_layer_i}.attention.inner_attention.q_norm.weight"
            )
            k_weight = loaded[0].pop(
                f"text_model.cross_attention_layers.{xattn_layer_i}.attention.inner_attention.k_norm.weight"
            )

        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.q_norm.weight"] = (
            q_weight.contiguous()
        )
        state_dict[f"model.language_model.cross_attention_layers.{xattn_layer_i}.self_attn.k_norm.weight"] = (
            k_weight.contiguous()
        )

        # save state dict of this layer

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = cross_attentions_filename
            param_count += v.numel()
        print(f"Saving {cross_attentions_filename} in {tmp_model_path}...")
        torch.save(state_dict, os.path.join(tmp_model_path, cross_attentions_filename))

    # then, converting the vision model double transformer (2 sets of layers, same width)

    # projection parameters for vision

    projection_filename = "pytorch_vision_projection.bin"
    state_dict = {
        "model.vision_model.vision_projection.weight": torch.cat(
            [loaded[i].pop("vision_model.vision_projection.weight") for i in range(num_shards)], dim=concat_dim
        ),
        "model.vision_model.vision_projection.bias": torch.cat(
            [loaded[i].pop("vision_model.vision_projection.bias") for i in range(num_shards)], dim=concat_dim
        ),
    }
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = projection_filename
        param_count += v.numel()
    print(f"Saving {projection_filename} in {tmp_model_path}...")
    torch.save(state_dict, os.path.join(tmp_model_path, projection_filename))

    # global vision layers - identical to a CLIPVisionModel except:
    # - there are 2 gating parameters
    # - there are no class embedding/patch embedding layers
    for global_vision_layer_i in range(n_layers_global_transformer):
        state_dict = {}
        global_vision_filename = (
            f"pytorch_global_vision_transformer-{global_vision_layer_i}-of-{n_layers_global_transformer + 1}.bin"
        )

        # the extra gating params gate_attn and gate_ffn are not sharded

        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.gate_attn"
        ] = loaded[0].pop(
            f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.gate_attn"
        )
        state_dict[f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.gate_ffn"] = (
            loaded[0].pop(f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.gate_ffn")
        )

        if model_size == "11B":
            state_dict[
                f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.q_proj.weight"
            ] = (
                loaded[0]
                .pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wq.weight"
                )
                .clone()
            )
            state_dict[
                f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.k_proj.weight"
            ] = (
                loaded[0]
                .pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wk.weight"
                )
                .clone()
            )
            state_dict[
                f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.v_proj.weight"
            ] = (
                loaded[0]
                .pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wv.weight"
                )
                .clone()
            )

        else:
            # attention weights and biases are sharded
            state_dict[
                f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.q_proj.weight"
            ] = permute(
                torch.cat(
                    [
                        loaded[i]
                        .pop(
                            f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wq.weight"
                        )
                        .view(n_vision_heads_per_shard, vision_dims_per_head, vision_hidden_dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(vision_hidden_dim, vision_hidden_dim),
                n_heads=n_heads_vision,
                dim1=vision_hidden_dim,
                dim2=vision_hidden_dim,
            )

            # for vision n_kv_heads = n_heads and local_kv heads = n_vision_heads per shard!
            #  same for normal and global image transformer

            state_dict[
                f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.k_proj.weight"
            ] = permute(
                torch.cat(
                    [
                        loaded[i]
                        .pop(
                            f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wk.weight"
                        )
                        .view(n_vision_heads_per_shard, vision_dims_per_head, vision_hidden_dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(vision_hidden_dim, vision_hidden_dim),
                n_heads=n_heads_vision,
                dim1=vision_hidden_dim,
                dim2=vision_hidden_dim,
            )

            state_dict[
                f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.v_proj.weight"
            ] = torch.cat(
                [
                    loaded[i]
                    .pop(
                        f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wv.weight"
                    )
                    .view(n_vision_heads_per_shard, vision_dims_per_head, vision_hidden_dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(vision_hidden_dim, vision_hidden_dim)

        # simple concatenation for sharded o_proj
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.o_proj.weight"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wo.weight"
                )
                for i in range(num_shards)
            ],
            dim=1,
        )

        # simple concatenation for sharded biases
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.q_proj.bias"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wq.bias"
                )
                for i in range(num_shards)
            ],
            dim=concat_dim,
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.k_proj.bias"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wk.bias"
                )
                for i in range(num_shards)
            ],
            dim=concat_dim,
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.v_proj.bias"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wv.bias"
                )
                for i in range(num_shards)
            ],
            dim=concat_dim,
        )

        # o_proj bias is not sharded (RowParallelLinear does not shard bias across devices)
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.self_attn.o_proj.bias"
        ] = loaded[0].pop(
            f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.attn.wo.bias"
        )

        # mlp layers

        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.mlp.fc1.weight"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.mlp.c_fc.weight"
                )
                for i in range(num_shards)
            ],
            dim=concat_dim,
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.mlp.fc1.bias"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.mlp.c_fc.bias"
                )
                for i in range(num_shards)
            ],
            dim=concat_dim,
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.mlp.fc2.weight"
        ] = torch.cat(
            [
                loaded[i].pop(
                    f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.mlp.c_proj.weight"
                )
                for i in range(num_shards)
            ],
            dim=1,
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.mlp.fc2.bias"
        ] = loaded[0].pop(
            f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.mlp.c_proj.bias"
        )

        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.layer_norm1.weight"
        ] = (
            loaded[0]
            .pop(f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.ln_1.weight")
            .clone()
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.layer_norm1.bias"
        ] = (
            loaded[0]
            .pop(f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.ln_1.bias")
            .clone()
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.layer_norm2.weight"
        ] = (
            loaded[0]
            .pop(f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.ln_2.weight")
            .clone()
        )
        state_dict[
            f"model.vision_model.vision_encoder.global_transformer.layers.{global_vision_layer_i}.layer_norm2.bias"
        ] = (
            loaded[0]
            .pop(f"vision_model.vision_encoder.global_transformer.resblocks.{global_vision_layer_i}.ln_2.bias")
            .clone()
        )

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = global_vision_filename
            param_count += v.numel()

        print(f"Saving {global_vision_filename} in {tmp_model_path}...")
        torch.save(state_dict, os.path.join(tmp_model_path, global_vision_filename))

    # the normal transformer - 32 layers for 90B, a CLIP model except with additional tile position embeddings
    # vision encoder embedding parameters

    encoder_embeddings_params_filename = "pytorch_embedding_params.bin"

    # patch embedding conv weights are sharded and based on _unfold so
    # they require to be reshaped properly

    linear_patch_weights = torch.cat(
        [loaded[i].pop("vision_model.vision_encoder.conv1._linear.weight") for i in range(num_shards)], dim=concat_dim
    )
    out_channels = linear_patch_weights.shape[0]
    linear_patch_weights = linear_patch_weights.view(out_channels, num_channels, patch_size, patch_size)

    state_dict = {
        # embeddings are not sharded
        "model.vision_model.vision_encoder.class_embedding": loaded[0].pop(
            "vision_model.vision_encoder.class_embedding"
        ),
        "model.vision_model.vision_encoder.positional_embedding": loaded[0].pop(
            "vision_model.vision_encoder.positional_embedding"
        ),
        "model.vision_model.vision_encoder.gated_positional_embedding": loaded[0].pop(
            "vision_model.vision_encoder.gated_positional_embedding"
        ),
        "model.vision_model.vision_encoder.gated_positional_embedding_gate": loaded[0].pop(
            "vision_model.vision_encoder.gated_positional_embedding_gate"
        ),
        "model.vision_model.vision_encoder.patch_embedding.weight": linear_patch_weights,
        # layer norms are not sharded
        "model.vision_model.vision_encoder.ln_post.weight": loaded[0].pop(
            "vision_model.vision_encoder.ln_post.weight"
        ),
        "model.vision_model.vision_encoder.ln_post.bias": loaded[0].pop("vision_model.vision_encoder.ln_post.bias"),
        "model.vision_model.vision_encoder.ln_pre.weight": loaded[0].pop("vision_model.vision_encoder.ln_pre.weight"),
        "model.vision_model.vision_encoder.ln_pre.bias": loaded[0].pop("vision_model.vision_encoder.ln_pre.bias"),
        # tile pos embeddings (specific to mllama) are not sharded
        "model.vision_model.vision_encoder.pre_tile_pos_embed.embedding": loaded[0].pop(
            "vision_model.vision_encoder.pre_tile_pos_embed.embedding"
        ),
        "model.vision_model.vision_encoder.pre_tile_pos_embed.gate": loaded[0].pop(
            "vision_model.vision_encoder.pre_tile_pos_embed.gate"
        ),
        "model.vision_model.vision_encoder.post_tile_pos_embed.embedding": loaded[0].pop(
            "vision_model.vision_encoder.post_tile_pos_embed.embedding"
        ),
        "model.vision_model.vision_encoder.post_tile_pos_embed.gate": loaded[0].pop(
            "vision_model.vision_encoder.post_tile_pos_embed.gate"
        ),
    }
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = encoder_embeddings_params_filename
        param_count += v.numel()
    print(f"Saving {encoder_embeddings_params_filename} in {tmp_model_path}...")
    torch.save(state_dict, os.path.join(tmp_model_path, encoder_embeddings_params_filename))

    # vision transformer layer parameters

    for vision_layer_i in range(n_layers_vision_transformer):
        state_dict = {}
        encoder_layer_parameters_filename = (
            f"pytorch_vision_transformer-{vision_layer_i}-of-{n_layers_vision_transformer + 1}.bin"
        )

        # attention weights and biases are sharded
        if model_size == "11B":
            state_dict[
                f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.q_proj.weight"
            ] = (
                loaded[0]
                .pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wq.weight")
                .clone()
            )
            state_dict[
                f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.k_proj.weight"
            ] = (
                loaded[0]
                .pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wk.weight")
                .clone()
            )
            state_dict[
                f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.v_proj.weight"
            ] = (
                loaded[0]
                .pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wv.weight")
                .clone()
            )

        else:
            state_dict[
                f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.q_proj.weight"
            ] = permute(
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wq.weight")
                        .view(n_vision_heads_per_shard, vision_dims_per_head, vision_hidden_dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(vision_hidden_dim, vision_hidden_dim),
                n_heads=n_heads_vision,
                dim1=vision_hidden_dim,
                dim2=vision_hidden_dim,
            )

            # for vision n_kv_heads = n_heads and local_kv heads = n_vision_heads per shard!
            #  same for normal and global image transformer

            state_dict[
                f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.k_proj.weight"
            ] = permute(
                torch.cat(
                    [
                        loaded[i]
                        .pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wk.weight")
                        .view(n_vision_heads_per_shard, vision_dims_per_head, vision_hidden_dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(vision_hidden_dim, vision_hidden_dim),
                n_heads=n_heads_vision,
                dim1=vision_hidden_dim,
                dim2=vision_hidden_dim,
            )

            state_dict[
                f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.v_proj.weight"
            ] = torch.cat(
                [
                    loaded[i]
                    .pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wv.weight")
                    .view(n_vision_heads_per_shard, vision_dims_per_head, vision_hidden_dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(vision_hidden_dim, vision_hidden_dim)

        # simple concatenation for sharded o_proj
        state_dict[
            f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.o_proj.weight"
        ] = torch.cat(
            [
                loaded[i].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wo.weight")
                for i in range(num_shards)
            ],
            dim=1,
        )

        # simple concatenation for sharded biases
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.q_proj.bias"] = (
            torch.cat(
                [
                    loaded[i].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wq.bias")
                    for i in range(num_shards)
                ],
                dim=concat_dim,
            )
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.k_proj.bias"] = (
            torch.cat(
                [
                    loaded[i].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wk.bias")
                    for i in range(num_shards)
                ],
                dim=concat_dim,
            )
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.v_proj.bias"] = (
            torch.cat(
                [
                    loaded[i].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wv.bias")
                    for i in range(num_shards)
                ],
                dim=concat_dim,
            )
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.self_attn.o_proj.bias"] = (
            loaded[0].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.attn.wo.bias")
        )

        # mlp layers
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.mlp.fc1.weight"] = (
            torch.cat(
                [
                    loaded[i].pop(
                        f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.mlp.c_fc.weight"
                    )
                    for i in range(num_shards)
                ],
                dim=concat_dim,
            )
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.mlp.fc1.bias"] = torch.cat(
            [
                loaded[i].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.mlp.c_fc.bias")
                for i in range(num_shards)
            ],
            dim=concat_dim,
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.mlp.fc2.weight"] = (
            torch.cat(
                [
                    loaded[i].pop(
                        f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.mlp.c_proj.weight"
                    )
                    for i in range(num_shards)
                ],
                dim=1,
            )
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.mlp.fc2.bias"] = loaded[
            0
        ].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.mlp.c_proj.bias")

        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.layer_norm1.weight"] = (
            loaded[0].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.ln_1.weight").clone()
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.layer_norm1.bias"] = (
            loaded[0].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.ln_1.bias").clone()
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.layer_norm2.weight"] = (
            loaded[0].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.ln_2.weight").clone()
        )
        state_dict[f"model.vision_model.vision_encoder.transformer.layers.{vision_layer_i}.layer_norm2.bias"] = (
            loaded[0].pop(f"vision_model.vision_encoder.transformer.resblocks.{vision_layer_i}.ln_2.bias").clone()
        )

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = encoder_layer_parameters_filename
            param_count += v.numel()

        print(f"Saving {encoder_layer_parameters_filename} in {tmp_model_path}...")
        torch.save(state_dict, os.path.join(tmp_model_path, encoder_layer_parameters_filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    vision_config = {
        "n_heads": params["n_heads"],
        "vision_chunk_size": params["vision_chunk_size"],
        "vision_max_num_chunks": params["vision_max_num_chunks"],
        "patch_size": patch_size,
        "projection_dim": params["dim"],  # need to figure out
        "vision_input_dim": 1280,  # constant, "self.vision_input_dim = 1280" for CrossAttentionTransformerVision in original code
        "return_intermediate": "3,7,15,23,30",
        "global_vision_layers": 8,  # constant "n_global_layers=8" for VisionEncoder in original code
        "max_num_tiles": 4,  # not used in the modeling code yet, "max_num_tiles=4" for VisionEncoder in original code
        "norm_eps": params["norm_eps"],
        "ffn_dim_multiplier": params["ffn_dim_multiplier"],
        "multiple_of": params["multiple_of"],
    }
    text_config = {
        "vocab_size": params["vocab_size"],
        "n_layers": params["n_layers"],
        "dim": params["dim"],
        "n_heads": params["n_heads"],
        "n_kv_heads": params["n_kv_heads"],
        "max_seq_len": 512,  # in examples
        "ffn_dim_multiplier": params["ffn_dim_multiplier"],
        "norm_eps": params["norm_eps"],
        "rope_theta": params["rope_theta"],
        "use_scaled_rope": params["use_scaled_rope"],
        "vision_num_cross_attention_layers": params[
            "vision_num_cross_attention_layers"
        ],  # should be in vision config?
        "multiple_of": params["multiple_of"],
        "vision_input_dim": 1280,  # constant, see "vision_input_dim" in vision config
    }
    config = MllamaConfig(vision_config=vision_config, text_config=text_config)
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    mllama_model = MllamaForConditionalGeneration.from_pretrained(
        tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # Avoid saving this as part of the config.
    del mllama_model.config._name_or_path
    mllama_model.config.torch_dtype = torch.float16  # not sure about this.
    print("Saving in the Transformers format.")
    mllama_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


# TODO: update to new provided code: python + video tokens
class MllamaConverter(TikTokenConverter):
    def __init__(self, vocab_file, num_reserved_special_tokens=256, **kwargs):
        super().__init__(vocab_file, **kwargs)
        tokenizer = self.converted()
        chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        )
        num_reserved_special_tokens = 256
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
            "<|image|>",
        ]
        special_tokens += [
            f"<|reserved_special_token_{i + 2}|>" for i in range(num_reserved_special_tokens - len(special_tokens))
        ]
        tokenizer.add_special_tokens(special_tokens)

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            pad_token="<|finetune_right_pad_id|>",
            chat_template=chat_template,
            model_input_names=["input_ids", "attention_mask"],
        )


def write_tokenizer(tokenizer_path: str, save_dir: str):
    converter = MllamaConverter(
        tokenizer_path,
        pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",  # noqa: W605
    )
    tokenizer = converter.tokenizer
    tokenizer.save_pretrained(save_dir)


def write_image_processor(config_path: str, save_dir: str):
    params = read_json(config_path)

    patch_size = params["vision_chunk_size"]
    max_image_tiles = params["vision_max_num_chunks"]

    image_processor = MllamaImageProcessor(
        do_resize=True,
        size={"height": patch_size, "width": patch_size},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_pad=True,
        max_image_tiles=max_image_tiles,
    )

    image_processor.save_pretrained(save_dir)


write_model(
    model_path="/home/ubuntu/projects/new-model-addition-mllama/mllama-11b",
    input_base_path="/home/ubuntu/projects/meta_mllama/weights-11b-biases",
    safe_serialization=True,
    model_size="11B",
    llama_version=3,
    vocab_size=128256,
)

write_tokenizer(
    tokenizer_path="/home/ubuntu/projects/meta_mllama/weights-11b-biases/tokenizer.model",
    save_dir="/home/ubuntu/projects/new-model-addition-mllama/mllama-11b",
)

write_image_processor(
    config_path="/home/ubuntu/projects/meta_mllama/weights-11b-biases/params.json",
    save_dir="/home/ubuntu/projects/new-model-addition-mllama/mllama-11b",
)
