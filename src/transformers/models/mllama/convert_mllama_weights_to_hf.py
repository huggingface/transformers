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
import math
import os
import warnings
from typing import List

import regex as re
import torch

from transformers import MllamaConfig, MllamaForConditionalGeneration, MllamaImageProcessor, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter
from transformers.models.mllama.configuration_mllama import MllamaTextConfig, MllamaVisionConfig


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"text_model.layers.(\d+).attention.wqkv.weight": r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"text_model.norm.weight":                                                                  r"language_model.model.norm.weight",
    r"text_model.output.weight":                                                                r"language_model.lm_head.weight",
    r"text_model.tok_embeddings":                                                               r"language_model.model.embed_tokens",
    r"text_model.learnable_embedding":                                                          r"language_model.model.learnable_embedding",
    r"text_model.rope.freqs":                                                                   None, # meaning we skip it and don't want it
    # For every cross attention layer, the layer needs to be updated
    r"text_model.cross_attention_layers.(\d+).gate_attn":                                       r"language_model.model.layers.\1.cross_attn_attn_gate",
    r"text_model.cross_attention_layers.(\d+).gate_ffwd":                                       r"language_model.model.layers.\1.cross_attn_mlp_gate",
    # special key, wqkv needs to be split afterwards
    r"text_model.cross_attention_layers.(\d+).attention.wq.weight":                             r"language_model.model.layers.\1.cross_attn.q_proj.weight",
    r"text_model.cross_attention_layers.(\d+).attention.wkv":                                   r"language_model.model.layers.\1.cross_attn.k|v_proj",
    r"text_model.cross_attention_layers.(\d+).attention.wo":                                    r"language_model.model.layers.\1.cross_attn.o_proj",
    r"text_model.cross_attention_layers.(\d+).attention.inner_attention.(q|k)_norm":            r"language_model.model.layers.\1.cross_attn.\2_norm",
    r"text_model.cross_attention_layers.(\d+).attention.wq.layer_norm_weight":                  r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).attention.wk.layer_norm_weight":                  r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.mlp.fc1.weight":                     r"language_model.model.layers.\1.mlp.up|gate_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.mlp.fc2.weight":                     r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.mlp.layer_norm_weight":              r"language_model.model.layers.\1.post_attention_layernorm.weight",
    # self attention layers
    r"text_model.layers.(\d+).attention.wqkv.weight":                                           r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight",
    r"text_model.layers.(\d+).attention.wo":                                                    r"language_model.model.layers.\1.self_attn.o_proj",
    r"text_model.layers.(\d+).attention.wqkv.layer_norm_weight":                                r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.layers.(\d+).feed_forward.mlp.layer_norm_weight":                              r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"text_model.layers.(\d+).feed_forward.mlp.fc2.":                                           r"language_model.model.layers.\1.mlp.down_proj.",
    r"text_model.layers.(\d+).feed_forward.mlp.fc1.":                                           r"language_model.model.layers.\1.mlp.up|gate_proj.",
    # Vision encoder mapping
    r"vision_model.vision_encoder.conv1._linear":                                               r"vision_model.patch_embedding",
    r'vision_model.vision_projection.':                                                         r"multi_modal_projector.",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wq":    r"vision_model.\1.layers.\2.self_attn.q_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wk":    r"vision_model.\1.layers.\2.self_attn.k_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wv":    r"vision_model.\1.layers.\2.self_attn.v_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wo":    r"vision_model.\1.layers.\2.self_attn.o_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_fc":   r"vision_model.\1.layers.\2.mlp.fc1",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_proj": r"vision_model.\1.layers.\2.mlp.fc2",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_1":       r"vision_model.\1.layers.\2.input_layernorm",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_2":       r"vision_model.\1.layers.\2.post_attention_layernorm",
    r"vision_model.vision_encoder.global_transformer.resblocks.(\d+).(gate_ffn|gate_attn)":     r"vision_model.global_transformer.layers.\1.\2",
    r'vision_model.vision_encoder.ln_(pre|post).(weight|bias)':                                 r'vision_model.vision_encoder.ln_\1.\2',
    r'vision_model.vision_encoder.gated_positional_embedding\b':                                r'vision_model.gated_positional_embedding.weight',
    r'vision_model.vision_encoder.gated_positional_embedding_gate':                             r'vision_model.gated_positional_embedding.gate',
    r"vision_model.vision_encoder.(?=\w)":                                                      r"vision_model.",
}
# fmt: on

CONFIG_KEY_MAPPING = {
    "n_heads": "num_attention_heads",
    "vocab_size": "vocab_size",
    "dim": "hidden_size",
    "norm_eps": "rms_norm_eps",
    "rope_theta": "rope_theta",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def pre_compute_positional_embedding(embedding):
    """
    Instead of iterating of the batch of images, and the ratios inside, we pre-compute the
    positional embeddings depending on the aspect ratio id. This is done to support `torch.compile`
    and efficient inference / training with different aspect ratios.
    """
    max_num_tiles, *shapes = embedding.shape
    hidden_size = shapes[-1]
    max_aspect_ratio_id = (max_num_tiles - 1) * max_num_tiles + 1
    if len(shapes) == 2:  # tile embedding does not have patches
        num_patches = 1
        precomputed_embeddings = torch.zeros(
            max_aspect_ratio_id + 1, max_num_tiles, num_patches, hidden_size, device=embedding.device, dtype=embedding.dtype
        )
    else:
        num_patches = shapes[1]
        precomputed_embeddings = torch.zeros(
            max_aspect_ratio_id + 1, max_num_tiles, num_patches, hidden_size, device=embedding.device, dtype=embedding.dtype
        )

    for height in range(1, max_num_tiles + 1):
        for width in range(1, max_num_tiles + 1):
            if height * width > max_num_tiles:
                continue
            aspect_ratio_id = (height - 1) * max_num_tiles + width
            current_embedding = embedding[:height, :width].reshape(height * width, num_patches, hidden_size)
            precomputed_embeddings[aspect_ratio_id, : height * width] = current_embedding
    return precomputed_embeddings


def is_param_different_across_shards(key):
    patterns = [
        r"vision_model.patch_embedding.weight",
        r"vision_model.(transformer|global_transformer).layers.(\d+).self_attn.(q|k|v|o)_proj.weight",
        r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc1.(weight|bias)",
        r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc2.weight",  # fc2 bias is shared across shards
        r"multi_modal_projector.(weight|bias)",
        r"language_model.model.embed_tokens.weight",
        r"language_model.lm_head.weight",
        r"language_model.model.layers.(\d+).self_attn.q\|k\|v\|_proj.weight",
        r"language_model.model.layers.(\d+).self_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).mlp.up\|gate_proj.weight",
        r"language_model.model.layers.(\d+).mlp.down_proj.weight",
        r"language_model.model.layers.(\d+).cross_attn.q_proj.weight",
        r"language_model.model.layers.(\d+).cross_attn.k\|v_proj.weight",
        r"language_model.model.layers.(\d+).cross_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).mlp.up\|gate_proj.weight",
        r"language_model.model.layers.(\d+).mlp.down_proj.weight",
        r"language_model.model.learnable_embedding.weight",
    ]
    return any(re.search(pattern, key) for pattern in patterns)


def get_concat_dim(key):
    concat_dim_1 = [
        r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc2.weight",
        r"vision_model.(transformer|global_transformer).layers.(\d+).self_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).cross_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).self_attn.o_proj.weight",
        r"language_model.model.layers.(\d+).mlp.down_proj.weight"
    ]
    if any(re.search(pattern, key) for pattern in concat_dim_1):
        return 1
    return 0


def compute_intermediate_size(hidden_dim, multiple_of=1024, ffn_dim_multiplier=1.3):
    hidden_dim = 4 * int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def write_model(
    model_path,
    input_base_path,
    num_shards,
    safe_serialization=True,
):
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(input_base_path, "params.json"), "r") as f:
        params = json.load(f)

    params = params.get("model", params)

    n_layers = params["n_layers"]  # language model self-attention layers
    n_layers_cross_attention = params["vision_num_cross_attention_layers"]  # language model cross-attention layers; 90B - 20, 11B - 8
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    patch_size = 14
    num_channels = 3
    intermediate_size = compute_intermediate_size(dim, multiple_of=params["multiple_of"])  # 28672 for 90B, 5120 for 11B

    # vision model
    n_layers_vision = 32  # constant
    n_layers_vision_global = 8  # constant
    dim_vision = 1280
    n_heads_vision = 16
    n_heads_per_shard_vision = n_heads_vision // num_shards
    dims_per_head_vision = dim_vision // n_heads_vision

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    if num_shards == 1:
        loaded = [torch.load(os.path.join(input_base_path, "consolidated.pth"), map_location="cpu", mmap=True)]
    else:
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu", mmap=True)
            for i in range(num_shards)
        ]

    # ************************************ DEBUG ********************************************

    # Filter out keys for layers > n_layers
    n_layers = 4
    n_layers_cross_attention = 1
    n_layers_vision = 4
    n_layers_vision_global = 1
    for shard in loaded:
        for key in list(shard.keys()):
            # text_model layers
            if "text_model.layers" in key and int(key.split(".")[2]) >= n_layers:
                del shard[key]
            # cross attention layers
            if "text_model.cross_attention_layers." in key and int(key.split(".")[2]) >= n_layers_cross_attention:
                del shard[key]
            # vision_model layers
            if "vision_model.vision_encoder.transformer" in key and int(key.split(".")[4]) >= n_layers_vision:
                del shard[key]
            # vision_model layers
            if "vision_model.vision_encoder.global_transformer" in key and int(key.split(".")[4]) >= n_layers_vision_global:
                del shard[key]
    
    # ****************************************************************************************

    print("1. Converting language model")
    all_keys = list(loaded[0].keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    cross_attention_frequency = math.ceil(n_layers / n_layers_cross_attention)
    cross_layer_shift = list(range(n_layers))[cross_attention_frequency - 1 :: cross_attention_frequency]
    attn_layer_shift = [k for k in range(len(cross_layer_shift) + n_layers) if k not in cross_layer_shift]

    state_dict = {}
    for key in all_keys:
        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.
        new_key = new_keys[key]
        if "cross_attention" in key and "language_model" in new_key:
            new_key = re.sub(
                r"layers.(\d+).", lambda _match: f"layers.{cross_layer_shift[int(_match.groups()[0])]}.", new_key
            )
        elif "text_model.layers" in key and "language_model" in new_key:
            new_key = re.sub(
                r"layers.(\d+).", lambda _match: f"layers.{attn_layer_shift[int(_match.groups()[0])]}.", new_key
            )

        current_parameter = [chunk.pop(key).contiguous().clone() for chunk in loaded]
        if not is_param_different_across_shards(new_key):
            current_parameter = current_parameter[0]
        
        concat_dim = get_concat_dim(new_key)

        # Post-process the current_parameter.
        if "self_attn.q|k|v|_proj" in new_key and "language_model" in new_key:
            qkv_splits = [
                torch.split(param, [dim // num_shards, key_value_dim // num_shards, key_value_dim // num_shards])
                for param in current_parameter
            ]

            # query
            queries = [q for q, _, _ in qkv_splits]
            query = torch.cat([param.view(n_heads_per_shard, dims_per_head, dim) for param in queries], dim=0)
            state_dict[new_key.replace("q|k|v|", "q")] = permute_for_rope(query, n_heads, dim, dim)

            # key
            keys = [k for _, k, _ in qkv_splits]
            key = torch.cat([param.view(num_local_key_value_heads, dims_per_head, dim) for param in keys], dim=0)
            state_dict[new_key.replace("q|k|v|", "k")] = permute_for_rope(key, num_key_value_heads, key_value_dim, dim)

            # value
            values = [v for _, _, v in qkv_splits]
            value = torch.cat([param.view(num_local_key_value_heads, dims_per_head, dim) for param in values], dim=0)
            state_dict[new_key.replace("q|k|v|", "v")] = value.reshape(num_key_value_heads * dims_per_head, dim)

        elif "cross_attn.q_proj.weight" in new_key and "language_model" in new_key:
            query = torch.cat([param.view(n_heads_per_shard, dims_per_head, dim) for param in current_parameter], dim=0)
            state_dict[new_key] = query.reshape(n_heads * dims_per_head, dim)

        elif "cross_attn.k|v_proj" in new_key and "language_model" in new_key:
            key_values = [param.chunk(2) for param in current_parameter]
            keys = [k for k, _ in key_values]
            values = [v for _, v in key_values]
            key = torch.cat([param.view(num_local_key_value_heads, dims_per_head, dim) for param in keys], dim=0)
            value = torch.cat([param.view(num_local_key_value_heads, dims_per_head, dim) for param in values], dim=0)
            state_dict[new_key.replace("k|v", "k")] = key.reshape(num_key_value_heads * dims_per_head, dim)
            state_dict[new_key.replace("k|v", "v")] = value.reshape(num_key_value_heads * dims_per_head, dim)

        elif "cross_attn" in key and "q_norm" in key or "k_norm" in key:
            state_dict[new_key] = current_parameter.view(-1, 2).t().reshape(-1)

        elif "mlp.up|gate_proj." in new_key:
            gate_and_up = [param.chunk(2) for param in current_parameter]
            gate = torch.cat([gate for gate, _ in gate_and_up], dim=concat_dim)
            up = torch.cat([up for _, up in gate_and_up], dim=concat_dim)
            state_dict[new_key.replace("up|gate", "up")] = up
            state_dict[new_key.replace("up|gate", "gate")] = gate
        
        elif "vision_model" in new_key and ("q_proj" in new_key or "k_proj" in new_key or "v_proj" in new_key):
            param = torch.cat([param.view(n_heads_per_shard_vision, dims_per_head_vision, dim_vision) for param in current_parameter], dim=0)
            state_dict[new_key] = param.reshape(n_heads_vision * dims_per_head_vision, dim_vision)

        elif new_key == "vision_model.patch_embedding.weight":
            current_parameter = torch.cat(current_parameter, dim=0)
            state_dict[new_key] = current_parameter.reshape(-1, num_channels, patch_size, patch_size)

        elif new_key.endswith("gate"):
            state_dict[new_key] = current_parameter[0].view(1)

        elif "tile_pos_embed.embedding" in new_key or "gated_positional_embedding.weight" in new_key:
            # pre-compute the embeddings
            state_dict[new_key] = pre_compute_positional_embedding(current_parameter)

        elif new_key != "":
            if isinstance(current_parameter, list):
                current_parameter = torch.cat(current_parameter, dim=concat_dim)
            state_dict[new_key] = current_parameter

    state_dict["language_model.model.embed_tokens.weight"] = torch.cat(
        [
            state_dict["language_model.model.embed_tokens.weight"],
            state_dict.pop("language_model.model.learnable_embedding.weight"),
        ],
        dim=0,
    )
    del loaded
    gc.collect()

    # Write configs
    config_parameters = {CONFIG_KEY_MAPPING[key]: params[key] for key in CONFIG_KEY_MAPPING.keys()}
    vision_config = MllamaVisionConfig(
        num_hidden_layers=n_layers_vision,
        vision_input_dim=dim_vision,  # Constant, taken directly from your notes
        return_intermediate=[3, 7, 15, 23, 30],  # Based on return_intermediate indices
        max_num_tiles=4,
        num_global_layers=n_layers_vision_global,
        vision_chunk_size=params["vision_chunk_size"],
        num_attention_heads=n_heads_vision,
    )
    text_config = MllamaTextConfig(
        **config_parameters,
        num_hidden_layers=len(cross_layer_shift) + n_layers,
        cross_attention_layers=cross_layer_shift,
        vision_input_dim=dim_vision,  # Constant, aligned with vision config
        attention_bias=False,  # Constant set to False
        tie_word_embeddings=False,  # Constant set to False
        intermediate_size=intermediate_size,
    )
    config = MllamaConfig(vision_config=vision_config, text_config=text_config)
    config.architectures = ["MllamaForConditionalGeneration"]
    config.save_pretrained(model_path)
    print("Loading the checkpoint in a Llama model.")

    with torch.device("meta"):
        model = MllamaForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del model.config._name_or_path
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    MllamaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )


class MllamaConverter(TikTokenConverter):
    def __init__(self, vocab_file, num_reserved_special_tokens=256, **kwargs):
        super().__init__(vocab_file, **kwargs)
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
        ]
        special_tokens += [
            f"<|reserved_special_token_{i + 2}|>" for i in range(num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens.append("<|image|>")
        self.additional_special_tokens = special_tokens
        tokenizer = self.converted()

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
    with open(config_path, "r") as f:
        params = json.load(f)

    tile_size = params["vision_chunk_size"]
    max_image_tiles = params["vision_max_num_chunks"]

    image_processor = MllamaImageProcessor(
        do_resize=True,
        size={"height": tile_size, "width": tile_size},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_pad=True,
        max_image_tiles=max_image_tiles,
    )

    image_processor.save_pretrained(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/home/ubuntu/projects/meta_mllama/weights-90b",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="converted-mllama-90b-debug",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=List[str],
        help="The list of special tokens that should be added to the model.",
    )
    parser.add_argument(
        "--num_shards",
        default=8,
        type=int,
        help="The number of individual shards used for the model. Does not have to be the same as the number of consolidated_xx.pth",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        num_shards=args.num_shards,
    )

    write_tokenizer(
        tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
        save_dir=args.output_dir,
    )

    write_image_processor(
        config_path=os.path.join(args.input_dir, "params.json"),
        save_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
