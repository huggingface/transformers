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
import re

import torch
from safetensors.torch import load_file

from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    MistralConfig,
    PixtralImageProcessorFast,
    PixtralProcessor,
    PixtralVisionConfig,
)
from transformers.integrations.mistral import convert_tekken_tokenizer


# fmt: off
STATE_DICT_MAPPING = {
    # Text model keys
    r"^output.weight":                            r"language_model.lm_head.weight",
    r"^norm.weight":                              r"language_model.model.norm.weight",
    r"^tok_embeddings.weight":                    r"language_model.model.embed_tokens.weight",
    r"^layers.(\d+).attention_norm.weight":       r"language_model.model.layers.\1.input_layernorm.weight",
    r"^layers.(\d+).ffn_norm.weight":             r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"^layers.(\d+).attention.w(q|k|v|o).weight": r"language_model.model.layers.\1.self_attn.\2_proj.weight",
    r"^layers.(\d+).feed_forward.w1.weight":      r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"^layers.(\d+).feed_forward.w2.weight":      r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"^layers.(\d+).feed_forward.w3.weight":      r"language_model.model.layers.\1.mlp.up_proj.weight",

    # Vision model keys
    r"vision_encoder.transformer.layers.(\d+).attention_norm.weight": r"vision_tower.transformer.layers.\1.attention_norm.weight",
    r"^vision_encoder.transformer.layers.(\d+).ffn_norm.weight": r"vision_tower.transformer.layers.\1.ffn_norm.weight",
    r"^vision_encoder.transformer.layers.(\d+).attention.w(q|k|v|o).weight": r"vision_tower.transformer.layers.\1.attention.\2_proj.weight",
    r"^vision_encoder.transformer.layers.(\d+).feed_forward.w1.weight": r"vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight",
    r"^vision_encoder.transformer.layers.(\d+).feed_forward.w2.weight": r"vision_tower.transformer.layers.\1.feed_forward.down_proj.weight",
    r"^vision_encoder.transformer.layers.(\d+).feed_forward.w3.weight": r"vision_tower.transformer.layers.\1.feed_forward.up_proj.weight",
    r"^vision_language_adapter.w_in": r"multi_modal_projector.linear_1",
    r"^vision_language_adapter.w_out": r"multi_modal_projector.linear_2",
    r"^vision_encoder.ln_pre.weight": r"vision_tower.ln_pre.weight",
    r"^vision_encoder.patch_conv.weight": r"vision_tower.patch_conv.weight",
    r"^patch_merger.merging_layer.weight": r"multi_modal_projector.patch_merger.merging_layer.weight",
    r"^pre_mm_projector_norm.weight": r"multi_modal_projector.norm.weight",
}
# fmt: on


def map_old_key_to_new(old_key):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def permute_for_rope(tensor, n_heads, dim1, dim2):
    """Permute the weights for the ROPE formulation."""
    tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(dim1, dim2)
    return tensor


def convert_state_dict(original_state_dict: dict, config: MistralConfig):
    """Convert a state dict file, when a single `nn.Module` is never sharded in different files (usual case)."""
    new_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        if "vision" in old_key:
            num_attention_heads = config.vision_config.num_attention_heads
            num_key_value_heads = num_attention_heads
            hidden_size = config.vision_config.hidden_size
            head_dim = config.vision_config.head_dim
            key_value_dim = head_dim * num_attention_heads
            query_dim = head_dim * num_attention_heads
        else:
            num_attention_heads = config.text_config.num_attention_heads
            hidden_size = config.text_config.hidden_size
            head_dim = config.text_config.head_dim
            num_key_value_heads = config.text_config.num_key_value_heads
            key_value_dim = head_dim * num_key_value_heads
            query_dim = head_dim * num_attention_heads

        if "q_proj" in new_key:
            tensor = permute_for_rope(tensor, num_attention_heads, query_dim, hidden_size)
        elif "k_proj" in new_key:
            tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim, hidden_size)

        new_dict[new_key] = tensor
    return new_dict


def convert_config(original_config: dict, max_position_embeddings: int = 131072):
    original_vision_config = original_config.pop("vision_encoder")
    original_text_config = original_config

    # Text config
    text_key_mapping = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "intermediate_size": "hidden_dim",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "rms_norm_eps": "norm_eps",
    }
    similar_text_keys_to_keep = [
        "head_dim",
        "vocab_size",
        "rope_theta",
    ]
    new_text_config_kwargs = {k: original_text_config[v] for k, v in text_key_mapping.items()}
    new_text_config_kwargs.update({k: v for k, v in original_text_config.items() if k in similar_text_keys_to_keep})
    # These are not always defined depending on `params.json`
    new_text_config_kwargs["sliding_window"] = original_text_config.get("sliding_window", None)
    new_text_config_kwargs["max_position_embeddings"] = original_text_config.get(
        "max_seq_len", max_position_embeddings
    )
    # This may sometimes be a string in `params.json`
    if new_text_config_kwargs["sliding_window"] is not None:
        new_text_config_kwargs["sliding_window"] = int(new_text_config_kwargs["sliding_window"])
    new_text_config = MistralConfig(**new_text_config_kwargs)

    # Vision config
    new_vision_config = original_vision_config
    adapter_bias = new_vision_config.pop("adapter_bias", False)
    _ = new_vision_config.pop("mm_projector_id", None)
    _ = new_vision_config.pop("add_pre_mm_projector_layer_norm", None)
    spatial_merge_size = new_vision_config.pop("spatial_merge_size")
    image_token_id = new_vision_config.pop("image_token_id", 10)
    _ = new_vision_config.pop("image_break_token_id", 12)
    _ = new_vision_config.pop("image_end_token_id", 13)
    _ = new_vision_config.pop("max_image_size")
    new_vision_config = PixtralVisionConfig(**new_vision_config)

    new_config = Mistral3Config(
        vision_config=new_vision_config,
        text_config=new_text_config,
        multimodal_projector_bias=adapter_bias,
        image_token_index=image_token_id,
        spatial_merge_size=spatial_merge_size,
        vision_feature_layer=-1,
    )
    return new_config


def convert_and_write_model(input_dir: str, output_dir: str, max_position_embeddings: int):
    """Convert the model and save it (this implicitly save the config as well)."""
    params = read_json(os.path.join(input_dir, "params.json"))
    config = convert_config(params, max_position_embeddings)

    full_state_dict = {}
    # The model may be split between different files, but a single nn.Module is always fully present in a single file
    shards = [file for file in os.listdir(input_dir) if file.endswith(".safetensors")]
    for shard_file in shards:
        original_state_dict = load_file(os.path.join(input_dir, shard_file))
        new_dict = convert_state_dict(original_state_dict, config)
        full_state_dict.update(new_dict)

    # Load weights into model and resave them
    with torch.device("meta"):
        model = Mistral3ForConditionalGeneration(config)
    model.load_state_dict(full_state_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)


def convert_and_write_processor(input_dir: str, output_dir: str):
    """Convert the tokenizer and save it."""
    tokenizer_file = os.path.join(input_dir, "tekken.json")
    tokenizer = convert_tekken_tokenizer(tokenizer_file)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    chat_template = '{%- if messages[0]["role"] == "system" %}{%- set system_message = messages[0]["content"] %}{%- set loop_messages = messages[1:] %}\n{%- else %}{%- set loop_messages = messages %}{%- endif %}{{- bos_token }}{%- for message in loop_messages %}{%- if (message[\'role\'] == \'user\') != (loop.index0 % 2 == 0) %}{{- raise_exception(\'After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\') }}{%- endif %}{%- if message["role"] == "user" %}{%- if loop.last and system_message is defined %}{{- "[INST]" + system_message + "\n\n" }}{%- else %}{{ "[INST]" }}{%- endif %}{%- endif %}{%- if message["content"] is not string %}{%- for chunk in message["content"] %}{%- if chunk["type"] == "text" %}{%- if "content" in chunk %}{{- chunk["content"] }}{%- elif "text" in chunk %}{{- chunk["text"] }}{%- endif %}{%- elif chunk["type"] == "image" %}{{- "[IMG]" }}{%- else %}{{- raise_exception("Unrecognized content type!") }}{%- endif %}{%- endfor %}{%- else %}{{- message["content"] }}{%- endif %}{%- if message["role"] == "user" %}{{- "[/INST]" }}{%- elif message["role"] == "assistant" %}{{- eos_token}}{%- else %}{{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}{%- endif %}{%- endfor %}'

    config = read_json(os.path.join(input_dir, "params.json"))
    patch_size = config["vision_encoder"]["patch_size"]
    spatial_merge_size = config["vision_encoder"]["spatial_merge_size"]
    max_image_size = config["vision_encoder"]["max_image_size"]
    image_processor = PixtralImageProcessorFast(patch_size=patch_size, size={"longest_edge": max_image_size})

    processor = PixtralProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token="[IMG]",
        patch_size=patch_size,
        chat_template=chat_template,
        spatial_merge_size=spatial_merge_size,
    )

    # Finally save it
    processor.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=131072,
        help="`max_position_embeddings` field in the config. This needs to be manually passed (not present anywhere otherwise).",
    )

    args = parser.parse_args()

    convert_and_write_model(args.input_dir, args.output_dir, args.max_position_embeddings)
    convert_and_write_processor(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
