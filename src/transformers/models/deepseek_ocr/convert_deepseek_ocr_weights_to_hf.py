# Copyright 2025 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
import re
from pathlib import Path
from textwrap import dedent

import torch
from safetensors.torch import load_file

from transformers import DeepseekOcrConfig, DeepseekOcrForConditionalGeneration


CHAT_TEMPLATE = dedent(
    """
    {%- for message in messages %}
        {%- set role = message['role'] | lower %}
        {%- if message['content'] is string %}
{{ message['content'] }}
        {%- else %}
            {%- for content in message['content'] %}
                {%- if content['type'] == 'text' %}
{{ content['text'] }}
                {%- elif content['type'] == 'image' %}
<image>
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {%- if not loop.last %}

        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
<|Assistant|>
    {%- endif %}
    """
).strip()


# fmt: off
STATE_DICT_MAPPING = {
    r"^model\.sam_model\.patch_embed\.proj\.(weight|bias)":                                      r"model.sam_model.patch_embed.projection.\1",
    r"^model\.sam_model\.blocks\.(\d+)\.norm(\d+)\.(weight|bias)":                               r"model.sam_model.layers.\1.layer_norm\2.\3",
    r"^model\.sam_model\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)":                               r"model.sam_model.layers.\1.attn.qkv.\2",
    r"^model\.sam_model\.blocks\.(\d+)\.attn\.proj\.(weight|bias)":                              r"model.sam_model.layers.\1.attn.proj.\2",
    r"^model\.sam_model\.blocks\.(\d+)\.attn\.rel_pos_([hw])":                                   r"model.sam_model.layers.\1.attn.rel_pos_\2",
    r"^model\.sam_model\.blocks\.(\d+)\.mlp\.lin(\d+)\.(weight|bias)":                           r"model.sam_model.layers.\1.mlp.lin\2.\3",
    r"^model\.sam_model\.neck\.0\.weight":                                                        r"model.sam_model.neck.conv1.weight",
    r"^model\.sam_model\.neck\.1\.(weight|bias)":                                                 r"model.sam_model.neck.layer_norm1.\1",
    r"^model\.sam_model\.neck\.2\.weight":                                                        r"model.sam_model.neck.conv2.weight",
    r"^model\.sam_model\.neck\.3\.(weight|bias)":                                                 r"model.sam_model.neck.layer_norm2.\1",
    r"^model\.sam_model\.net_2\.weight":                                                          r"model.sam_model.net_2.weight",
    r"^model\.sam_model\.net_3\.weight":                                                          r"model.sam_model.net_3.weight",
    r"^model\.sam_model\.pos_embed":                                                              r"model.sam_model.pos_embed",

    r"^model\.vision_model\.embeddings\.class_embedding":                                         r"model.clip_model.vision_model.embeddings.class_embedding",
    r"^model\.vision_model\.embeddings\.patch_embedding\.weight":                                 r"model.clip_model.vision_model.embeddings.patch_embedding.weight",
    r"^model\.vision_model\.embeddings\.position_embedding\.weight":                              r"model.clip_model.vision_model.embeddings.position_embedding.weight",
    r"^model\.vision_model\.pre_layrnorm\.(weight|bias)":                                         r"model.clip_model.vision_model.pre_layrnorm.\1",
    r"^model\.vision_model\.transformer\.layers\.(\d+)\.layer_norm(\d+)\.(weight|bias)":          r"model.clip_model.vision_model.encoder.layers.\1.layer_norm\2.\3",
    r"^model\.vision_model\.transformer\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)":      r"model.clip_model.vision_model.encoder.layers.\1.self_attn.qkv_proj.\2",
    r"^model\.vision_model\.transformer\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)":      r"model.clip_model.vision_model.encoder.layers.\1.self_attn.out_proj.\2",
    r"^model\.vision_model\.transformer\.layers\.(\d+)\.mlp\.fc(\d+)\.(weight|bias)":             r"model.clip_model.vision_model.encoder.layers.\1.mlp.fc\2.\3",
    r"^model\.vision_model\.post_layernorm\.(weight|bias)":                                       r"model.clip_model.vision_model.post_layernorm.\1",

    r"^model\.projector\.layers\.(weight|bias)":                                                  r"model.projector.layers.\1",

    r"^model\.embed_tokens\.weight":                                                              r"model.language_model.embed_tokens.weight",
    r"^model\.layers\.(\d+)\.input_layernorm\.weight":                                            r"model.language_model.layers.\1.input_layernorm.weight",
    r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight":                                   r"model.language_model.layers.\1.post_attention_layernorm.weight",
    r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight":                                  r"model.language_model.layers.\1.self_attn.\2_proj.weight",
    r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight":                                   r"model.language_model.layers.\1.mlp.\2_proj.weight",
    r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)\.(weight|bias)":                                 r"model.language_model.layers.\1.mlp.\2.\3",
    r"^model\.norm\.weight":                                                                      r"model.language_model.norm.weight",
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|bias)":            r"model.language_model.layers.\1.mlp.experts.\2.\3_proj.\4",
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(\d+)\.(gate|up|down)_proj\.(weight|bias)":     r"model.language_model.layers.\1.mlp.shared_experts.\2.\3_proj.\4",
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)\.(weight|bias)":                 r"model.language_model.layers.\1.mlp.experts.\2.\3.\4",
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(\d+)\.(gate|up|down)\.(weight|bias)":          r"model.language_model.layers.\1.mlp.shared_experts.\2.\3.\4",
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|up|down)_proj\.(weight|bias)":            r"model.language_model.layers.\1.mlp.shared_experts.\2_proj.\3",
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|up|down)\.(weight|bias)":                 r"model.language_model.layers.\1.mlp.shared_experts.\2.\3",

    r"^model\.image_newline":                                                                     r"model.image_newline",
    r"^model\.view_seperator":                                                                    r"model.view_seperator",

    r"^lm_head\.weight":                                                                          r"lm_head.weight",
}
# fmt: on


def map_old_key_to_new(old_key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def split_qkv_weights(key, tensor, num_heads, hidden_size):
    if "qkv_proj.weight" in key:
        q, k, v = torch.split(tensor, hidden_size, dim=0)
        return {
            key.replace("qkv_proj.weight", "q_proj.weight"): q,
            key.replace("qkv_proj.weight", "k_proj.weight"): k,
            key.replace("qkv_proj.weight", "v_proj.weight"): v,
        }
    elif "qkv_proj.bias" in key:
        q, k, v = torch.split(tensor, hidden_size, dim=0)
        return {
            key.replace("qkv_proj.bias", "q_proj.bias"): q,
            key.replace("qkv_proj.bias", "k_proj.bias"): k,
            key.replace("qkv_proj.bias", "v_proj.bias"): v,
        }

    return {key: tensor}


def convert_state_dict(original_state_dict, config):
    new_state_dict = {}

    clip_hidden_size = config.vision_config.clip_config.hidden_size
    clip_num_heads = config.vision_config.clip_config.num_attention_heads

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        if "qkv_proj" in new_key and "clip_model" in new_key:
            split_dict = split_qkv_weights(new_key, tensor, clip_num_heads, clip_hidden_size)
            new_state_dict.update(split_dict)
        else:
            new_state_dict[new_key] = tensor

    return new_state_dict


TOKENIZER_RELATED_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "chat_template.json",
]


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSeek OCR weights to HuggingFace format")
    parser.add_argument(
        "--original_checkpoint_path",
        type=str,
        required=True,
        help="Path to the original checkpoint file (.safetensors)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Repository ID for pushing to the Hub (required if --push_to_hub is set)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.original_checkpoint_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_path.parent / "config.json"

    print(f"Loading original checkpoint from {checkpoint_path}")
    original_state_dict = load_file(checkpoint_path)

    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DeepseekOcrConfig(**config_dict)
    else:
        print("Config not found, using default config")
        config = DeepseekOcrConfig()

    print("Converting state dict...")
    converted_state_dict = convert_state_dict(original_state_dict, config)
    reference_dtype = next(iter(original_state_dict.values())).dtype

    print("Creating model...")
    model = DeepseekOcrForConditionalGeneration(config)
    model.to(dtype=reference_dtype)

    print("Loading converted state dict into model...")
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print(f"Saving converted model to {output_path}")
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    print("Conversion complete!")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo_id must be provided when --push_to_hub is set")

        print(f"Pushing model to Hub: {args.repo_id}")
        model.push_to_hub(args.repo_id)
        print("Model pushed successfully!")

    print("Copying tokenizer / processor files if available...")
    for filename in TOKENIZER_RELATED_FILES:
        source_file = checkpoint_path.parent / filename
        if source_file.exists():
            target_file = output_path / filename
            target_file.write_bytes(source_file.read_bytes())
            print(f"Copied {filename}")

    tokenizer_config_path = output_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["chat_template"] = CHAT_TEMPLATE
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    processor_config_path = output_path / "processor_config.json"
    if processor_config_path.exists():
        with open(processor_config_path, "r", encoding="utf-8") as f:
            processor_config = json.load(f)
        processor_config["chat_template"] = CHAT_TEMPLATE
        with open(processor_config_path, "w", encoding="utf-8") as f:
            json.dump(processor_config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
