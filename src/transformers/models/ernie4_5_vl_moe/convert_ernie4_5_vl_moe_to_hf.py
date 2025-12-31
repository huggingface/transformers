# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Converts Ernie 4.5 VL config and processor to Hugging Face format."""

import argparse
import json
import os
from pathlib import Path
from shutil import copyfile

from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import AddedToken

from transformers import (
    AutoTokenizer,
    Ernie4_5_VL_MoeConfig,
    Ernie4_5_VL_MoeImageProcessorFast,
    Ernie4_5_VL_MoeProcessor,
    Ernie4_5_VL_MoeVideoProcessor,
    LlamaTokenizer,
)


CONFIG_NAME = "config.json"
VALID_VISION_CONFIG_KEYS = [
    "depth",
    "hidden_size",
    "hidden_act",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
]
VALID_TEXT_CONFIG_KEYS = [
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "moe_intermediate_size",
    "moe_k",
    "moe_layer_interval",
    "moe_num_shared_experts",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
    "tie_word_embeddings",
    "use_cache",
    "use_bias",
]
TEXT_TO_VISION_CONFIG_KEYS = [
    "spatial_conv_size",
    "temporal_conv_size",
]
ALL_VISION_CONFIG_KEYS = VALID_VISION_CONFIG_KEYS + TEXT_TO_VISION_CONFIG_KEYS + ["intermediate_size"]
ALL_TEXT_CONFIG_KEYS = VALID_TEXT_CONFIG_KEYS + [
    "hidden_act",
    "mlp_layer_types",
    "moe_num_experts",
    "rope_parameters",
]

TMP_TOKENIZER_DIR = "/tmp/ernie_vl_tokenizer"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
DEFAULT_CHAT_TEMPLATE = """
{%- set image_count = namespace(value=0) -%}
{%- set video_count = namespace(value=0) -%}
{{- '<|begin_of_sentence|>' }}
{%- for message in messages -%}
    {%- if message.role in ['system', 'user'] -%}
        {%- if message.role == 'user' -%}
            {{- 'User: ' -}}
        {%- endif -%}
        {%- if message.content is string -%}
            {{- message.content -}}
        {%- else -%}
            {%- for content_item in message.content -%}
                {%- if content_item.type == 'text' -%}
                    {{- content_item.text -}}
                {%- elif content_item.type in ['image_url', 'image'] -%}
                    {%- set image_count.value = image_count.value + 1 -%}
                    Picture {{ image_count.value }}:<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>
                {%- elif content_item.type in ['video_url', 'video'] -%}
                    {%- set video_count.value = video_count.value + 1 -%}
                    Video {{ video_count.value }}:<|VIDEO_START|><|VIDEO_PLACEHOLDER|><|VIDEO_END|>
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.role == 'system' -%}
            {{- '
            ' -}}
        {%- endif -%}
    {%- elif message.role == 'assistant' -%}
        {%- macro extract_text_content(content_field) -%}
            {%- if content_field is string -%}
                {{- content_field -}}
            {%- elif content_field is iterable and content_field is not string -%}
                {%- set ns = namespace(text_parts=[]) -%}
                {%- set text_parts = [] -%}
                {%- for item in content_field -%}
                    {%- if item.type == 'text' -%}
                        {%- set ns.text_parts = ns.text_parts + [item.text] -%}
                    {%- endif -%}
                {%- endfor -%}
                {{- ns.text_parts | join("") -}}
            {%- else -%}
                {{- '' -}}
            {%- endif -%}
        {%- endmacro -%}
        {%- set reasoning_content = extract_text_content(message.reasoning_content) -%}
        {%- set content = extract_text_content(message.content) -%}
        {%- if '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('
                        ').split('<think>')[-1].lstrip('
                        ') %}
            {%- set content = content.split('</think>')[-1].lstrip('
                        ') %}
        {%- endif %}
        {%- if reasoning_content %}
            {{- '
            ' + 'Assistant: ' + '<think>
            ' + reasoning_content.strip('
                        ') + '
            </think>
            ' + content.lstrip('
            ') }}
        {%- else %}
            {{- '
            ' + 'Assistant: ' + content }}
        {%- endif %}
        {{- '<|end_of_sentence |>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt is not defined or add_generation_prompt is true %}
    {{- '\nAssistant: ' -}}
    {%- if (enable_thinking is defined and enable_thinking is false) or enable_thinking is not defined %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
    {%- if enable_thinking is defined and enable_thinking is true %}{{- '<think>' }}{%- endif %}
{%- endif %}"""
FONT_REPO = "AntonV/ernie4_5_fonts"
FONT_NAME = "Roboto-Regular.ttf"


def load_json(save_dir, filename):
    with open(os.path.join(save_dir, filename), "r") as f:
        return json.load(f)


def write_json(json_object, save_dir, filename):
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(json_object, f, indent=2, sort_keys=True, ensure_ascii=False)


def convert_vision_config_to_hf(vision_config, original_config, original_vision_config):
    # convert vision related stuff
    for key in VALID_VISION_CONFIG_KEYS:
        vision_config[key] = original_vision_config[key]
    vision_config["intermediate_size"] = original_vision_config["hidden_size"] * original_vision_config["mlp_ratio"]

    # convert originally text attributes to vision
    for key in TEXT_TO_VISION_CONFIG_KEYS:
        vision_config[key.replace("conv", "merge")] = original_config[key]
    vision_config["rms_norm_eps"] = 1e-6

    # delete everything else
    for key in list(vision_config.keys()):
        if key not in ALL_VISION_CONFIG_KEYS:
            del vision_config[key]

    return vision_config


def convert_text_config_to_hf(text_config, original_config):
    # carry directly over
    for key in VALID_TEXT_CONFIG_KEYS:
        text_config[key] = original_config.get(key)

    # special cases
    text_config["hidden_act"] = "silu"  # default value which is not explicit in their json
    text_config["use_cache"] = True  # not always included but we should default to `True`
    text_config["moe_num_experts"] = original_config["moe_num_experts"][0]  # the same for both modalities
    text_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 500_000.0,
        "mrope_section": [22, 22, 20],
    }
    if text_config["moe_num_shared_experts"] is None:
        text_config["moe_num_shared_experts"] = 0

    # ernie logic to construct mlp/moe layers
    text_config["mlp_layer_types"] = []
    for layer_idx in range(text_config["num_hidden_layers"]):
        if (
            ((layer_idx + 1) % text_config["moe_layer_interval"] == 0)
            and layer_idx >= min(original_config["moe_layer_start_index"])
            and layer_idx <= max(original_config["moe_layer_end_index"])
        ):
            text_config["mlp_layer_types"].append("sparse")
        else:
            text_config["mlp_layer_types"].append("dense")
    text_config.pop("moe_layer_interval", None)

    # delete everything else
    for key in list(text_config.keys()):
        if key not in ALL_TEXT_CONFIG_KEYS:
            del text_config[key]

    return text_config


def convert_config(model_path, save_dir):
    checkpoint_path = snapshot_download(repo_id=model_path, allow_patterns=["*config*"])
    for filename in sorted(os.listdir(checkpoint_path)):
        if filename == CONFIG_NAME:
            hf_config = Ernie4_5_VL_MoeConfig()
            original_config = load_json(checkpoint_path, filename)

            # general config
            image_token_id = original_config["im_patch_id"]

            # vision config
            vision_config = hf_config.vision_config.to_dict()
            original_vision_config = original_config["vision_config"]
            vision_config = convert_vision_config_to_hf(vision_config, original_config, original_vision_config)

            # text config
            text_config = hf_config.text_config.to_dict()
            text_config = convert_text_config_to_hf(text_config, original_config)

            # total config
            final_config = Ernie4_5_VL_MoeConfig(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=image_token_id,
            )

            final_config.save_pretrained(save_dir)
            break
    print("Converted model config\n")


def convert_tokenizer(original_tokenizer_path, save_dir):
    # Load in legacy mode
    hf_tok = LlamaTokenizer.from_pretrained(
        original_tokenizer_path,
        pad_token="<unk>",
        cls_token="<|begin_of_sentence|>",
        sep_token="<|end_of_sentence|>",
        mask_token="<mask:1>",
        add_bos_token=False,
        add_prefix_space=False,
        chat_template=DEFAULT_CHAT_TEMPLATE,
    )
    hf_tok.model_max_length = 131072
    hf_tok.init_kwargs.pop("auto_map", None)  # remote specific
    # SPM special added but we want to treat them as non-special
    hf_tok.add_tokens([AddedToken(f"{i}", normalized=False, special=False) for i in range(10)])
    hf_tok.save_pretrained(TMP_TOKENIZER_DIR)

    # Manipulate special tokens and add video token
    tokenizer_config = load_json(TMP_TOKENIZER_DIR, TOKENIZER_CONFIG_FILE)
    # Doubled usage of extra and inherint special tokens
    tokenizer_config["extra_special_tokens"].remove("<s>")
    tokenizer_config["extra_special_tokens"].remove("</s>")
    # SPM special added but we want to treat them as non-special
    for i in range(10):
        tokenizer_config["extra_special_tokens"].remove(f"{i}")
    # Removed from list, re-add
    tokenizer_config["extra_special_tokens"].append("<|IMAGE_PLACEHOLDER|>")
    tokenizer_config["extra_special_tokens"].append("<|IMAGE_START|>")
    tokenizer_config["extra_special_tokens"].append("<|IMAGE_END|>")
    tokenizer_config["extra_special_tokens"].append("<|VIDEO_PLACEHOLDER|>")
    tokenizer_config["extra_special_tokens"].append("<|VIDEO_START|>")
    tokenizer_config["extra_special_tokens"].append("<|VIDEO_END|>")
    tokenizer_config["extra_special_tokens"].append("<think>")
    tokenizer_config["extra_special_tokens"].append("</think>")
    # To be called via `.xxx_token`
    tokenizer_config |= {
        "image_token": "<|IMAGE_PLACEHOLDER|>",
        "image_end_token": "<|IMAGE_END|>",
        "image_start_token": "<|IMAGE_START|>",
        "video_token": "<|VIDEO_PLACEHOLDER|>",
        "video_end_token": "<|VIDEO_END|>",
        "video_start_token": "<|VIDEO_START|>",
    }
    write_json(tokenizer_config, TMP_TOKENIZER_DIR, TOKENIZER_CONFIG_FILE)

    # Reload and save to get correct formatting
    tokenizer = AutoTokenizer.from_pretrained(TMP_TOKENIZER_DIR)
    tokenizer.save_pretrained(save_dir)


def convert_processor(model_path, save_dir):
    print("Starting to convert processor")

    convert_tokenizer(model_path, save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    # font used within the video processor
    copyfile(hf_hub_download(FONT_REPO, FONT_NAME), Path(save_dir, FONT_NAME))

    processor = Ernie4_5_VL_MoeProcessor(
        image_processor=Ernie4_5_VL_MoeImageProcessorFast(),
        tokenizer=tokenizer,
        video_processor=Ernie4_5_VL_MoeVideoProcessor(font=str(Path(save_dir, FONT_NAME))),
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(save_dir)

    print("Finished converting the processor\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="baidu/ERNIE-4.5-VL-28B-A3B-PT",
        help="Path to the downloaded checkpoint",
    )
    parser.add_argument("--output_folder", default="AntonV/ErnieVL", type=str, help="Path to your output directory.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=True,
        help="Whether or not the preprocessor (tokenizer + image/video processors) should be converted along with the model.",
    )
    args = parser.parse_args()

    convert_config(args.checkpoint_path, args.output_folder)
    if args.convert_preprocessor:
        convert_processor(args.checkpoint_path, args.output_folder)

    print(f"Saved converted checkpoint to {args.output_folder}")
