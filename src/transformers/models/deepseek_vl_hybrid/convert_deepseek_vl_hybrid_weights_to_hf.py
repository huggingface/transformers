# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
from typing import Optional

import regex as re
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    DeepseekVLHybridConfig,
    DeepseekVLHybridForConditionalGeneration,
    DeepseekVLHybridImageProcessor,
    DeepseekVLHybridProcessor,
)
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
)


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # # Sam (High Resolution)
    r"vision_model.vision_tower_high.vision_tower.pos_embed":                                 r"model.high_res_vision_model.vision_encoder.pos_embed",
    r"vision_model.vision_tower_high.vision_tower.patch_embed.proj.(weight|bias)":            r"model.high_res_vision_model.vision_encoder.patch_embed.projection.\1",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).norm(\d+).(weight|bias)":      r"model.high_res_vision_model.vision_encoder.layers.\1.layer_norm\2.\3",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).attn.rel_pos_(h|w)":           r"model.high_res_vision_model.vision_encoder.layers.\1.attn.rel_pos_\2",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).attn.qkv.(weight|bias)":       r"model.high_res_vision_model.vision_encoder.layers.\1.attn.qkv.\2",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).attn.proj.(weight|bias)":      r"model.high_res_vision_model.vision_encoder.layers.\1.attn.proj.\2",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).mlp.lin(\d+).(weight|bias)":   r"model.high_res_vision_model.vision_encoder.layers.\1.mlp.lin\2.\3",
    r"vision_model.vision_tower_high.vision_tower.neck.0.weight":                             r"model.high_res_vision_model.vision_encoder.neck.conv1.weight",
    r"vision_model.vision_tower_high.vision_tower.neck.1.(weight|bias)":                      r"model.high_res_vision_model.vision_encoder.neck.layer_norm1.\1",
    r"vision_model.vision_tower_high.vision_tower.neck.2.weight":                             r"model.high_res_vision_model.vision_encoder.neck.conv2.weight",
    r"vision_model.vision_tower_high.vision_tower.neck.3.(weight|bias)":                      r"model.high_res_vision_model.vision_encoder.neck.layer_norm2.\1",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.0.weight":                          r"model.high_res_vision_neck.conv1.weight",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.1.(weight|bias)":                   r"model.high_res_vision_neck.layer_norm1.\1",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.2.weight":                          r"model.high_res_vision_neck.conv2.weight",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.3.(weight|bias)":                   r"model.high_res_vision_neck.layer_norm2.\1",
    r"vision_model.vision_tower_high.vision_tower.downsamples.0.weight":                      r"model.high_res_vision_proj.conv1.weight",
    r"vision_model.vision_tower_high.vision_tower.downsamples.1.weight":                      r"model.high_res_vision_proj.conv2.weight",
    r"vision_model.vision_tower_high.vision_tower.hd_alpha_downsamples":                      r"model.high_res_vision_alpha",

    # Siglip (Low Resolution)
    r"vision_model.vision_tower_low.vision_tower.pos_embed":                                  r"model.vision_model.vision_model.embeddings.position_embedding.weight",
    r"vision_model.vision_tower_low.vision_tower.patch_embed.proj.(weight|bias)":             r"model.vision_model.vision_model.embeddings.patch_embedding.\1",
    r"vision_model.vision_tower_low.vision_tower.blocks.(\d+).attn.qkv.(weight|bias)":        r"model.vision_model.vision_model.encoder.layers.\1.self_attn.(q|k|v)_proj.\2",
    r"vision_model.vision_tower_low.vision_tower.blocks.(\d+).attn.proj.(weight|bias)":       r"model.vision_model.vision_model.encoder.layers.\1.self_attn.out_proj.\2",
    r"vision_model.vision_tower_low.vision_tower.blocks.(\d+).norm(\d+).(weight|bias)":       r"model.vision_model.vision_model.encoder.layers.\1.layer_norm\2.\3",
    r"vision_model.vision_tower_low.vision_tower.blocks.(\d+).mlp.fc(\d+).(weight|bias)":     r"model.vision_model.vision_model.encoder.layers.\1.mlp.fc\2.\3",
    r"vision_model.vision_tower_low.vision_tower.norm.(weight|bias)":                         r"model.vision_model.vision_model.post_layernorm.\1",
    r"vision_model.vision_tower_low.vision_tower.attn_pool.latent":                           r"model.vision_model.vision_model.head.probe",
    r"vision_model.vision_tower_low.vision_tower.attn_pool.proj.(weight|bias)":               r"model.vision_model.vision_model.head.attention.out_proj.\1",
    r"vision_model.vision_tower_low.vision_tower.attn_pool.norm.(weight|bias)":               r"model.vision_model.vision_model.head.layernorm.\1",
    r"vision_model.vision_tower_low.vision_tower.attn_pool.mlp.fc(\d+).(weight|bias)":        r"model.vision_model.vision_model.head.mlp.fc\1.\2",

    # Vision Projection
    r"aligner.layers.1.(weight|bias)":        r"model.aligner.proj.\1",
    r"aligner.low_up_proj.(weight|bias)":     r"model.aligner.vision_proj.\1",
    r"aligner.high_up_proj.(weight|bias)":    r"model.aligner.high_res_vision_proj.\1",

    # Llama (Text Model)
    r"language_model.model.(\w+)":            r"model.language_model.\1",
    r"language_model.lm_head.(weight|bias)":  r"lm_head.\1",
}
# fmt: on

# Adopted from https://github.com/deepseek-ai/DeepSeek-VL/blob/main/deepseek_vl/utils/conversation.py#L80-L91
CHAT_TEMPLATE = (
    # Define separators and initialize counter
    "{% set seps = ['\n\n', '<\uff5cend\u2581of\u2581sentence\uff5c>'] %}"
    "{% set i = 0 %}"
    # Start with default system prompt
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.\n\n"
    # Iterate through messages
    "{% for message in messages %}"
    # Identify user or assistant role
    "{% if message['role']|lower == 'user' %}"
    "User: "
    "{% elif message['role']|lower == 'assistant' %}"
    "Assistant:{% if not (loop.last and not add_generation_prompt and message['content'][0]['type']=='text' and message['content'][0]['text']=='') %} {% endif %}"
    "{% else %}"
    "{{ message['role'].capitalize() }}: "
    "{% endif %}"
    # Iterate through message content (text/images)
    "{% for content in message['content'] %}"
    # If content is an image, replace with placeholder
    "{% if content['type'] == 'image' %}"
    "<image_placeholder>"
    # If content is text, handle formatting
    "{% elif content['type'] == 'text' %}"
    "{% set text = content['text'] %}"
    # Strip whitespace for first and last text blocks
    "{% if loop.first %}{% set text = text.lstrip() %}{% endif %}"
    "{% if loop.last %}{% set text = text.rstrip() %}{% endif %}"
    # If previous content was text, add space
    "{% if not loop.first and message['content'][loop.index0-1]['type'] == 'text' %}"
    "{{ ' ' + text }}"
    "{% else %}"
    "{{ text }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"  # End message content loop
    # Add separators between messages
    "{% if not loop.last or add_generation_prompt %}"
    "{% if message['role']|lower == 'user' %}"
    "{{ seps[0] }}"
    "{% else %}"
    "{{ seps[1] }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"  # End messages loop
    # Add final Assistant prompt if required
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)


def convert_old_keys_to_new_keys(state_dict_keys: dict):
    output_dict = {}

    old_text = "\n".join(state_dict_keys)
    new_text = old_text
    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        if replacement is None:
            new_text = re.sub(pattern, "", new_text)  # an empty line
            continue
        new_text = re.sub(pattern, replacement, new_text)
    output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))

    return output_dict


def get_qkv_state_dict(key, parameter):
    """
    new key which looks like this
    xxxx.(q|k|v).xxx    (m, n)

    is converted to
    xxxx.q.xxxx         (m//3, n)
    xxxx.k.xxxx         (m//3, n)
    xxxx.v.xxxx         (m//3, n)
    """
    qkv_state_dict = {}
    placeholder = re.search(r"(\(.*?\))", key).group(1)  # finds   "(query|key|value)"
    replacements_keys = placeholder[1:-1].split("|")  # creates ['query', 'key', 'value']
    replacements_vals = torch.split(
        parameter, split_size_or_sections=parameter.size(0) // len(replacements_keys), dim=0
    )
    for replacement_key, replacement_val in zip(replacements_keys, replacements_vals):
        qkv_state_dict[key.replace(placeholder, replacement_key)] = replacement_val
    return qkv_state_dict


def update_state_dict(old_state_dict):
    all_keys = list(old_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        current_parameter = old_state_dict.pop(key)

        if "qkv" in key and "vision_tower_high" not in key:
            qkv_state_dict = get_qkv_state_dict(new_key, current_parameter)
            state_dict.update(qkv_state_dict)
        elif "pos_embed" in key:
            if "vision_tower_high" not in key:
                # timm implementation of siglip creates this param of size [1, 576, 1024]
                # transformers implementation of siglip creates this param of size [576, 1024]
                state_dict[new_key] = current_parameter.squeeze(0)
            else:
                state_dict[new_key] = current_parameter
        else:
            state_dict[new_key] = current_parameter

    return state_dict


def load_model_state_dict(input_path: str) -> dict:
    """
    Load model state dict, handling both single and sharded files.
    """
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    single_file_path = os.path.join(input_path, "model.safetensors")

    # Check if we have a sharded model
    if os.path.exists(index_path):
        print("Loading sharded model...")
        state_dict = {}
        with open(index_path, "r") as f:
            index = json.load(f)

        # Get unique shard files and load each one only once
        unique_shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in unique_shard_files:
            print(f"Loading shard {shard_file}...")
            shard_path = os.path.join(input_path, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        return state_dict

    # Single file model
    elif os.path.exists(single_file_path):
        print("Loading single file model...")
        return load_file(single_file_path, device="cpu")

    else:
        raise ValueError(f"No model files found in {input_path}")


def convert_model(
    hf_repo_id: str,
    output_dir: Optional[str] = None,
    output_hub_path: Optional[str] = None,
    safe_serialization: bool = True,
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        input_path = snapshot_download(hf_repo_id)
    except HFValidationError:
        # If the input path is not a HF repo ID, assume it's a local path
        input_path = hf_repo_id

    # ------------------------------------------------------------
    # Create and save config
    # ------------------------------------------------------------

    config = DeepseekVLHybridConfig(
        text_config={
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "max_position_embeddings": 16384,
            "num_attention_heads": 32,
            "num_hidden_layers": 30,
            "vocab_size": 102400,
        },
        vision_config={
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "image_size": 384,
            "patch_size": 16,
            "hidden_act": "gelu",
            "vision_use_head": False,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
        },
        high_res_vision_config={
            "hidden_size": 768,
            "intermediate_size": 3072,
            "image_size": 1024,
            "patch_size": 16,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
        },
    )

    # save config
    if output_dir:
        config.save_pretrained(output_dir)
        print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert processor
    # ------------------------------------------------------------

    image_processor = DeepseekVLHybridImageProcessor(
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        high_res_image_mean=OPENAI_CLIP_MEAN,
        high_res_image_std=OPENAI_CLIP_STD,
        resample=PILImageResampling.BILINEAR,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        input_path,
        extra_special_tokens={
            "pad_token": "<｜end▁of▁sentence｜>",
            "image_token": "<image_placeholder>",
        },
    )

    processor = DeepseekVLHybridProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    if output_dir:
        print(f"Saving processor to {output_dir}...")
        processor.save_pretrained(output_dir)
    if output_hub_path:
        print(f"Pushing processor to hub at {output_hub_path}...")
        processor.push_to_hub(output_hub_path)

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    print("Creating empty model...")
    with init_empty_weights():
        model = DeepseekVLHybridForConditionalGeneration(config)

    # Load and convert state dict
    print("Loading state dict...")
    state_dict = load_model_state_dict(input_path)
    state_dict = update_state_dict(state_dict)

    # Load converted state dict
    print("Loading converted weights into model...")
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    if len(info.missing_keys) > 0:
        raise ValueError(f"Missing keys: {info.missing_keys}")

    # Tie weights before any device mapping
    print("Tying weights...")
    model.tie_weights()

    # Save the model
    if output_dir:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    if output_hub_path:
        print(f"Pushing model to hub at {output_hub_path}...")
        model.push_to_hub(output_hub_path, safe_serialization=safe_serialization)

    del state_dict, model
    gc.collect()

    # Validate the saved model if saved locally
    if output_dir:
        print("Reloading the local model to check if it's saved correctly...")
        DeepseekVLHybridForConditionalGeneration.from_pretrained(output_dir, device_map="auto")
        print("Local model reloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default="deepseek-ai/deepseek-vl-7b-chat",
        help="Location of official weights from DeepseekAI on HF",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Location to write the converted model and processor",
    )
    parser.add_argument(
        "--output_hub_path",
        default=None,
        help="Repository ID to push model to hub (e.g. 'username/model-name')",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    args = parser.parse_args()

    convert_model(
        hf_repo_id=args.hf_repo_id,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
