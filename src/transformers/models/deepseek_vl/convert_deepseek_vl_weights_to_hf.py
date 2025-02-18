# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os

import regex as re
import torch
from huggingface_hub import hf_hub_download

from transformers import (
    DepthProConfig,
    DepthProForDepthEstimation,
    DepthProImageProcessorFast,
)


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # OLD: NEW

    # Sam (High Resolution)
    r"vision_model.vision_tower_high.vision_tower.pos_embed": r"model.vision_model.high_res_model.pos_embed",
    r"vision_model.vision_tower_high.vision_tower.hd_alpha_downsamples": r"",
    r"vision_model.vision_tower_high.vision_tower.patch_embed.proj.(weight|bias)": r"model.vision_model.high_res_model.patch_embed.projection.\1",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).norm(\d+).(weight|bias)": r"model.vision_model.high_res_model.layers.\1.layer_norm\2.\3",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).attn.rel_pos_(h|w)": r"model.vision_model.high_res_model.layers.\1.attn.rel_pos_\2",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).attn.qkv.(weight|bias)": r"model.vision_model.high_res_model.layers.\1.attn.qkv.\2",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).attn.proj.(weight|bias)": r"model.vision_model.high_res_model.layers.\1.attn.proj.\2",
    r"vision_model.vision_tower_high.vision_tower.blocks.(\d+).mlp.lin(\d+).(weight|bias)": r"model.vision_model.high_res_model.layers.\1.mlp.lin\2.\3",
    r"vision_model.vision_tower_high.vision_tower.neck.0.weight": r"model.vision_model.high_res_model.neck.conv1.weight",
    r"vision_model.vision_tower_high.vision_tower.neck.1.(weight|bias)": r"model.vision_model.high_res_model.neck.layer_norm1.\1",
    r"vision_model.vision_tower_high.vision_tower.neck.2.weight": r"model.vision_model.high_res_model.neck.conv2.weight",
    r"vision_model.vision_tower_high.vision_tower.neck.3.(weight|bias)": r"model.vision_model.high_res_model.neck.layer_norm2.\1",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.0.weight": r"model.vision_model.high_res_model_global_neck.conv1.weight",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.1.(weight|bias)": r"model.vision_model.high_res_model_global_neck.layer_norm1.\1",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.2.weight": r"model.vision_model.high_res_model_global_neck.conv2.weight",
    r"vision_model.vision_tower_high.vision_tower.neck_hd.3.(weight|bias)": r"model.vision_model.high_res_model_global_neck.layer_norm2.\1",
    r"vision_model.vision_tower_high.vision_tower.downsamples.0.weight": r"model.vision_model.high_res_neck.conv1.weight",
    r"vision_model.vision_tower_high.vision_tower.downsamples.1.weight": r"model.vision_model.high_res_neck.conv2.weight",
    r"vision_model.vision_tower_high.vision_tower.hd_alpha_downsamples": r"model.vision_model.high_res_alpha",

    # Siglip (Low Resolution)
    r"vision_model(?:.vision_tower_low)?.vision_tower.pos_embed": r"model.vision_model.low_res_model.vision_model.embeddings.position_embedding.weight",
    r"vision_model(?:.vision_tower_low)?.vision_tower.patch_embed.proj.(weight|bias)": r"model.vision_model.low_res_model.vision_model.embeddings.patch_embedding.\1",
    r"vision_model(?:.vision_tower_low)?.vision_tower.blocks.(\d+).attn.qkv.(weight|bias)": r"model.vision_model.low_res_model.vision_model.encoder.layers.\1.self_attn.(q|k|v)_proj.\2",
    r"vision_model(?:.vision_tower_low)?.vision_tower.blocks.(\d+).attn.proj.(weight|bias)": r"model.vision_model.low_res_model.vision_model.encoder.layers.\1.self_attn.out_proj.\2",
    r"vision_model(?:.vision_tower_low)?.vision_tower.blocks.(\d+).norm(\d+).(weight|bias)": r"model.vision_model.low_res_model.vision_model.encoder.layers.\1.layer_norm\2.\3",
    r"vision_model(?:.vision_tower_low)?.vision_tower.blocks.(\d+).mlp.fc(\d+).(weight|bias)": r"model.vision_model.low_res_model.vision_model.encoder.layers.\1.mlp.fc\2.\3",
    r"vision_model(?:.vision_tower_low)?.vision_tower.norm.(weight|bias)": r"model.vision_model.low_res_model.vision_model.post_layernorm.\1",
    r"vision_model(?:.vision_tower_low)?.vision_tower.attn_pool.latent": r"model.vision_model.low_res_model.vision_model.head.probe",
    r"vision_model(?:.vision_tower_low)?.vision_tower.attn_pool.proj.(weight|bias)": r"model.vision_model.low_res_model.vision_model.head.attention.out_proj.\1",
    r"vision_model(?:.vision_tower_low)?.vision_tower.attn_pool.norm.(weight|bias)": r"model.vision_model.low_res_model.vision_model.head.layernorm.\1",
    r"vision_model(?:.vision_tower_low)?.vision_tower.attn_pool.mlp.fc(\d+).(weight|bias)": r"model.vision_model.low_res_model.vision_model.head.mlp.fc\1.\2",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
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

def write_model(
    hf_repo_id: str,
    output_dir: str,
    safe_serialization: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Create and save config
    # ------------------------------------------------------------

    # create config
    backbone_config = {
        "model_type": "dinov2",
        "num_hidden_layers": 24,
        "patch_size": 16,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "image_size": 384,
        "use_mask_token": False,
    }
    config = DepthProConfig(
        # original implementation uses same config for all 3 models
        image_model_config=backbone_config,
        patch_model_config=backbone_config,
        fov_model_config=backbone_config,
        use_fov_model=True,
    )

    # save config
    config.save_pretrained(output_dir)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    # download and load state_dict from hf repo
    file_path = hf_hub_download(hf_repo_id, "depth_pro.pt")
    loaded = torch.load(file_path, weights_only=True)

    print("Converting model...")
    all_keys = list(loaded.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        current_parameter = loaded.pop(key)

        if "qkv" in key:
            qkv_state_dict = get_qkv_state_dict(new_key, current_parameter)
            state_dict.update(qkv_state_dict)
        else:
            state_dict[new_key] = current_parameter

    print("Loading the checkpoint in a DepthPro model.")
    model = DepthProForDepthEstimation(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")

    print("Saving the model.")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = DepthProForDepthEstimation.from_pretrained(output_dir, device_map="auto")
    print("Model reloaded successfully.")
    return model


def write_image_processor(output_dir: str):
    image_processor = DepthProImageProcessorFast()
    image_processor.save_pretrained(output_dir)
    return image_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default="apple/DepthPro",
        help="Location of official weights from apple on HF",
    )
    parser.add_argument(
        "--output_dir",
        default="apple_DepthPro",
        help="Location to write the converted model and processor",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--push_to_hub",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to push the converted model to the huggingface hub.",
    )
    parser.add_argument(
        "--hub_repo_id",
        default="apple/DepthPro-hf",
        help="Huggingface hub repo to write the converted model and processor",
    )
    args = parser.parse_args()

    model = write_model(
        hf_repo_id=args.hf_repo_id,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    image_processor = write_image_processor(
        output_dir=args.output_dir,
    )

    if args.push_to_hub:
        print("Pushing to hub...")
        model.push_to_hub(args.hub_repo_id)
        image_processor.push_to_hub(args.hub_repo_id)


if __name__ == "__main__":
    main()
