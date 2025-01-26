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
from transformers.image_utils import PILImageResampling


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {

    # encoder and head
    r"encoder.(patch|image)_encoder.cls_token":                                 r"depth_pro.encoder.\1_encoder.embeddings.cls_token",
    r"encoder.(patch|image)_encoder.pos_embed":                                 r"depth_pro.encoder.\1_encoder.embeddings.position_embeddings",
    r"encoder.(patch|image)_encoder.patch_embed.proj.(weight|bias)":            r"depth_pro.encoder.\1_encoder.embeddings.patch_embeddings.projection.\2",
    r"encoder.(patch|image)_encoder.blocks.(\d+).norm(\d+).(weight|bias)":      r"depth_pro.encoder.\1_encoder.encoder.layer.\2.norm\3.\4",
    r"encoder.(patch|image)_encoder.blocks.(\d+).attn.qkv.(weight|bias)":       r"depth_pro.encoder.\1_encoder.encoder.layer.\2.attention.attention.(query|key|value).\3",
    r"encoder.(patch|image)_encoder.blocks.(\d+).attn.proj.(weight|bias)":      r"depth_pro.encoder.\1_encoder.encoder.layer.\2.attention.output.dense.\3",
    r"encoder.(patch|image)_encoder.blocks.(\d+).ls(\d+).gamma":                r"depth_pro.encoder.\1_encoder.encoder.layer.\2.layer_scale\3.lambda1",
    r"encoder.(patch|image)_encoder.blocks.(\d+).mlp.fc(\d+).(weight|bias)":    r"depth_pro.encoder.\1_encoder.encoder.layer.\2.mlp.fc\3.\4",
    r"encoder.(patch|image)_encoder.norm.(weight|bias)":                        r"depth_pro.encoder.\1_encoder.layernorm.\2",
    r"encoder.fuse_lowres.(weight|bias)":                                       r"depth_pro.encoder.fuse_image_with_low_res.\1",
    r"head.(\d+).(weight|bias)":                                                r"head.head.\1.\2",

    # fov
    r"fov.encoder.0.cls_token":                                                 r"fov_model.encoder.embeddings.cls_token",
    r"fov.encoder.0.pos_embed":                                                 r"fov_model.encoder.embeddings.position_embeddings",
    r"fov.encoder.0.patch_embed.proj.(weight|bias)":                            r"fov_model.encoder.embeddings.patch_embeddings.projection.\1",
    r"fov.encoder.0.blocks.(\d+).norm(\d+).(weight|bias)":                      r"fov_model.encoder.encoder.layer.\1.norm\2.\3",
    r"fov.encoder.0.blocks.(\d+).attn.qkv.(weight|bias)":                       r"fov_model.encoder.encoder.layer.\1.attention.attention.(query|key|value).\2",
    r"fov.encoder.0.blocks.(\d+).attn.proj.(weight|bias)":                      r"fov_model.encoder.encoder.layer.\1.attention.output.dense.\2",
    r"fov.encoder.0.blocks.(\d+).ls(\d+).gamma":                                r"fov_model.encoder.encoder.layer.\1.layer_scale\2.lambda1",
    r"fov.encoder.0.blocks.(\d+).mlp.fc(\d+).(weight|bias)":                    r"fov_model.encoder.encoder.layer.\1.mlp.fc\2.\3",
    r"fov.encoder.0.norm.(weight|bias)":                                        r"fov_model.encoder.layernorm.\1",
    r"fov.downsample.(\d+).(weight|bias)":                                      r"fov_model.global_neck.\1.\2",
    r"fov.encoder.1.(weight|bias)":                                             r"fov_model.encoder_neck.\1",
    r"fov.head.head.(\d+).(weight|bias)":                                       r"fov_model.head.\1.\2",

    # upsamples
    r"encoder.upsample_lowres.(weight|bias)":                                   r"depth_pro.encoder.feature_upsample.upsample_blocks.image.layers.0.\1",
    r"encoder.upsample_latent(\d+).(\d+).(weight|bias)": lambda match: (
        f"depth_pro.encoder.feature_upsample.upsample_blocks.intermediate_{1-int(match.group(1))}.layers.{match.group(2)}.{match.group(3)}"
    ),
    r"encoder.upsample(\d+).(\d+).(weight|bias)": lambda match: (
        f"depth_pro.encoder.feature_upsample.upsample_blocks.scaled_images_{2-int(match.group(1))}.layers.{match.group(2)}.{match.group(3)}"
    ),

    # projections between encoder and fusion
    r"decoder.convs.(\d+).weight": lambda match: (
        f"depth_pro.encoder.feature_projection.projections.{4-int(match.group(1))}.weight"
    ),

    # fusion stage
    r"decoder.fusions.(\d+).resnet(\d+).residual.(\d+).(weight|bias)": lambda match: (
        f"fusion_stage.layers.{4-int(match.group(1))}.residual_layer{match.group(2)}.convolution{(int(match.group(3))+1)//2}.{match.group(4)}"
    ),
    r"decoder.fusions.(\d+).out_conv.(weight|bias)": lambda match: (
        f"fusion_stage.layers.{4-int(match.group(1))}.projection.{match.group(2)}"
    ),
    r"decoder.fusions.(\d+).deconv.(weight|bias)": lambda match: (
        f"fusion_stage.layers.{4-int(match.group(1))}.deconv.{match.group(2)}"
    ),
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
    # file_path = "/home/geetu/work/hf/depth_pro/depth_pro.pt" # when you already have the files locally
    loaded = torch.load(file_path, weights_only=True)

    # ensure state_dict is in float32
    for key in loaded.keys():
        loaded[key] = loaded[key].to(torch.float32)

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
    image_processor = DepthProImageProcessorFast(
        do_resize=True,
        size={"height": 1536, "width": 1536},
        resample=PILImageResampling.BILINEAR,
        antialias=False,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=0.5,
        image_std=0.5,
    )
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
        default="geetu040/DepthPro",
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
