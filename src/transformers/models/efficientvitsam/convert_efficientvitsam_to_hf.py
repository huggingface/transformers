# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""
Convert EfficientViT-SAM checkpoints from the original repository.

URL: https://github.com/mit-han-lab/efficientvit
"""

import argparse
import re

import torch
from huggingface_hub import hf_hub_download

from transformers import (
    EfficientViTSamConfig,
    EfficientViTSamImageProcessor,
    EfficientViTSamModel,
    EfficientViTSamProcessor,
    EfficientViTSamVisionConfig,
)


def get_config(model_name):
    if model_name == "efficientvit-sam-l0":
        vision_config = EfficientViTSamVisionConfig(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 1, 1, 4, 4],
            head_depth=4,
            image_size=512,
        )
    elif model_name == "efficientvit-sam-l1":
        vision_config = EfficientViTSamVisionConfig(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 1, 1, 6, 6],
            head_depth=8,
            image_size=512,
        )
    elif model_name == "efficientvit-sam-l2":
        vision_config = EfficientViTSamVisionConfig(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 2, 2, 8, 8],
            head_depth=12,
            image_size=512,
        )
    elif model_name == "efficientvit-sam-xl0":
        vision_config = EfficientViTSamVisionConfig(
            width_list=[32, 64, 128, 256, 512, 1024],
            depth_list=[0, 1, 1, 2, 3, 3],
            block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
            expand_list=[1, 4, 4, 4, 4, 6],
            fewer_norm_list=[False, False, False, False, True, True],
            fid_list=["stage5", "stage4", "stage3"],
            in_channel_list=[1024, 512, 256],
            head_depth=6,
            expand_ratio=4.0,
            image_size=1024,
        )
    elif model_name == "efficientvit-sam-xl1":
        vision_config = EfficientViTSamVisionConfig(
            width_list=[32, 64, 128, 256, 512, 1024],
            depth_list=[1, 2, 2, 4, 6, 6],
            block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
            expand_list=[1, 4, 4, 4, 4, 6],
            fewer_norm_list=[False, False, False, False, True, True],
            fid_list=["stage5", "stage4", "stage3"],
            in_channel_list=[1024, 512, 256],
            head_depth=12,
            expand_ratio=4.0,
            image_size=1024,
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")

    config = EfficientViTSamConfig(
        vision_config=vision_config,
    )
    return config


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    ".norm": ".layer_norm",
}


def replace_keys(state_dict, model_name):
    if "xl" in model_name:
        fid_list = ["stage5", "stage4", "stage3"]
    else:
        fid_list = ["stage4", "stage3", "stage2"]

    model_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # 1. Map neck inputs
        if "image_encoder.neck.input_ops.0.op_list.0." in new_key:
            new_key = new_key.replace(
                "image_encoder.neck.input_ops.0.op_list.0.", f"vision_encoder.neck.proj_layers.{fid_list[0]}.0."
            )
        elif "image_encoder.neck.input_ops.1.op_list.0." in new_key:
            new_key = new_key.replace(
                "image_encoder.neck.input_ops.1.op_list.0.", f"vision_encoder.neck.proj_layers.{fid_list[1]}.0."
            )
        elif "image_encoder.neck.input_ops.2.op_list.0." in new_key:
            new_key = new_key.replace(
                "image_encoder.neck.input_ops.2.op_list.0.", f"vision_encoder.neck.proj_layers.{fid_list[2]}.0."
            )
        elif "image_encoder.neck.input_ops.3.op_list.0." in new_key:
            new_key = new_key.replace(
                "image_encoder.neck.input_ops.3.op_list.0.", f"vision_encoder.neck.proj_layers.{fid_list[3]}.0."
            )

        # 2. Map neck outputs
        if "image_encoder.neck.output_ops.0.op_list.0.conv." in new_key:
            new_key = new_key.replace(
                "image_encoder.neck.output_ops.0.op_list.0.conv.", "vision_encoder.neck.proj_out.conv."
            )

        # 3. Map neck middle
        if "image_encoder.neck.middle.op_list." in new_key:
            new_key = new_key.replace("image_encoder.neck.middle.op_list.", "vision_encoder.neck.middle.")

        # 4. Map backbone stages
        match = re.search(r"image_encoder\.backbone\.stages\.(\d+)\.op_list\.", new_key)
        if match:
            stage_idx = int(match.group(1))
            new_key = new_key.replace(
                f"image_encoder.backbone.stages.{stage_idx}.op_list.",
                f"vision_encoder.backbone.stages.{stage_idx}.",
            )

        # 5. Map general image_encoder prefix
        if new_key.startswith("image_encoder."):
            new_key = new_key.replace("image_encoder.", "vision_encoder.", 1)

        # 6. Map prompt_encoder and mask_decoder keys to standard HF Sam names
        if "prompt_encoder" in new_key or "mask_decoder" in new_key:
            for key_to_modify, replacement in KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in new_key:
                    new_key = new_key.replace(key_to_modify, replacement)

            # output_hypernetworks_mlps renames
            output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps\.(\d+)\.layers\.(\d+).*"
            match_mlp = re.match(output_hypernetworks_mlps_pattern, new_key)
            if match_mlp:
                layer_nb = int(match_mlp.group(2))
                if layer_nb == 0:
                    new_key = new_key.replace("layers.0", "proj_in")
                elif layer_nb == 1:
                    new_key = new_key.replace("layers.1", "layers.0")
                elif layer_nb == 2:
                    new_key = new_key.replace("layers.2", "proj_out")

        model_state_dict[new_key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    return model_state_dict


def convert_efficientvitsam_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub):
    config = get_config(model_name)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    state_dict = replace_keys(state_dict, model_name)

    image_processor = EfficientViTSamImageProcessor(
        size={"longest_edge": config.vision_config.image_size},
        pad_size={"height": config.vision_config.image_size, "width": config.vision_config.image_size},
    )
    processor = EfficientViTSamProcessor(image_processor=image_processor)
    hf_model = EfficientViTSamModel(config)
    hf_model.eval()

    hf_model.load_state_dict(state_dict)

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        repo_id = f"mit-han-lab/{model_name}"
        processor.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = [
        "efficientvit-sam-l0",
        "efficientvit-sam-l1",
        "efficientvit-sam-l2",
        "efficientvit-sam-xl0",
        "efficientvit-sam-xl1",
    ]
    parser.add_argument(
        "--model_name",
        default="efficientvit-sam-l0",
        choices=choices,
        type=str,
        help="Name of the original model to convert",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the original checkpoint",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download("mit-han-lab/efficientvit-sam", f"{args.model_name.replace('-', '_')}.pt")

    convert_efficientvitsam_checkpoint(
        args.model_name, checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
