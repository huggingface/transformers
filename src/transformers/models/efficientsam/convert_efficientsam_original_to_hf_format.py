# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
Convert EFFICIENTSAM checkpoints from the original repository.
"""
import argparse
import re

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    EfficientSamConfig,
    SamImageProcessor,
    EfficientSamModel,
    SamProcessor,
    EfficientSamVisionConfig,
)


KEYS_TO_MODIFY_MAPPING = {
    "mask_decoder.final_output_upscaling_layers.0.0": "mask_decoder.upscale_conv1",
    "mask_decoder.final_output_upscaling_layers.1.0": "mask_decoder.upscale_conv2",
    "mask_decoder.final_output_upscaling_layers.0.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "image_encoder": "vision_encoder",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "patch_embed.proj": "patch_embed.projection",
    ".norm": ".layer_norm",
    "blocks": "layers",
    "iou_prediction_head.fc": "iou_prediction_head.proj_out",
    "mlp.layers.0.0": "mlp.lin1",
    "mlp.fc.": "mlp.lin2.",
    "mlp.fc1.": "mlp.lin1.",
    "mlp.fc2.": "mlp.lin2.",
}


def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"
    output_hypernetworks_mlps_fc_pattern = r".*.output_hypernetworks_mlps.(\d+).fc"
    iou_prediction_head_pattern = r".*.iou_prediction_head.layers.(\d+).(\d+)"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace(f"layers.{layer_nb}.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace(f"layers.{layer_nb}.0", "layers.0")
        elif re.match(output_hypernetworks_mlps_fc_pattern, key):
            key = key.replace("fc", "proj_out")
        elif re.match(iou_prediction_head_pattern, key):
            layer_nb = int(re.match(iou_prediction_head_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace(f"layers.{layer_nb}.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace(f"layers.{layer_nb}.0", "layers.0")

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    model_state_dict["prompt_encoder.point_embeddings.weight"] = torch.cat(
        [
            model_state_dict.pop("prompt_encoder.invalid_points.weight"), 
            model_state_dict.pop("prompt_encoder.bbox_top_left_embeddings.weight"),
            model_state_dict.pop("prompt_encoder.bbox_bottom_right_embeddings.weight"),
            model_state_dict.pop("prompt_encoder.point_embed.weight")
        ], dim=0
    )

    return model_state_dict


def convert_efficientsam_checkpoint(model_name, pytorch_dump_folder, push_to_hub, model_hub_id="ybelkada/segment-anything"):
    checkpoint_path = hf_hub_download(model_hub_id, f"checkpoints/{model_name}.pth")

    if "efficientsam_ti" in model_name:
        config = EfficientSamConfig()

    config.vision_config.use_rel_pos = False
    config.mask_decoder_config.hidden_act = "gelu"

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = replace_keys(state_dict)

    image_processor = SamImageProcessor(do_pad=False, size={"height": 1024, "width": 1024})

    processor = SamProcessor(image_processor=image_processor)
    hf_model = EfficientSamModel(config)

    hf_model.load_state_dict(state_dict)
    hf_model = hf_model.to("cuda")
    # TODO: push the image on the Hub 
    # img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    raw_image = Image.open("/home/younes_huggingface_co/code/EfficientSAM/figs/examples/dogs.jpg")

    input_points = [[[580, 350], [650, 350]]]
    input_labels = [[1, 1]]

    inputs = processor(images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = hf_model(**inputs)
    scores = output.iou_scores.squeeze()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["efficientsam_ti"]
    parser.add_argument(
        "--model_name",
        default="efficientsam_ti",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )
    parser.add_argument(
        "--model_hub_id",
        default="ybelkada/segment-anything",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )

    args = parser.parse_args()

    convert_efficientsam_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.model_hub_id)
