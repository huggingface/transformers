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
Convert SAM-HQ checkpoints from the original repository.
URL: https://github.com/SysCV/sam-hq
Also supports downloading from HuggingFace Hub: https://huggingface.co/Uminosachi/sam-hq
"""

import argparse
import re

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SamHQConfig,
    SamHQImageProcessor,
    SamHQModel,
    SamHQProcessor,
    SamHQVisionConfig,
)

def get_config(model_name):
    if "vit_b" in model_name:
        vision_config = SamHQVisionConfig()
    elif "vit_l" in model_name:
        vision_config = SamHQVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )
    elif "vit_h" in model_name:
        vision_config = SamHQVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )

    config = SamHQConfig(
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
    "pe_layer.positional_encoding_gaussian_matrix": "shared_image_embedding.positional_embedding",
    "image_encoder": "vision_encoder",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "patch_embed.proj": "patch_embed.projection",
    ".norm": ".layer_norm",
    "blocks": "layers",
    "compress_vit.0": "compress_vit_conv1",
    "compress_vit.1": "compress_vit_norm",
    "compress_vit.3": "compress_vit_conv2",
    "embedding_encoder.0": "encoder_conv1",
    "embedding_encoder.1": "encoder_norm",
    "embedding_encoder.3": "encoder_conv2",
    "embedding_maskfeature.0": "mask_conv1",
    "embedding_maskfeature.1": "mask_norm",
    "embedding_maskfeature.3": "mask_conv2",
}


def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    return model_state_dict


def convert_samhq_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub):
    config = get_config(model_name)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = replace_keys(state_dict)

    image_processor = SamHQImageProcessor()
    processor = SamHQProcessor(image_processor=image_processor)
    hf_model = SamHQModel(config)
    hf_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model.load_state_dict(state_dict)
    hf_model = hf_model.to(device)

    # Test the model with an example image
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    input_points = [[[500, 375]]]
    input_labels = [[1]]

    inputs = processor(images=np.array(raw_image), return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = hf_model(**inputs)

    # Additional testing based on model variant
    if model_name == "sam_hq_vit_b":
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs)

    elif model_name == "sam_hq_vit_h":
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs)

        # Test with boxes
        input_boxes = ((75, 275, 1725, 850),)
        inputs = processor(images=np.array(raw_image), input_boxes=input_boxes, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs)

        # Test with multiple points
        input_points = [[[400, 650], [800, 650]]]
        input_labels = [[1, 1]]

        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs)

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        repo_id = f"Uminosachi/{model_name}"
        processor.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["sam_hq_vit_b", "sam_hq_vit_h", "sam_hq_vit_l"]
    parser.add_argument(
        "--model_name",
        default="sam_hq_vit_h",
        choices=choices,
        type=str,
        help="Name of the SAM-HQ model to convert",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the original SAM-HQ checkpoint. If not provided, will download from HuggingFace Hub.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model and processor to the hub",
    )

    args = parser.parse_args()

    
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download("Uminosachi/sam-hq", f"{args.model_name}.pth")

    convert_samhq_checkpoint(args.model_name, checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)