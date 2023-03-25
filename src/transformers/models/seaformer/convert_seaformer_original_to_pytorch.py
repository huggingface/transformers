# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert Seaformer checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SeaformerConfig,
    SeaformerImageProcessor,
    SeaformerForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_keys(state_dict, encoder_only=False):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if encoder_only and not key.startswith("head"):
            key = "seaformer.encoder." + key
        if key.startswith("backbone"):
            key = key.replace("backbone", "seaformer.encoder")
        if "patch_embed" in key:
            # replace for example patch_embed1 by patch_embeddings.0
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        if '.c.' in key:
            key = key.replace(".c.", ".convolution.")
        if '.bn.' in key:
            key = key.replace(".bn.", ".batchnorm.")
        if "seaformer.encoder.layer_norm" in key:
            # replace for example layer_norm1 by layer_norm.0
            idx = key[key.find("seaformer.encoder.layer_norm") + len("seaformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        if "linear_c" in key:
            # replace for example linear_c4 by linear_c.3
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        if key.startswith("head"):
            key = key.replace("head", "classifier")
        new_state_dict[key] = value

    return new_state_dict


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_seaformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our Seaformer structure.
    """

    # load default Seaformer configuration
    config = SeaformerConfig()
    encoder_only = False

    # set attributes based on model_name
    repo_id = "huggingface/label-files"
    size = model_name[len("SeaFormer_") : len("SeaFormer_") + 1]

    if "bs" in model_name:
        if "ade" in model_name:
            config.num_labels = 150
            filename = "ade20k-id2label.json"
            expected_shape = (1, 150, 64, 64)
        else:
            raise ValueError(f"Model {model_name} not supported")
    else:
        raise ValueError(f"Model {model_name} not supported")

    # set config attributes
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == "L":
        config.num_encoder_blocks = 3
        config.depths = [3, 3, 3]
        config.channels = [32, 64, 128, 192, 256, 320]
        config.mv2_blocks_cfgs = [
                [   [3, 3, 32, 1],  
                    [3, 4, 64, 2], 
                    [3, 4, 64, 1]],  
                [
                    [5, 4, 128, 2],  
                    [5, 4, 128, 1]],  
                [
                    [3, 4, 192, 2],  
                    [3, 4, 192, 1]],
                [
                    [5, 4, 256, 2]],  
                [
                    [3, 6, 320, 2]]
            ]
        config.emb_dims = [192, 256, 320]
        config.key_dims = [16, 20, 24]
        config.num_attention_heads = 8
        config.mlp_ratios = [2, 4, 6]
        config.in_channels = [128, 192, 256, 320]
        config.in_index = [0, 1, 2, 3]
        config.decoder_channels = 192
        config.embed_dims = [128, 160, 192]
        config.hidden_sizes = [128]

    else:
        raise ValueError(f"Size {size} not supported")

    # load feature extractor (only resize + normalize)
    feature_extractor = SeaformerImageProcessor(
        image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
    )

    # prepare image
    image = prepare_img()
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    logger.info(f"Converting model {model_name}...")

    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]

    # rename keys
    state_dict = rename_keys(state_dict, encoder_only=encoder_only)

    # create HuggingFace model and load state dict
    model = SeaformerForSemanticSegmentation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # forward pass
    outputs = model(pixel_values)
    logits = outputs.logits

    # set expected_slice based on model name
    # ADE20k checkpoints
    if model_name == "SeaFormer_L_bs32_43.8_ade":
        expected_slice = torch.tensor(
            [   
                [[ -2.0818,  -4.6320,  -5.9963], [ -3.2360,  -7.2340,  -8.7455], [ -2.9308,  -8.1080, -9.9713]],
                [[ -5.4941,  -7.2591,  -8.4649], [ -6.2536,  -8.9669, -10.4255], [ -6.1386,  -9.4373, -11.4133]],
                [[ -9.2548, -11.4705, -13.2432], [-10.3784, -13.9842, -16.0520], [-10.4125, -14.8483, -17.2390]],
            ]
        )

    # verify logits
    assert logits.shape == expected_shape
    assert torch.allclose(logits[0, :3, :3, :3], expected_slice, atol=1e-2)

    # finally, save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and feature extractor to the hub...")
        model.push_to_hub(f"Inderpreet01/seaformer-semantic-segmentation-large")
        feature_extractor.push_to_hub(f"Inderpreet01/seaformer-semantic-segmentation-large")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="seaformer.b0.512x512.ade.160k",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_seaformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
