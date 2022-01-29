# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert PoolFormer checkpoints."""

import math
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import (
    PoolFormerConfig,
    PoolFormerForImageClassification,
)
from transformers.utils import logging
from transformers.utils.dummy_vision_objects import DeiTFeatureExtractor


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def replace_key_with_offset(key, offset, original_name, new_name):
    """
    Replaces the key by subtracting the offset from the original layer number
    """
    to_find = original_name.split(".")[0]
    orig_block_num = int(key[:key.find(to_find)][-4])
    layer_num = int(key[:key.find(to_find)][-2])
    new_block_num = orig_block_num - offset
    
    key = key.replace(
        f"{orig_block_num}.{layer_num}.{original_name}", 
        f"block.{new_block_num}.{layer_num}.{new_name}"
    )
    return key

def calc_crop_size(original_size, crop_pct):
    """
    Calculates the new crop size
    """
    return int(math.floor(original_size[0] / crop_pct))

def rename_keys(state_dict):
    new_state_dict = OrderedDict()
    total_embed_found, patch_emb_offset = 0, 0
    for key, value in state_dict.items():
        if key.startswith("network"):
            key = key.replace("network", "poolformer.encoder")
        if "proj" in key:
            # Works for the first embedding as well as the internal embedding layers
            if key.endswith("bias") and "patch_embed" not in key:
                patch_emb_offset += 1
            to_replace = key[:key.find("proj")]
            key = key.replace(to_replace, f"patch_embeddings.{total_embed_found}.")
            key = key.replace("proj", "projection")
            if key.endswith("bias"):
                total_embed_found += 1
        if "patch_embeddings" in key:
            key = "poolformer.encoder." + key
        if "mlp.fc1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc1", "output.conv1")
        if "mlp.fc2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc2", "output.conv2")
        if "norm1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "norm1", "before_norm")
        if "norm2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "norm2", "after_norm")
        if "layer_scale" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale", "layer_scale")
        if "head" in key:
            key = key.replace("head", "classifier")
        new_state_dict[key] = value
    return new_state_dict


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image

@torch.no_grad()
def convert_poolformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PoolFormer structure.
    """

    # load default PoolFormer configuration
    config = PoolFormerConfig()

    # set attributes based on model_name
    repo_id = "datasets/huggingface/label-files"
    size = model_name[-3:]
    config.num_labels = 1000
    filename = "imagenet-1k-id2label.json"
    expected_shape = (1, 1000)

    # set config attributes
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == "s12":
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pcent = 0.9
    elif size == "s24":
        config.depths = [4, 4, 12, 4]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pcent = 0.9
    elif size == "s36":
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pcent = 0.9
    elif size == "m36":
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pcent = 0.95
    elif size == "m48":
        config.depths = [8, 8, 24, 8]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pcent = 0.95
    else:
        raise ValueError(f"Size {size} not supported")

    # load feature extractor (only resize + normalize)
    size = (224, 224)
    feature_extractor = DeiTFeatureExtractor(
        size=size,
        resample=Image.BILINEAR,
        crop_size=224,
    )

    # Prepare image
    image = prepare_img()
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    logger.info(f"Converting model {model_name}...")

    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]

    # rename keys
    state_dict = rename_keys(state_dict)
    
    # create HuggingFace model and load state dict
    model = PoolFormerForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()

    # forward pass
    outputs = model(pixel_values)
    logits = outputs.logits

    # define expected logit slices for different models 
    if size == "s12":
        expected_slice = torch.tensor([-0.5267, -0.0709, -0.3930])
    elif size == "s24":
        expected_slice = torch.tensor([-0.0488,  0.1637, -0.0669])
    elif size == "s36":
        expected_slice = torch.tensor([-0.4368, -0.7142,  0.0809])
    elif size == "m36":
        expected_slice = torch.tensor([ 0.2335, -0.3859, -0.6380])
    elif size == "m48":
        expected_slice = torch.tensor([-0.3545,  0.1915,  0.2404])

    # verify logits
    assert logits.shape == expected_shape
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-2)

    # finally, save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="poolformer_s12",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_poolformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)