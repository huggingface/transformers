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
"""Convert ViTPose checkpoints from the original repository.

URL: https://github.com/vitae-transformer/vitpose
"""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import ViTFeatureExtractor, ViTPoseConfig, ViTPoseForPoseEstimation
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(name):
    if "backbone" in name:
        name = name.replace("backbone", "vitpose")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "embeddings.patch_embeddings")
    if "layers" in name:
        name = "encoder." + name
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "swin." + name

    return name


def convert_state_dict(orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            # layer_num = int(key_split[1])

            # # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
            # in_proj_weight = orig_state_dict.pop(f"blocks.{i}.attn.qkv.weight")
            # in_proj_bias = orig_state_dict.pop(f"blocks.{i}.attn.qkv.bias")
            # # next, add query, keys and values (in that order) to the state dict
            # orig_state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            #     : config.hidden_size, :
            # ]
            # orig_state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
            # orig_state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            #     config.hidden_size : config.hidden_size * 2, :
            # ]
            # orig_state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            #     config.hidden_size : config.hidden_size * 2
            # ]
            # orig_state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            #     -config.hidden_size :, :
            # ]
            # orig_state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]
            pass

        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vitpose_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViTPose structure.
    """

    # define default ViTPose configuration
    config = ViTPoseConfig()

    # size of the architecture
    if "small" in model_name:
        config.hidden_size = 768
        config.intermediate_size = 2304
        config.num_hidden_layers = 8
        config.num_attention_heads = 8
    elif "large" in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "huge" in model_name:
        config.hidden_size = 1280
        config.intermediate_size = 5120
        config.num_hidden_layers = 32
        config.num_attention_heads = 16

    # load HuggingFace model
    model = ViTPoseForPoseEstimation(config)
    model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # Check outputs on an image, prepared by ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor(size=config.image_size)
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    # TODO assert logits
    print(outputs.keys())

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vitpose_base",
        type=str,
        help="Name of the ViTPose model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/ViTPose/Original checkpoints/vitpose-b-simple.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pt file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vitpose_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
