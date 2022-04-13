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
"""Convert YOLOS checkpoints from the original repository. URL: https://github.com/hustvl/YOLOS"""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download

from transformers import DetrFeatureExtractor, YolosConfig, YolosModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"yolos.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"yolos.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"yolos.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"yolos.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"yolos.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"yolos.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"yolos.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"yolos.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"yolos.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"yolos.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "yolos.embeddings.cls_token"),
            ("patch_embed.proj.weight", "yolos.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "yolos.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "yolos.embeddings.position_embeddings"),
        ]
    )

    if base_model:
        # layernorm + pooler
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
                ("pre_logits.fc.weight", "pooler.dense.weight"),
                ("pre_logits.fc.bias", "pooler.dense.bias"),
            ]
        )

        # if just the base model, we should remove "yolos" from all keys that start with "yolos"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("yolos") else pair for pair in rename_keys]
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("norm.weight", "yolos.layernorm.weight"),
                ("norm.bias", "yolos.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "yolos."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_yolos_checkpoint(yolos_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our YOLOS structure.
    """

    # define default Yolos configuration
    config = YolosConfig()
    
    # size of the architecture
    if "yolos_s" in yolos_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.load(yolos_name, map_location="cpu")["model"]
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = YolosModel(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by DetrFeatureExtractor
    feature_extractor = DetrFeatureExtractor(size=config.image_size)
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    # TODO assert logits

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {yolos_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--yolos_name",
        default="yolos_s_200_pre",
        type=str,
        help="Name of the YOLOS model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_yolos_checkpoint(args.yolos_name, args.pytorch_dump_folder_path)