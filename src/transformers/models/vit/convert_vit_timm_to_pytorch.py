# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert ViT checkpoints from the timm library."""


import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from packaging import version
from PIL import Image
from torch import nn

import requests
import timm
from transformers import ViTConfig, ViTForImageClassification, ViTModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            ("blocks." + str(i) + ".norm1.weight", "vit.encoder.layer." + str(i) + ".layernorm_before.weight")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".norm1.bias", "vit.encoder.layer." + str(i) + ".layernorm_before.bias")
        )
        rename_keys.append(
            (
                "blocks." + str(i) + ".attn.proj.weight",
                "vit.encoder.layer." + str(i) + ".attention.output.dense.weight",
            )
        )
        rename_keys.append(
            ("blocks." + str(i) + ".attn.proj.bias", "vit.encoder.layer." + str(i) + ".attention.output.dense.bias")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".norm2.weight", "vit.encoder.layer." + str(i) + ".layernorm_after.weight")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".norm2.bias", "vit.encoder.layer." + str(i) + ".layernorm_after.bias")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".mlp.fc1.weight", "vit.encoder.layer." + str(i) + ".intermediate.dense.weight")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".mlp.fc1.bias", "vit.encoder.layer." + str(i) + ".intermediate.dense.bias")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".mlp.fc2.weight", "vit.encoder.layer." + str(i) + ".output.dense.weight")
        )
        rename_keys.append(
            ("blocks." + str(i) + ".mlp.fc2.bias", "vit.encoder.layer." + str(i) + ".output.dense.bias")
        )

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "vit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )

    # classification head
    rename_keys.extend(
        [
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
            ("norm.weight", "layernorm.weight"),
            ("norm.bias", "layernorm.bias"),
        ]
    )

    # to do: add base model support
    # if just the base model, we should remove "vit" from all keys
    if base_model:
        pass

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop("blocks." + str(i) + ".attn.qkv.weight")
        in_proj_bias = state_dict.pop("blocks." + str(i) + ".attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict["vit.encoder.layer." + str(i) + ".attention.self.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict["vit.encoder.layer." + str(i) + ".attention.self.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict["vit.encoder.layer." + str(i) + ".attention.self.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict["vit.encoder.layer." + str(i) + ".attention.self.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict["vit.encoder.layer." + str(i) + ".attention.self.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict["vit.encoder.layer." + str(i) + ".attention.self.value.bias"] = in_proj_bias[-config.hidden_size :]

    # to do: add base model support
    if base_model:
        pass


def remove_classification_head_(state_dict):
    ignore_keys = [
        "norm.weight",
        "norm.bias",
        "head.weight",
        "head.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img(image_resolution):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose(
        [T.Resize((image_resolution, image_resolution)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    return img


@torch.no_grad()
def convert_vit_checkpoint(vit_name, pytorch_dump_folder_path, base_model=False):
    """
    Copy/paste/tweak model's weights to our ViT structure.
    """

    # define HuggingFace configuration
    config = ViTConfig()
    if vit_name == "vit_base_patch16_224":
        config.num_labels = 1000
    elif vit_name == "vit_base_patch32_224":
        config.patch_size = 32
        config.num_labels = 1000
    elif vit_name == "vit_base_patch16_384":
        config.image_size = 384
        config.num_labels = 1000
    elif vit_name == "vit_base_patch32_384":
        config.image_size = 384
        config.patch_size = 32
        config.num_labels = 1000

    # load original model from timm
    vit = timm.create_model(vit_name, pretrained=True)
    vit.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = vit.state_dict()
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)
    if base_model:
        remove_classification_head_(state_dict)

    model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check logits on an image
    img = prepare_img(config.image_size)
    logits = vit(img)
    outputs = model(img)

    assert logits.shape == outputs.logits.shape
    assert torch.allclose(logits, outputs.logits, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--vit_name",
        default="vit_base_patch16_224",
        type=str,
        help="Name of the ViT timm model you'd like to convert, currently supports ViT base models.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--base_model",
        default=False,
        action="store_true",
        help="Whether to just load the base model without any head.",
    )
    args = parser.parse_args()
    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path)
