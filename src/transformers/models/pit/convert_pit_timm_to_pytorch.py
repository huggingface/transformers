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
"""Convert PiT checkpoints from the timm library."""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
import timm
from transformers import PiTConfig, PiTFeatureExtractor, PiTForImageClassification
from transformers.utils import logging
from transformers.utils.imagenet_classes import id2label


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for block_idx, num_layers in enumerate(config.depths):
        for layer_idx in range(num_layers):
            timm_prefix = f"transformers.{block_idx}.blocks.{layer_idx}"
            hf_prefix = f"pit.encoder.stages.{block_idx}.layer.{layer_idx}"

            # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
            rename_keys.append((f"{timm_prefix}.norm1.weight", f"{hf_prefix}.layernorm_before.weight"))
            rename_keys.append((f"{timm_prefix}.norm1.bias", f"{hf_prefix}.layernorm_before.bias"))
            rename_keys.append((f"{timm_prefix}.attn.proj.weight", f"{hf_prefix}.attention.output.dense.weight"))
            rename_keys.append((f"{timm_prefix}.attn.proj.bias", f"{hf_prefix}.attention.output.dense.bias"))
            rename_keys.append((f"{timm_prefix}.norm2.weight", f"{hf_prefix}.layernorm_after.weight"))
            rename_keys.append((f"{timm_prefix}.norm2.bias", f"{hf_prefix}.layernorm_after.bias"))
            rename_keys.append((f"{timm_prefix}.mlp.fc1.weight", f"{hf_prefix}.intermediate.dense.weight"))
            rename_keys.append((f"{timm_prefix}.mlp.fc1.bias", f"{hf_prefix}.intermediate.dense.bias"))
            rename_keys.append((f"{timm_prefix}.mlp.fc2.weight", f"{hf_prefix}.output.dense.weight"))
            rename_keys.append((f"{timm_prefix}.mlp.fc2.bias", f"{hf_prefix}.output.dense.bias"))

        # conv pooling
        if block_idx < (len(config.depths) - 1):
            timm_prefix = f"transformers.{block_idx}.pool"
            hf_prefix = f"pit.encoder.stages.{block_idx}.pool"

            rename_keys.extend(
                [
                    (f"{timm_prefix}.conv.weight", f"{hf_prefix}.conv.weight"),
                    (f"{timm_prefix}.conv.bias", f"{hf_prefix}.conv.bias"),
                    (f"{timm_prefix}.fc.weight", f"{hf_prefix}.fc.weight"),
                    (f"{timm_prefix}.fc.bias", f"{hf_prefix}.fc.bias"),
                ]
            )

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "pit.encoder.cls_token"),
            ("patch_embed.conv.weight", "pit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.conv.bias", "pit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "pit.embeddings.position_embeddings"),
        ]
    )

    # layernorm + classification head
    rename_keys.extend(
        [
            ("norm.weight", "pit.layernorm.weight"),
            ("norm.bias", "pit.layernorm.bias"),
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for block_idx, num_layers in range(config.depths):
        for layer_idx in range(num_layers):
            prefix = f"pit.encoder.stages.{block_idx}.layer.{layer_idx}.attention.attention"

            # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"transformers.{block_idx}.blocks.{layer_idx}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"transformers.{block_idx}.blocks.{layer_idx}.attn.qkv.bias")

            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"{prefix}.query.weight"] = in_proj_weight[: config.hidden_size, :]
            state_dict[f"{prefix}.query.bias"] = in_proj_bias[: config.hidden_size]
            state_dict[f"{prefix}.key.weight"] = in_proj_weight[config.hidden_size : config.hidden_size * 2, :]
            state_dict[f"{prefix}.key.bias"] = in_proj_bias[config.hidden_size : config.hidden_size * 2]
            state_dict[f"{prefix}.value.weight"] = in_proj_weight[-config.hidden_size :, :]
            state_dict[f"{prefix}.value.bias"] = in_proj_bias[-config.hidden_size :]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_pit_checkpoint(config_path, pit_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PiT structure.
    """

    config = PiTConfig.from_pretrained(config_path)
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load original model from timm
    timm_model = timm.create_model(pit_name, pretrained=True)
    timm_model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = timm_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = PiTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by PiTFeatureExtractor
    feature_extractor = PiTFeatureExtractor(size=config.image_size)
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    timm_logits = timm_model(pixel_values)
    assert timm_logits.shape == outputs.logits.shape
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {pit_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--config_path", type=str, help="Path to the PiTConfig json file.")
    parser.add_argument(
        "--pit_name",
        default="pit_base_patch16_224",
        type=str,
        help="Name of the PiT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_pit_checkpoint(args.config_path, args.pit_name, args.pytorch_dump_folder_path)
