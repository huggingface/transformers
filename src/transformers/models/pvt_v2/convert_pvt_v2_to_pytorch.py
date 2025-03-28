# coding=utf-8
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
"""Convert PvtV2 checkpoints from the original library."""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import PvtImageProcessor, PvtV2Config, PvtV2ForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    for i in range(config.num_encoder_blocks):
        # Remane embedings' paramters
        rename_keys.append(
            (f"patch_embed{i + 1}.proj.weight", f"pvt_v2.encoder.layers.{i}.patch_embedding.proj.weight")
        )
        rename_keys.append((f"patch_embed{i + 1}.proj.bias", f"pvt_v2.encoder.layers.{i}.patch_embedding.proj.bias"))
        rename_keys.append(
            (f"patch_embed{i + 1}.norm.weight", f"pvt_v2.encoder.layers.{i}.patch_embedding.layer_norm.weight")
        )
        rename_keys.append(
            (f"patch_embed{i + 1}.norm.bias", f"pvt_v2.encoder.layers.{i}.patch_embedding.layer_norm.bias")
        )
        rename_keys.append((f"norm{i + 1}.weight", f"pvt_v2.encoder.layers.{i}.layer_norm.weight"))
        rename_keys.append((f"norm{i + 1}.bias", f"pvt_v2.encoder.layers.{i}.layer_norm.bias"))

        for j in range(config.depths[i]):
            # Rename blocks' parameters
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.q.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.query.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.q.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.query.bias")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.kv.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.kv.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.bias")
            )

            if config.linear_attention or config.sr_ratios[i] > 1:
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.norm.weight",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.layer_norm.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.norm.bias",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.layer_norm.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.sr.weight",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.spatial_reduction.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.sr.bias",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.spatial_reduction.bias",
                    )
                )

            rename_keys.append(
                (f"block{i + 1}.{j}.attn.proj.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.proj.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.proj.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.proj.bias")
            )

            rename_keys.append(
                (f"block{i + 1}.{j}.norm1.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_1.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.norm1.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_1.bias")
            )

            rename_keys.append(
                (f"block{i + 1}.{j}.norm2.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_2.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.norm2.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_2.bias")
            )

            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc1.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense1.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc1.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense1.bias")
            )
            rename_keys.append(
                (
                    f"block{i + 1}.{j}.mlp.dwconv.dwconv.weight",
                    f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dwconv.dwconv.weight",
                )
            )
            rename_keys.append(
                (
                    f"block{i + 1}.{j}.mlp.dwconv.dwconv.bias",
                    f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dwconv.dwconv.bias",
                )
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc2.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense2.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc2.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense2.bias")
            )

    rename_keys.extend(
        [
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_k_v(state_dict, config):
    # for each of the encoder blocks:
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # read in weights + bias of keys and values (which is a single matrix in the original implementation)
            kv_weight = state_dict.pop(f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.weight")
            kv_bias = state_dict.pop(f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.bias")
            # next, add keys and values (in that order) to the state dict
            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.key.bias"] = kv_bias[: config.hidden_sizes[i]]

            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.value.bias"] = kv_bias[
                config.hidden_sizes[i] :
            ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_pvt_v2_checkpoint(pvt_v2_size, pvt_v2_checkpoint, pytorch_dump_folder_path, verify_imagenet_weights=False):
    """
    Copy/paste/tweak model's weights to our PVT structure.
    """

    # define default PvtV2 configuration
    if pvt_v2_size == "b0":
        config_path = "OpenGVLab/pvt_v2_b0"
    elif pvt_v2_size == "b1":
        config_path = "OpenGVLab/pvt_v2_b1"
    elif pvt_v2_size == "b2":
        config_path = "OpenGVLab/pvt_v2_b2"
    elif pvt_v2_size == "b2-linear":
        config_path = "OpenGVLab/pvt_v2_b2_linear"
    elif pvt_v2_size == "b3":
        config_path = "OpenGVLab/pvt_v2_b3"
    elif pvt_v2_size == "b4":
        config_path = "OpenGVLab/pvt_v2_b4"
    elif pvt_v2_size == "b5":
        config_path = "OpenGVLab/pvt_v2_b5"
    else:
        raise ValueError(
            f"Available model sizes: 'b0', 'b1', 'b2', 'b2-linear', 'b3', 'b4', 'b5', but '{pvt_v2_size}' was given"
        )
    config = PvtV2Config.from_pretrained(config_path)
    # load original model from https://github.com/whai362/PVT
    state_dict = torch.load(pvt_v2_checkpoint, map_location="cpu")

    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_k_v(state_dict, config)

    # load HuggingFace model
    model = PvtV2ForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    image_processor = PvtImageProcessor(size=config.image_size)

    if verify_imagenet_weights:
        # Check outputs on an image, prepared by PvtImageProcessor
        print("Verifying conversion of pretrained ImageNet weights...")
        encoding = image_processor(images=prepare_img(), return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        outputs = model(pixel_values)
        logits = outputs.logits.detach().cpu()

        if pvt_v2_size == "b0":
            expected_slice_logits = torch.tensor([-1.1939, -1.4547, -0.1076])
        elif pvt_v2_size == "b1":
            expected_slice_logits = torch.tensor([-0.4716, -0.7335, -0.4600])
        elif pvt_v2_size == "b2":
            expected_slice_logits = torch.tensor([0.0795, -0.3170, 0.2247])
        elif pvt_v2_size == "b2-linear":
            expected_slice_logits = torch.tensor([0.0968, 0.3937, -0.4252])
        elif pvt_v2_size == "b3":
            expected_slice_logits = torch.tensor([-0.4595, -0.2870, 0.0940])
        elif pvt_v2_size == "b4":
            expected_slice_logits = torch.tensor([-0.1769, -0.1747, -0.0143])
        elif pvt_v2_size == "b5":
            expected_slice_logits = torch.tensor([-0.2943, -0.1008, 0.6812])
        else:
            raise ValueError(
                f"Available model sizes: 'b0', 'b1', 'b2', 'b2-linear', 'b3', 'b4', 'b5', but "
                f"'{pvt_v2_size}' was given"
            )

        assert torch.allclose(
            logits[0, :3], expected_slice_logits, atol=1e-4
        ), "ImageNet weights not converted successfully."

        print("ImageNet weights verified, conversion successful.")

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model pytorch_model.bin to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pvt_v2_size",
        default="b0",
        type=str,
        help="Size of the PVTv2 pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pvt_v2_checkpoint",
        default="pvt_v2_b0.pth",
        type=str,
        help="Checkpoint of the PVTv2 pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--verify-imagenet-weights",
        action="store_true",
        default=False,
        help="Verifies the correct conversion of author-published pretrained ImageNet weights.",
    )

    args = parser.parse_args()
    convert_pvt_v2_checkpoint(
        pvt_v2_size=args.pvt_v2_size,
        pvt_v2_checkpoint=args.pvt_v2_checkpoint,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        verify_imagenet_weights=args.verify_imagenet_weights,
    )
