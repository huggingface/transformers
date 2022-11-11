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
"""Convert MaskFormer checkpoints from the original repository. URL: https://github.com/facebookresearch/MaskFormer"""


import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_maskformer_config(model_name: str):
    backbone_config = ResNetConfig()
    config = MaskFormerConfig(backbone_config=backbone_config)

    # TODO id2label mappings
    config.num_labels = 150

    return config


# def rename_key(name: str) -> str:
#     # stem
#     if "stem.conv1.weight" in name:
#         name = name.replace("stem.conv1.weight", "embedder.embedder.convolution.weight")
#     return name


def create_rename_keys(config):
    rename_keys = []
    # stem
    rename_keys.append(
        ("backbone.stem.conv1.weight", "model.pixel_level_module.encoder.model.embedder.embedder.convolution.weight")
    )
    rename_keys.append(
        (
            "backbone.stem.conv1.norm.weight",
            "model.pixel_level_module.encoder.model.embedder.embedder.normalization.weight",
        )
    )
    rename_keys.append(
        (
            "backbone.stem.conv1.norm.bias",
            "model.pixel_level_module.encoder.model.embedder.embedder.normalization.bias",
        )
    )
    # stages
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if layer_idx == 0:
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.weight",
                        f"model.pixel_level_module.encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.weight",
                        f"model.pixel_level_module.encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.bias",
                        f"model.pixel_level_module.encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
            # 3 convs
            for i in range(3):
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.weight",
                        f"model.pixel_level_module.encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.norm.weight",
                        f"model.pixel_level_module.encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.norm.bias",
                        f"model.pixel_level_module.encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.bias",
                    )
                )

    # FPN
    # fmt: off
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    # fmt: on

    # Transformer decoder

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our MaskFormer structure.
    """
    config = get_maskformer_config(model_name)

    # load original state_dict
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # update to torch tensors
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # for name, param in state_dict.items():
    #     print(name, param.shape)

    # load 🤗 model
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Unexpected keys:")
    for key in unexpected_keys:
        if "running" not in key:
            print(key)

    # TODO assert values
    # assert torch.allclose(logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    # assert torch.allclose(pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(f"nielsr/{model_name}")
        # feature_extractor.push_to_hub(model_name, organization="hustvl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="maskformer-resnet50-ade",
        type=str,
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/python_projecten/MaskFormer_checkpoints/MaskFormer-ResNet-50-ADE20k/model_final_d8dbeb.pkl",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
