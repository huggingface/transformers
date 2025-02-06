# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DFineConfig, DFineForObjectDetection, DFineResNetStageConfig, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_d_fine_config(model_name: str) -> DFineConfig:
    config = DFineConfig()

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    config.backbone_config.hidden_sizes = [64, 128, 256, 512]
    config.backbone_config.layer_type = "basic"
    config.backbone_config.embedding_size = 32
    config.hidden_expansion = 1.0
    config.decoder_layers = 6

    if model_name in ["dfine_x_coco", "dfine_x_obj2coco", "dfine_x_obj365"]:
        config.backbone_config.stage_config = DFineResNetStageConfig(
            stage1=[64, 64, 128, 1, False, False, 3, 6],
            stage2=[128, 128, 512, 2, True, False, 3, 6],
            stage3=[512, 256, 1024, 5, True, True, 5, 6],
            stage4=[1024, 512, 2048, 2, True, True, 5, 6],
        )
        config.backbone_config.stem_channels = [3, 32, 64]
        config.encoder_in_channels = [512, 1024, 2048]
        config.encoder_hidden_dim = 384
        config.encoder_ffn_dim = 2048
        config.decoder_n_points = [3, 6, 3]
        config.decoder_in_channels = [384, 384, 384]
        if model_name == "dfine_x_obj365":
            config.num_labels = 366
    elif model_name in ["dfine_m_coco", "dfine_m_obj2coco", "dfine_m_obj365"]:
        config.backbone_config.stem_channels = [3, 24, 32]
        config.backbone_config.stage_config = DFineResNetStageConfig(
            stage1=[32, 32, 96, 1, False, False, 3, 4],
            stage2=[96, 64, 384, 1, True, False, 3, 4],
            stage3=[384, 128, 768, 3, True, True, 5, 4],
            stage4=[768, 256, 1536, 1, True, True, 5, 4],
        )
        config.decoder_layers = 4
        config.encoder_in_channels = [384, 768, 1536]
        config.backbone_config.use_lab = True
        config.depth_mult = 0.67
        if model_name == "dfine_m_obj365":
            config.num_labels = 366
    elif model_name in ["dfine_l_coco", "dfine_l_obj2coco_e25", "dfine_l_obj365"]:
        config.backbone_config.stem_channels = [3, 32, 48]
        config.backbone_config.stage_config = DFineResNetStageConfig(
            stage1=[48, 48, 128, 1, False, False, 3, 6],
            stage2=[128, 96, 512, 1, True, False, 3, 6],
            stage3=[512, 192, 1024, 3, True, True, 5, 6],
            stage4=[1024, 384, 2048, 1, True, True, 5, 6],
        )
        config.encoder_ffn_dim = 1024
        config.encoder_in_channels = [512, 1024, 2048]
        if model_name == "dfine_l_obj365":
            config.num_labels = 366
    else:
        config.backbone_config.stem_channels = [3, 16, 16]
        config.backbone_config.stage_config = DFineResNetStageConfig(
            stage1=[16, 16, 64, 1, False, False, 3, 3],
            stage2=[64, 32, 256, 1, True, False, 3, 3],
            stage3=[256, 64, 512, 2, True, True, 5, 3],
            stage4=[512, 128, 1024, 1, True, True, 5, 3],
        )
        config.decoder_layers = 3
        config.hidden_expansion = 0.5
        config.depth_mult = 0.34
        config.encoder_in_channels = [256, 512, 1024]
        config.backbone_config.use_lab = True
        if model_name == "dfine_s_obj365":
            config.num_labels = 366

    return config


def load_original_state_dict(repo_id, model_name):
    directory_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}.pth")

    original_state_dict = {}
    model = torch.load(directory_path, map_location="cpu")["model"]
    for key in model.keys():
        original_state_dict[key] = model[key]

    return original_state_dict


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []

    rename_keys.append(("decoder.valid_mask", "model.decoder.valid_mask"))
    rename_keys.append(("decoder.anchors", "model.decoder.anchors"))
    rename_keys.append(("decoder.up", "model.decoder.up"))
    rename_keys.append(("decoder.reg_scale", "model.decoder.reg_scale"))

    # stem
    # fmt: off
    last_key = ["weight", "bias", "running_mean", "running_var"]
    lab_keys = ["lab.scale", "lab.bias"]

    rename_keys.append(("backbone.stem.stem1.conv.weight", "model.backbone.model.embedder.stem1.convolution.weight"))
    rename_keys.append(("backbone.stem.stem2a.conv.weight", "model.backbone.model.embedder.stem2a.convolution.weight"))
    rename_keys.append(("backbone.stem.stem2b.conv.weight", "model.backbone.model.embedder.stem2b.convolution.weight"))
    rename_keys.append(("backbone.stem.stem3.conv.weight", "model.backbone.model.embedder.stem3.convolution.weight"))
    rename_keys.append(("backbone.stem.stem4.conv.weight", "model.backbone.model.embedder.stem4.convolution.weight"))
    for last in last_key:
        rename_keys.append((f"backbone.stem.stem1.bn.{last}", f"model.backbone.model.embedder.stem1.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem2a.bn.{last}", f"model.backbone.model.embedder.stem2a.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem2b.bn.{last}", f"model.backbone.model.embedder.stem2b.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem3.bn.{last}", f"model.backbone.model.embedder.stem3.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem4.bn.{last}", f"model.backbone.model.embedder.stem4.normalization.{last}"))
    if config.backbone_config.use_lab:
        for lab in lab_keys:
            rename_keys.append((f"backbone.stem.stem1.{lab}", f"model.backbone.model.embedder.stem1.{lab}"))
            rename_keys.append((f"backbone.stem.stem2a.{lab}", f"model.backbone.model.embedder.stem2a.{lab}"))
            rename_keys.append((f"backbone.stem.stem2b.{lab}", f"model.backbone.model.embedder.stem2b.{lab}"))
            rename_keys.append((f"backbone.stem.stem3.{lab}", f"model.backbone.model.embedder.stem3.{lab}"))
            rename_keys.append((f"backbone.stem.stem4.{lab}", f"model.backbone.model.embedder.stem4.{lab}"))

    for stage_idx, stage in enumerate(config.backbone_config.stage_config):
        _, _, _, block_num, downsample, _, _, layer_num = stage
        for b in range(block_num):
            for layer_idx in range(layer_num):
                if stage_idx == 0 or stage_idx == 3:
                    rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.convolution.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.bn.{last}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.normalization.{last}",
                        )
                    )
                    if config.backbone_config.use_lab:
                        for lab in lab_keys:
                            rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.{lab}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.{lab}",
                        )
                    )
                    #layers
                    rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.conv.weigh",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.convolution.weight",
                        )
                    )
                    rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.conv.weigh",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.convolution.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append((
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.bn.{last}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.normalization.{last}",
                            ))

                    for last in last_key:
                        rename_keys.append((
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.bn.{last}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.normalization.{last}",
                            ))
                    if config.backbone_config.use_lab:
                        for lab in lab_keys:
                            rename_keys.append((
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.{lab}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.{lab}",
                            ))
                            rename_keys.append((
                            f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.{lab}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.{lab}",
                            ))


                #aggregation
                rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.{b}.aggregation.0.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.aggregation.0.convolution.weight",
                        )
                    )
                rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.{b}.aggregation.1.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.aggregation.1.convolution.weight",
                        )
                    )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.{b}.aggregation.0.bn.{last}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.aggregation.0.normalization.{last}",
                        )
                    )
                    rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.{b}.aggregation.1.bn.{last}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.aggregation.1.normalization.{last}",
                        )
                    )
                if config.backbone_config.use_lab:
                    for lab in lab_keys:
                        rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.blocks.{b}.aggregation.0.{lab}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.aggregation.0.{lab}",
                        )
                    )
                        rename_keys.append(
                            (
                                f"backbone.stages.{stage_idx}.blocks.{b}.aggregation.1.{lab}",
                                f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.aggregation.1.{lab}",
                            )
                        )

                #layers
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.normalization.{last}",
                    )
                )

                if config.backbone_config.use_lab:
                    for lab in lab_keys:
                        rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.{lab}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.{lab}",
                    )
                )

                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv1.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv1.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv1.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv1.normalization.{last}",
                        ))

                for last in last_key:
                    rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.normalization.{last}",
                        ))
                if config.backbone_config.use_lab:
                    for lab in lab_keys:
                        rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.{lab}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.{lab}",
                        ))

            #downsamle
            if downsample:
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.downsample.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.downsample.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"backbone.stages.{stage_idx}.downsample.bn.{last}",
                            f"model.backbone.model.encoder.stages.{stage_idx}.downsample.normalization.{last}",
                        )
                    )

    # fmt: on
    for i in range(config.encoder_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.weight",
                f"model.encoder.encoder.{i}.layers.0.fc1.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.bias",
                f"model.encoder.encoder.{i}.layers.0.fc1.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.weight",
                f"model.encoder.encoder.{i}.layers.0.fc2.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.bias",
                f"model.encoder.encoder.{i}.layers.0.fc2.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.weight",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.bias",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.bias",
            )
        )

    for j in range(0, 3):
        rename_keys.append((f"encoder.input_proj.{j}.conv.weight", f"model.encoder_input_proj.{j}.0.weight"))
        for last in last_key:
            rename_keys.append((f"encoder.input_proj.{j}.norm.{last}", f"model.encoder_input_proj.{j}.1.{last}"))

    for i in range(len(config.encoder_in_channels) - 1):
        # fpn_block
        # number of cv is 4 according to original implementation
        for j in range(1, 5):
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.cv{j}.conv.weight", f"model.encoder.fpn_blocks.{i}.cv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.norm.{last}",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.norm.{last}",
                    )
                )
            if j == 2 or j == 3:
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.1.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.1.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.1.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.1.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.0.conv1.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.0.conv1.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.0.conv2.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.0.conv2.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.1.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.1.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{j}.1.norm.{last}",
                            f"model.encoder.fpn_blocks.{i}.cv{j}.1.norm.{last}",
                        )
                    )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{j}.0.conv1.norm.{last}",
                            f"model.encoder.fpn_blocks.{i}.cv{j}.0.conv1.norm.{last}",
                        )
                    )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{j}.0.conv2.norm.{last}",
                            f"model.encoder.fpn_blocks.{i}.cv{j}.0.conv2.norm.{last}",
                        )
                    )

        # lateral_convs
        rename_keys.append((f"encoder.lateral_convs.{i}.conv.weight", f"model.encoder.lateral_convs.{i}.conv.weight"))
        for last in last_key:
            rename_keys.append(
                (f"encoder.lateral_convs.{i}.norm.{last}", f"model.encoder.lateral_convs.{i}.norm.{last}")
            )

        # fpn_bottlenecks
        for c in range(2, 4):
            for j in range(3):
                for k in range(1, 3):
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                            f"model.encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                                f"model.encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                            )
                        )
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.conv.weight",
                            f"model.encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.conv.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.norm.{last}",
                                f"model.encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.norm.{last}",
                            )
                        )

        # pan_block
        for j in range(1, 5):
            if j == 2 or j == 3:
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.cv{j}.0.conv1.conv.weight",
                        f"model.encoder.pan_blocks.{i}.cv{j}.0.conv1.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.cv{j}.0.conv2.conv.weight",
                        f"model.encoder.pan_blocks.{i}.cv{j}.0.conv2.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.cv{j}.0.conv1.norm.{last}",
                            f"model.encoder.pan_blocks.{i}.cv{j}.0.conv1.norm.{last}",
                        )
                    )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.cv{j}.0.conv2.norm.{last}",
                            f"model.encoder.pan_blocks.{i}.cv{j}.0.conv2.norm.{last}",
                        )
                    )
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.cv{j}.1.conv.weight",
                        f"model.encoder.pan_blocks.{i}.cv{j}.1.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.cv{j}.1.norm.{last}",
                            f"model.encoder.pan_blocks.{i}.cv{j}.1.norm.{last}",
                        )
                    )

            rename_keys.append(
                (f"encoder.pan_blocks.{i}.cv{j}.conv.weight", f"model.encoder.pan_blocks.{i}.cv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.cv{j}.norm.{last}",
                        f"model.encoder.pan_blocks.{i}.cv{j}.norm.{last}",
                    )
                )

        # pan_bottlenecks
        for c in range(2, 4):
            for j in range(3):
                for k in range(1, 3):
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                            f"model.encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                                f"model.encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                            )
                        )

    # downsample_convs
    rename_keys.append(
        ("encoder.downsample_convs.0.0.cv1.conv.weight", "model.encoder.downsample_convs.0.0.cv1.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.0.0.cv1.norm.{last}", f"model.encoder.downsample_convs.0.0.cv1.norm.{last}")
        )
    rename_keys.append(
        ("encoder.downsample_convs.1.0.cv1.conv.weight", "model.encoder.downsample_convs.1.0.cv1.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.1.0.cv1.norm.{last}", f"model.encoder.downsample_convs.1.0.cv1.norm.{last}")
        )
    rename_keys.append(
        ("encoder.downsample_convs.0.0.cv2.conv.weight", "model.encoder.downsample_convs.0.0.cv2.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.0.0.cv2.norm.{last}", f"model.encoder.downsample_convs.0.0.cv2.norm.{last}")
        )
    rename_keys.append(
        ("encoder.downsample_convs.1.0.cv2.conv.weight", "model.encoder.downsample_convs.1.0.cv2.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.1.0.cv2.norm.{last}", f"model.encoder.downsample_convs.1.0.cv2.norm.{last}")
        )

    for i in range(config.decoder_layers):
        # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.self_attn.out_proj.weight",
                f"model.decoder.layers.{i}.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.self_attn.out_proj.bias",
                f"model.decoder.layers.{i}.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.weight",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.num_points_scale",
                f"model.decoder.layers.{i}.encoder_attn.num_points_scale",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.bias",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.weight",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.bias",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.gate.weight",
                f"model.decoder.layers.{i}.gateway.gate.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.gate.bias",
                f"model.decoder.layers.{i}.gateway.gate.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.norm.weight",
                f"model.decoder.layers.{i}.gateway.norm.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.norm.bias",
                f"model.decoder.layers.{i}.gateway.norm.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.0.weight",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.0.bias",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.0.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.1.weight",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.1.bias",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.1.bias",
            )
        )

        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias")
        )

    for i in range(config.decoder_layers):
        # decoder + class and bounding box heads
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.weight",
                f"model.decoder.class_embed.{i}.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.bias",
                f"model.decoder.class_embed.{i}.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.weight",
                f"model.decoder.bbox_embed.{i}.layers.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.bias",
                f"model.decoder.bbox_embed.{i}.layers.0.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.weight",
                f"model.decoder.bbox_embed.{i}.layers.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.bias",
                f"model.decoder.bbox_embed.{i}.layers.1.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.weight",
                f"model.decoder.bbox_embed.{i}.layers.2.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.bias",
                f"model.decoder.bbox_embed.{i}.layers.2.bias",
            )
        )

    # decoder projection
    for i in range(len(config.decoder_in_channels)):
        rename_keys.append(
            (
                f"decoder.input_proj.{i}.conv.weight",
                f"model.decoder_input_proj.{i}.0.weight",
            )
        )
        for last in last_key:
            rename_keys.append(
                (
                    f"decoder.input_proj.{i}.norm.{last}",
                    f"model.decoder_input_proj.{i}.1.{last}",
                )
            )

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    rename_keys.extend(
        [
            ("decoder.denoising_class_embed.weight", "model.denoising_class_embed.weight"),
            ("decoder.query_pos_head.layers.0.weight", "model.decoder.query_pos_head.layers.0.weight"),
            ("decoder.query_pos_head.layers.0.bias", "model.decoder.query_pos_head.layers.0.bias"),
            ("decoder.query_pos_head.layers.1.weight", "model.decoder.query_pos_head.layers.1.weight"),
            ("decoder.query_pos_head.layers.1.bias", "model.decoder.query_pos_head.layers.1.bias"),
            ("decoder.enc_output.proj.weight", "model.enc_output.0.weight"),
            ("decoder.enc_output.proj.bias", "model.enc_output.0.bias"),
            ("decoder.enc_output.norm.weight", "model.enc_output.1.weight"),
            ("decoder.enc_output.norm.bias", "model.enc_output.1.bias"),
            ("decoder.enc_score_head.weight", "model.enc_score_head.weight"),
            ("decoder.enc_score_head.bias", "model.enc_score_head.bias"),
            ("decoder.enc_bbox_head.layers.0.weight", "model.enc_bbox_head.layers.0.weight"),
            ("decoder.enc_bbox_head.layers.0.bias", "model.enc_bbox_head.layers.0.bias"),
            ("decoder.enc_bbox_head.layers.1.weight", "model.enc_bbox_head.layers.1.weight"),
            ("decoder.enc_bbox_head.layers.1.bias", "model.enc_bbox_head.layers.1.bias"),
            ("decoder.enc_bbox_head.layers.2.weight", "model.enc_bbox_head.layers.2.weight"),
            ("decoder.enc_bbox_head.layers.2.bias", "model.enc_bbox_head.layers.2.bias"),
            ("decoder.pre_bbox_head.layers.0.weight", "model.decoder.pre_bbox_head.layers.0.weight"),
            ("decoder.pre_bbox_head.layers.0.bias", "model.decoder.pre_bbox_head.layers.0.bias"),
            ("decoder.pre_bbox_head.layers.1.weight", "model.decoder.pre_bbox_head.layers.1.weight"),
            ("decoder.pre_bbox_head.layers.1.bias", "model.decoder.pre_bbox_head.layers.1.bias"),
            ("decoder.pre_bbox_head.layers.2.weight", "model.decoder.pre_bbox_head.layers.2.weight"),
            ("decoder.pre_bbox_head.layers.2.bias", "model.decoder.pre_bbox_head.layers.2.bias"),
        ]
    )

    return rename_keys


def rename_key(state_dict, old, new):
    try:
        val = state_dict.pop(old)
        state_dict[new] = val
    except Exception:
        pass


def read_in_q_k_v(state_dict, config):
    prefix = ""
    encoder_hidden_dim = config.encoder_hidden_dim

    # first: transformer encoder
    for i in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
            :encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
            encoder_hidden_dim : 2 * encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
            encoder_hidden_dim : 2 * encoder_hidden_dim
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
            -encoder_hidden_dim:, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight", None)
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias", None)
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_d_fine_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our D-FINE structure.
    """

    # load default config
    config = get_d_fine_config(model_name)
    state_dict = load_original_state_dict(repo_id, model_name)
    model = DFineForObjectDetection(config)
    logger.info(f"Converting model {model_name}...")

    # rename keys
    renamed_keys = create_rename_keys(config)
    for src, dest in renamed_keys:
        rename_key(state_dict, src, dest)

    if not config.anchor_image_size:
        state_dict.pop("model.decoder.valid_mask", None)
        state_dict.pop("model.decoder.anchors", None)

    state_dict.pop("decoder.decoder.up", None)
    state_dict.pop("decoder.decoder.reg_scale", None)

    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    for key in state_dict.copy().keys():
        if key.endswith("num_batches_tracked"):
            del state_dict[key]
        # for two_stage
        if "bbox_embed" in key or ("class_embed" in key and "denoising_" not in key):
            state_dict[key.split("model.decoder.")[-1]] = state_dict[key]

    # finally, create HuggingFace model and load state dict
    model.load_state_dict(state_dict)
    model.eval()

    # load image processor
    image_processor = RTDetrImageProcessor()

    # prepare image
    img = prepare_img()

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize([640, 640], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    original_pixel_values = transformations(img).unsqueeze(0)  # insert batch dimension

    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    assert torch.allclose(original_pixel_values, pixel_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    outputs = model(pixel_values)

    if model_name == "dfine_x_coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.84472, -4.72931, -4.59713],
                [-4.55426, -4.61722, -4.62792],
                [-4.39344, -4.60641, -4.13995],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.256524, 0.54776, 0.476448],
                [0.76900, 0.41423, 0.46148],
                [0.16880, 0.19923, 0.21118],
            ]
        )
    elif model_name == "dfine_x_obj2coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.059431, -6.19076, -4.66398],
                [-3.865726, -5.84998, -4.42653],
                [-3.874609, -6.228559, -4.60206],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.25807, 0.54150, 0.47592],
                [0.76958, 0.40960, 0.45886],
                [0.169469, 0.198879, 0.21171],
            ]
        )
    elif model_name == "dfine_x_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-6.3844957, -3.7549126, -4.687326],
                [-5.8433194, -3.4490551, -3.322890],
                [-6.5314736, -3.78566215, -4.895984],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.770304, 0.413294, 0.459321],
                [0.168981, 0.198763, 0.210507],
                [0.251349, 0.551761, 0.486412],
            ]
        )
    elif model_name == "dfine_m_coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.560220, -4.85852, -4.17702],
                [-4.316123, -5.05847, -3.67138],
                [-4.746527, -6.84849, -3.16082],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.254904, 0.546769, 0.475268],
                [0.765929, 0.412381, 0.465382],
                [0.168203, 0.199216, 0.212074],
            ]
        )
    elif model_name == "dfine_m_obj2coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.366883, -7.473241, -5.812667],
                [-4.159411, -7.463147, -5.5588631],
                [-4.689057, -5.662412, -4.8570761],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2582458, 0.546585, 0.474364],
                [0.7702161, 0.410875, 0.458311],
                [0.5499019, 0.275631, 0.059633],
            ]
        )
    elif model_name == "dfine_m_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-5.869976, -2.919317, -5.000304],
                [-6.024388, -3.356399, -4.868721],
                [-6.174082, -3.646999, -3.296291],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.255842, 0.551198, 0.476844],
                [0.7716257, 0.41158, 0.456440],
                [0.5497970, 0.27594, 0.058975],
            ]
        )
    elif model_name == "dfine_l_coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.163772, -5.08069, -4.358980],
                [-3.941459, -4.85244, -4.142285],
                [-4.292777, -6.10656, -4.974356],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2549637, 0.549357, 0.478588],
                [0.7679444, 0.415660, 0.460468],
                [0.1684390, 0.198810, 0.213304],
            ]
        )
    elif model_name == "dfine_l_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-5.794356, -3.31482, -5.403600],
                [-5.581522, -3.68510, -5.791995],
                [-6.050641, -3.18308, -4.332631],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.768685, 0.413633, 0.460345],
                [0.250496, 0.552994, 0.478527],
                [0.168896, 0.198750, 0.211770],
            ]
        )
    elif model_name == "dfine_l_obj2coco_e25":
        expected_slice_logits = torch.tensor(
            [
                [-3.456711, -6.703585, -5.255528],
                [-3.561902, -6.936790, -5.589331],
                [-4.282646, -5.987232, -4.452727],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.769001, 0.410186, 0.459154],
                [0.255007, 0.549077, 0.479700],
                [0.168469, 0.198881, 0.212796],
            ]
        )
    elif model_name == "dfine_s_coco":
        expected_slice_logits = torch.tensor(
            [
                [-3.319429, -4.463451, -5.628054],
                [-4.910080, -9.160397, -5.950480],
                [-4.046249, -4.022937, -6.300149],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.765489, 0.415563, 0.46872],
                [0.168150, 0.198066, 0.21442],
                [0.258652, 0.547807, 0.47930],
            ]
        )
    elif model_name == "dfine_s_obj2coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.881455, -6.95667, -4.667908],
                [-3.432105, -8.56579, -6.258423],
                [-4.239889, -8.90363, -5.805731],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.168693, 0.198404, 0.212517],
                [0.765760, 0.4120095, 0.464718],
                [0.256592, 0.5509163, 0.476652],
            ]
        )
    elif model_name == "dfine_s_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-6.464325, -3.859787, -6.328770],
                [-6.630722, -3.216558, -5.556852],
                [-5.627515, -3.938537, -3.713672],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2507714, 0.5541206, 0.480323],
                [0.7640793, 0.4124037, 0.470100],
                [0.1691108, 0.198352, 0.212930],
            ]
        )
    else:
        raise ValueError(f"Unknown d_fine_name: {model_name}")

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits.to(outputs.logits.device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes.to(outputs.pred_boxes.device), atol=1e-3)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Upload model, image processor and config to the hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        config.push_to_hub(
            repo_id=repo_id, commit_message="Add config from convert_d_fine_original_pytorch_checkpoint_to_hf.py"
        )
        model.push_to_hub(
            repo_id=repo_id, commit_message="Add model from convert_d_fine_original_pytorch_checkpoint_to_hf.py"
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_d_fine_original_pytorch_checkpoint_to_hf.py",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dfine_x_coco",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_d_fine_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
