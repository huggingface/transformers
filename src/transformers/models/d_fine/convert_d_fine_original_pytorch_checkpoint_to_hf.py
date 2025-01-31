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

from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor, DFineResNetStageConfig
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

    if model_name == "dfine_x_coco":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.layer_type = "basic"
        config.backbone_config.embedding_size = 32
        config.backbone_config.stage_config = DFineResNetStageConfig(
            stage1=[64, 64, 128, 1, False, False, 3, 6],
            stage2=[128, 128, 512, 2, True, False, 3, 6],
            stage3=[512, 256, 1024, 5, True, True, 5, 6],
            stage4=[1024, 512, 2048, 2, True, True, 5, 6],
        )
        config.backbone_config.stem_channels = [3, 32, 64]
        config.encoder_in_channels = [512, 1024, 2048]
        config.encoder_hidden_dim = 384
        config.hidden_expansion = 1.0
        config.decoder_layers = 6
        config.encoder_ffn_dim = 2048
    else:
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.layer_type = "basic"
        config.backbone_config.embedding_size = 32
        config.encoder_in_channels = [512, 1024, 2048]
        config.hidden_expansion = 1.0
        config.decoder_layers = 6

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

    rename_keys.append((f"decoder.valid_mask", f"model.decoder.valid_mask"))
    rename_keys.append((f"decoder.anchors", f"model.decoder.anchors"))
    rename_keys.append((f"decoder.up", f"model.decoder.up"))
    rename_keys.append((f"decoder.reg_scale", f"model.decoder.reg_scale"))

    # stem
    # fmt: off
    last_key = ["weight", "bias", "running_mean", "running_var"]

    rename_keys.append((f"backbone.stem.stem1.conv.weight", f"model.backbone.model.embedder.stem1.convolution.weight"))
    rename_keys.append((f"backbone.stem.stem2a.conv.weight", f"model.backbone.model.embedder.stem2a.convolution.weight"))
    rename_keys.append((f"backbone.stem.stem2b.conv.weight", f"model.backbone.model.embedder.stem2b.convolution.weight"))
    rename_keys.append((f"backbone.stem.stem3.conv.weight", f"model.backbone.model.embedder.stem3.convolution.weight"))
    rename_keys.append((f"backbone.stem.stem4.conv.weight", f"model.backbone.model.embedder.stem4.convolution.weight"))
    for last in last_key:
        rename_keys.append((f"backbone.stem.stem1.bn.{last}", f"model.backbone.model.embedder.stem1.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem2a.bn.{last}", f"model.backbone.model.embedder.stem2a.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem2b.bn.{last}", f"model.backbone.model.embedder.stem2b.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem3.bn.{last}", f"model.backbone.model.embedder.stem3.normalization.{last}"))
        rename_keys.append((f"backbone.stem.stem4.bn.{last}", f"model.backbone.model.embedder.stem4.normalization.{last}"))

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
        (f"encoder.downsample_convs.0.0.cv1.conv.weight", f"model.encoder.downsample_convs.0.0.cv1.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.0.0.cv1.norm.{last}", f"model.encoder.downsample_convs.0.0.cv1.norm.{last}")
        )
    rename_keys.append(
        (f"encoder.downsample_convs.1.0.cv1.conv.weight", f"model.encoder.downsample_convs.1.0.cv1.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.1.0.cv1.norm.{last}", f"model.encoder.downsample_convs.1.0.cv1.norm.{last}")
        )
    rename_keys.append(
        (f"encoder.downsample_convs.0.0.cv2.conv.weight", f"model.encoder.downsample_convs.0.0.cv2.conv.weight")
    )
    for last in last_key:
        rename_keys.append(
            (f"encoder.downsample_convs.0.0.cv2.norm.{last}", f"model.encoder.downsample_convs.0.0.cv2.norm.{last}")
        )
    rename_keys.append(
        (f"encoder.downsample_convs.1.0.cv2.conv.weight", f"model.encoder.downsample_convs.1.0.cv2.conv.weight")
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
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
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
def convert_rt_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our RTDETR structure.
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
        state_dict.pop("model.decoder.valid_mask")
        state_dict.pop("model.decoder.anchors")

    # those parameters are comming from config
    # state_dict.pop("decoder.up")
    # state_dict.pop("decoder.reg_scale")
    state_dict.pop("decoder.decoder.up")
    state_dict.pop("decoder.decoder.reg_scale")

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

    # Pass image by the model
    outputs = model(pixel_values)

    if model_name == "dfine_x_coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.3364253, -6.465683, -3.6130402],
                [-4.083815, -6.4039373, -6.97881],
                [-4.192215, -7.3410473, -6.9027247],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16868353, 0.19833282, 0.21182671],
                [0.25559652, 0.55121744, 0.47988364],
                [0.7698693, 0.4124569, 0.46036878],
            ]
        )
    else:
        raise ValueError(f"Unknown rt_detr_name: {model_name}")

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
            repo_id=repo_id, commit_message="Add config from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        model.push_to_hub(
            repo_id=repo_id, commit_message="Add model from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py",
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
        default="Peterande/D-FINE",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_rt_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
