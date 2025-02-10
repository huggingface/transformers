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

import re
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


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
   # Decoder base mappings
    r"decoder.valid_mask": r"model.decoder.valid_mask",
    r"decoder.anchors": r"model.decoder.anchors",
    r"decoder.up": r"model.decoder.up",
    r"decoder.reg_scale": r"model.decoder.reg_scale",

    # Backbone stem mappings - including stem2a and stem2b
    r"backbone.stem.stem1.conv.weight": r"model.backbone.model.embedder.stem1.convolution.weight",
    r"backbone.stem.stem2a.conv.weight": r"model.backbone.model.embedder.stem2a.convolution.weight",
    r"backbone.stem.stem2b.conv.weight": r"model.backbone.model.embedder.stem2b.convolution.weight",
    r"backbone.stem.stem3.conv.weight": r"model.backbone.model.embedder.stem3.convolution.weight",
    r"backbone.stem.stem4.conv.weight": r"model.backbone.model.embedder.stem4.convolution.weight",
    
    # Stem normalization
    r"backbone.stem.stem1.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem1.normalization.\1",
    r"backbone.stem.stem2a.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem2a.normalization.\1",
    r"backbone.stem.stem2b.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem2b.normalization.\1",
    r"backbone.stem.stem3.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem3.normalization.\1",
    r"backbone.stem.stem4.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem4.normalization.\1",
    
    # Stem lab parameters - fixed with .lab in the path
    r"backbone.stem.stem1.lab.(scale|bias)": r"model.backbone.model.embedder.stem1.lab.\1",
    r"backbone.stem.stem2a.lab.(scale|bias)": r"model.backbone.model.embedder.stem2a.lab.\1",
    r"backbone.stem.stem2b.lab.(scale|bias)": r"model.backbone.model.embedder.stem2b.lab.\1",
    r"backbone.stem.stem3.lab.(scale|bias)": r"model.backbone.model.embedder.stem3.lab.\1",
    r"backbone.stem.stem4.lab.(scale|bias)": r"model.backbone.model.embedder.stem4.lab.\1",

    # Backbone stages mappings
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv1.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv2.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv1.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv2.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.normalization.\4",

    # Backbone stages aggregation
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.0.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.1.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.0.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.normalization.\3",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.1.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.normalization.\3",
    
    # Backbone stages lab parameters for aggregation
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.0.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.lab.\3",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.1.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.lab.\3",

    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.lab.\4",
    
    # Conv1/Conv2 layers with lab
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv1.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.lab.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv2.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.lab.\4",
    
    # Downsample with lab
    r"backbone.stages.(\d+).downsample.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.downsample.lab.\2",

    # Backbone downsample
    r"backbone.stages.(\d+).downsample.conv.weight": r"model.backbone.model.encoder.stages.\1.downsample.convolution.weight",
    r"backbone.stages.(\d+).downsample.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.downsample.normalization.\2",

    # Encoder mappings
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.(weight|bias)": r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.\2",
    r"encoder.encoder.(\d+).layers.0.linear1.(weight|bias)": r"model.encoder.encoder.\1.layers.0.fc1.\2",
    r"encoder.encoder.(\d+).layers.0.linear2.(weight|bias)": r"model.encoder.encoder.\1.layers.0.fc2.\2",
    r"encoder.encoder.(\d+).layers.0.norm1.(weight|bias)": r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.\2",
    r"encoder.encoder.(\d+).layers.0.norm2.(weight|bias)": r"model.encoder.encoder.\1.layers.0.final_layer_norm.\2",

    # Encoder projections and convolutions
    r"encoder.input_proj.(\d+).conv.weight": r"model.encoder_input_proj.\1.0.weight",
    r"encoder.input_proj.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder_input_proj.\1.1.\2",
    r"encoder.lateral_convs.(\d+).conv.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder.lateral_convs.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.lateral_convs.\1.norm.\2",

    # FPN blocks - complete structure
    # Basic cv1-cv4 convolutions
    r"encoder.fpn_blocks.(\d+).cv1.conv.weight": r"model.encoder.fpn_blocks.\1.cv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.conv.weight": r"model.encoder.fpn_blocks.\1.cv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.conv.weight": r"model.encoder.fpn_blocks.\1.cv3.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv3.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv4.conv.weight": r"model.encoder.fpn_blocks.\1.cv4.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv4.norm.\2",

    # FPN cv2/cv3 special structure
    # cv2 structure
    r"encoder.fpn_blocks.(\d+).cv2.0.conv1.conv.weight": r"model.encoder.fpn_blocks.\1.cv2.0.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv2.0.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv2.conv.weight": r"model.encoder.fpn_blocks.\1.cv2.0.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv2.0.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.fpn_blocks.\1.cv2.1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv2.1.norm.\2",

    # cv3 structure
    r"encoder.fpn_blocks.(\d+).cv3.0.conv1.conv.weight": r"model.encoder.fpn_blocks.\1.cv3.0.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv3.0.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv2.conv.weight": r"model.encoder.fpn_blocks.\1.cv3.0.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv3.0.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.fpn_blocks.\1.cv3.1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv3.1.norm.\2",

    # FPN bottlenecks for cv2
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.fpn_blocks.\1.cv2.0.bottlenecks.\2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv2.0.bottlenecks.\2.conv1.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.fpn_blocks.\1.cv2.0.bottlenecks.\2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv2.0.bottlenecks.\2.conv2.norm.\3",

    # FPN bottlenecks for cv3
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.fpn_blocks.\1.cv3.0.bottlenecks.\2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv3.0.bottlenecks.\2.conv1.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.fpn_blocks.\1.cv3.0.bottlenecks.\2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.cv3.0.bottlenecks.\2.conv2.norm.\3",

    # PAN blocks - complete structure
    # Basic cv1-cv4 convolutions
    r"encoder.pan_blocks.(\d+).cv1.conv.weight": r"model.encoder.pan_blocks.\1.cv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.conv.weight": r"model.encoder.pan_blocks.\1.cv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.conv.weight": r"model.encoder.pan_blocks.\1.cv3.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv3.norm.\2",
    r"encoder.pan_blocks.(\d+).cv4.conv.weight": r"model.encoder.pan_blocks.\1.cv4.conv.weight",
    r"encoder.pan_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv4.norm.\2",

    # PAN cv2/cv3 special structure
    # cv2 structure
    r"encoder.pan_blocks.(\d+).cv2.0.conv1.conv.weight": r"model.encoder.pan_blocks.\1.cv2.0.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv2.0.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.0.conv2.conv.weight": r"model.encoder.pan_blocks.\1.cv2.0.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv2.0.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.pan_blocks.\1.cv2.1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv2.1.norm.\2",

    # cv3 structure
    r"encoder.pan_blocks.(\d+).cv3.0.conv1.conv.weight": r"model.encoder.pan_blocks.\1.cv3.0.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv3.0.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.0.conv2.conv.weight": r"model.encoder.pan_blocks.\1.cv3.0.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv3.0.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.pan_blocks.\1.cv3.1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv3.1.norm.\2",

    # PAN bottlenecks for cv2
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.pan_blocks.\1.cv2.0.bottlenecks.\2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv2.0.bottlenecks.\2.conv1.norm.\3",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.pan_blocks.\1.cv2.0.bottlenecks.\2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv2.0.bottlenecks.\2.conv2.norm.\3",

    # PAN bottlenecks for cv3
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.pan_blocks.\1.cv3.0.bottlenecks.\2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv3.0.bottlenecks.\2.conv1.norm.\3",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.pan_blocks.\1.cv3.0.bottlenecks.\2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.cv3.0.bottlenecks.\2.conv2.norm.\3",

    # Downsample convolutions
    r"encoder.downsample_convs.(\d+).0.cv(\d+).conv.weight": r"model.encoder.downsample_convs.\1.0.cv\2.conv.weight",
    r"encoder.downsample_convs.(\d+).0.cv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.0.cv\2.norm.\3",

    # Decoder layers
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.(weight|bias)": r"model.decoder.layers.\1.self_attn.out_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.attention_weights.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.value_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.output_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.num_points_scale": r"model.decoder.layers.\1.encoder_attn.num_points_scale",
    r"decoder.decoder.layers.(\d+).gateway.gate.(weight|bias)": r"model.decoder.layers.\1.gateway.gate.\2",
    r"decoder.decoder.layers.(\d+).gateway.norm.(weight|bias)": r"model.decoder.layers.\1.gateway.norm.\2",
    r"decoder.decoder.layers.(\d+).norm1.(weight|bias)": r"model.decoder.layers.\1.self_attn_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).norm2.(weight|bias)": r"model.decoder.layers.\1.encoder_attn_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).norm3.(weight|bias)": r"model.decoder.layers.\1.final_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).linear1.(weight|bias)": r"model.decoder.layers.\1.fc1.\2",
    r"decoder.decoder.layers.(\d+).linear2.(weight|bias)": r"model.decoder.layers.\1.fc2.\2",

    # LQE layers
    r"decoder.decoder.lqe_layers.(\d+).reg_conf.layers.(\d+).(weight|bias)": r"model.decoder.lqe_layers.\1.reg_conf.layers.\2.\3",

    # Decoder heads and projections
    r"decoder.dec_score_head.(\d+).(weight|bias)": r"model.decoder.class_embed.\1.\2",
    r"decoder.dec_bbox_head.(\d+).layers.(\d+).(weight|bias)": r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder.pre_bbox_head.layers.(\d+).(weight|bias)": r"model.decoder.pre_bbox_head.layers.\1.\2",
    r"decoder.input_proj.(\d+).conv.weight": r"model.decoder_input_proj.\1.0.weight",
    r"decoder.input_proj.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.decoder_input_proj.\1.1.\2",

    # Other decoder components
    r"decoder.denoising_class_embed.weight": r"model.denoising_class_embed.weight",
    r"decoder.query_pos_head.layers.(\d+).(weight|bias)": r"model.decoder.query_pos_head.layers.\1.\2",
    r"decoder.enc_output.proj.(weight|bias)": r"model.enc_output.0.\1",
    r"decoder.enc_output.norm.(weight|bias)": r"model.enc_output.1.\1",
    r"decoder.enc_score_head.(weight|bias)": r"model.enc_score_head.\1",
    r"decoder.enc_bbox_head.layers.(\d+).(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    # Use the mapping to rename keys
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict_keys.keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict_keys[new_key] = state_dict_keys.pop(key)

    return state_dict_keys


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
    state_dict.pop("decoder.decoder.up", None)
    state_dict.pop("decoder.decoder.reg_scale", None)
    state_dict = convert_old_keys_to_new_keys(state_dict)

    if not config.anchor_image_size:
        state_dict.pop("model.decoder.valid_mask", None)
        state_dict.pop("model.decoder.anchors", None)
    
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
        default="dfine_s_coco",
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
