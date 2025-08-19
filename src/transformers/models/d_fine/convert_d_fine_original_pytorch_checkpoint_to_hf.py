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
import re
from pathlib import Path
from typing import Optional

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_d_fine_config(model_name: str) -> DFineConfig:
    config = DFineConfig()

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "object365-id2label.json" if "obj365" in model_name else "coco-detection-mmdet-id2label.json"
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
        config.backbone_config.hidden_sizes = [256, 512, 1024, 2048]
        config.backbone_config.stage_in_channels = [64, 128, 512, 1024]
        config.backbone_config.stage_mid_channels = [64, 128, 256, 512]
        config.backbone_config.stage_out_channels = [128, 512, 1024, 2048]
        config.backbone_config.stage_num_blocks = [1, 2, 5, 2]
        config.backbone_config.stage_downsample = [False, True, True, True]
        config.backbone_config.stage_light_block = [False, False, True, True]
        config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
        config.backbone_config.stage_numb_of_layers = [6, 6, 6, 6]
        config.backbone_config.stem_channels = [3, 32, 64]
        config.encoder_in_channels = [512, 1024, 2048]
        config.encoder_hidden_dim = 384
        config.encoder_ffn_dim = 2048
        config.decoder_n_points = [3, 6, 3]
        config.decoder_in_channels = [384, 384, 384]
        if model_name == "dfine_x_obj365":
            config.num_labels = 366
    elif model_name in ["dfine_m_coco", "dfine_m_obj2coco", "dfine_m_obj365"]:
        config.backbone_config.hidden_sizes = [192, 384, 768, 1536]
        config.backbone_config.stem_channels = [3, 24, 32]
        config.backbone_config.stage_in_channels = [32, 96, 384, 768]
        config.backbone_config.stage_mid_channels = [32, 64, 128, 256]
        config.backbone_config.stage_out_channels = [96, 384, 768, 1536]
        config.backbone_config.stage_num_blocks = [1, 1, 3, 1]
        config.backbone_config.stage_downsample = [False, True, True, True]
        config.backbone_config.stage_light_block = [False, False, True, True]
        config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
        config.backbone_config.stage_numb_of_layers = [4, 4, 4, 4]
        config.decoder_layers = 4
        config.decoder_n_points = [3, 6, 3]
        config.encoder_in_channels = [384, 768, 1536]
        config.backbone_config.use_learnable_affine_block = True
        config.depth_mult = 0.67
        if model_name == "dfine_m_obj365":
            config.num_labels = 366
    elif model_name in ["dfine_l_coco", "dfine_l_obj2coco_e25", "dfine_l_obj365"]:
        config.backbone_config.hidden_sizes = [256, 512, 1024, 2048]
        config.backbone_config.stem_channels = [3, 32, 48]
        config.backbone_config.stage_in_channels = [48, 128, 512, 1024]
        config.backbone_config.stage_mid_channels = [48, 96, 192, 384]
        config.backbone_config.stage_out_channels = [128, 512, 1024, 2048]
        config.backbone_config.stage_num_blocks = [1, 1, 3, 1]
        config.backbone_config.stage_downsample = [False, True, True, True]
        config.backbone_config.stage_light_block = [False, False, True, True]
        config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
        config.backbone_config.stage_numb_of_layers = [6, 6, 6, 6]
        config.encoder_ffn_dim = 1024
        config.encoder_in_channels = [512, 1024, 2048]
        config.decoder_n_points = [3, 6, 3]
        if model_name == "dfine_l_obj365":
            config.num_labels = 366
    elif model_name in ["dfine_n_coco", "dfine_n_obj2coco_e25", "dfine_n_obj365"]:
        config.backbone_config.hidden_sizes = [128, 256, 512, 1024]
        config.backbone_config.stem_channels = [3, 16, 16]
        config.backbone_config.stage_in_channels = [16, 64, 256, 512]
        config.backbone_config.stage_mid_channels = [16, 32, 64, 128]
        config.backbone_config.stage_out_channels = [64, 256, 512, 1024]
        config.backbone_config.stage_num_blocks = [1, 1, 2, 1]
        config.backbone_config.stage_downsample = [False, True, True, True]
        config.backbone_config.stage_light_block = [False, False, True, True]
        config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
        config.backbone_config.stage_numb_of_layers = [3, 3, 3, 3]
        config.backbone_config.out_indices = [3, 4]
        config.backbone_config.use_learnable_affine_block = True
        config.num_feature_levels = 2
        config.encoder_ffn_dim = 512
        config.encode_proj_layers = [1]
        config.d_model = 128
        config.encoder_hidden_dim = 128
        config.decoder_ffn_dim = 512
        config.encoder_in_channels = [512, 1024]
        config.decoder_n_points = [6, 6]
        config.decoder_in_channels = [128, 128]
        config.feat_strides = [16, 32]
        config.depth_mult = 0.5
        config.decoder_layers = 3
        config.hidden_expansion = 0.34
        if model_name == "dfine_n_obj365":
            config.num_labels = 366
    else:
        config.backbone_config.hidden_sizes = [128, 256, 512, 1024]
        config.backbone_config.stem_channels = [3, 16, 16]
        config.backbone_config.stage_in_channels = [16, 64, 256, 512]
        config.backbone_config.stage_mid_channels = [16, 32, 64, 128]
        config.backbone_config.stage_out_channels = [64, 256, 512, 1024]
        config.backbone_config.stage_num_blocks = [1, 1, 2, 1]
        config.backbone_config.stage_downsample = [False, True, True, True]
        config.backbone_config.stage_light_block = [False, False, True, True]
        config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
        config.backbone_config.stage_numb_of_layers = [3, 3, 3, 3]
        config.decoder_layers = 3
        config.hidden_expansion = 0.5
        config.depth_mult = 0.34
        config.decoder_n_points = [3, 6, 3]
        config.encoder_in_channels = [256, 512, 1024]
        config.backbone_config.use_learnable_affine_block = True
        if model_name == "dfine_s_obj365":
            config.num_labels = 366

    return config


def load_original_state_dict(repo_id, model_name):
    directory_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}.pth")

    original_state_dict = {}
    model = torch.load(directory_path, map_location="cpu")["model"]
    for key in model:
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
    # Basic convolutions
    r"encoder.fpn_blocks.(\d+).cv1.conv.weight": r"model.encoder.fpn_blocks.\1.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv1.norm.\2",
    # CSP Rep1 path
    r"encoder.fpn_blocks.(\d+).cv2.0.conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.fpn_blocks.\1.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv2.norm.\2",
    # CSP Rep2 path
    r"encoder.fpn_blocks.(\d+).cv3.0.conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.fpn_blocks.\1.conv3.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv3.norm.\2",
    # Final conv
    r"encoder.fpn_blocks.(\d+).cv4.conv.weight": r"model.encoder.fpn_blocks.\1.conv4.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv4.norm.\2",
    # Bottlenecks for CSP Rep1
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    # Bottlenecks for CSP Rep2
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    # PAN blocks - complete structure
    # Basic convolutions
    r"encoder.pan_blocks.(\d+).cv1.conv.weight": r"model.encoder.pan_blocks.\1.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv1.norm.\2",
    # CSP Rep1 path
    r"encoder.pan_blocks.(\d+).cv2.0.conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.0.conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.pan_blocks.\1.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv2.norm.\2",
    # CSP Rep2 path
    r"encoder.pan_blocks.(\d+).cv3.0.conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.0.conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.pan_blocks.\1.conv3.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv3.norm.\2",
    # Final conv
    r"encoder.pan_blocks.(\d+).cv4.conv.weight": r"model.encoder.pan_blocks.\1.conv4.conv.weight",
    r"encoder.pan_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv4.norm.\2",
    # Bottlenecks for CSP Rep1
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    # Bottlenecks for CSP Rep2
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    # Downsample convolutions
    r"encoder.downsample_convs.(\d+).0.cv(\d+).conv.weight": r"model.encoder.downsample_convs.\1.conv\2.conv.weight",
    r"encoder.downsample_convs.(\d+).0.cv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.conv\2.norm.\3",
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


def convert_old_keys_to_new_keys(state_dict_keys: Optional[dict] = None):
    # Use the mapping to rename keys
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict_keys.keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict_keys[new_key] = state_dict_keys.pop(key)

    return state_dict_keys


def read_in_q_k_v(state_dict, config, model_name):
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
        if model_name in ["dfine_n_coco", "dfine_n_obj2coco_e25", "dfine_n_obj365"]:
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:128, :]
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:128]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[128:256, :]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[128:256]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-128:, :]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-128:]
        else:
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
    state_dict.pop("decoder.valid_mask", None)
    state_dict.pop("decoder.anchors", None)
    model = DFineForObjectDetection(config)
    logger.info(f"Converting model {model_name}...")

    state_dict = convert_old_keys_to_new_keys(state_dict)
    state_dict.pop("decoder.model.decoder.up", None)
    state_dict.pop("decoder.model.decoder.reg_scale", None)

    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config, model_name)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    for key in state_dict.copy():
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
                [-4.844723, -4.7293096, -4.5971327],
                [-4.554266, -4.61723, -4.627926],
                [-4.3934402, -4.6064143, -4.139952],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2565248, 0.5477609, 0.47644863],
                [0.7690029, 0.41423926, 0.46148556],
                [0.1688096, 0.19923759, 0.21118002],
            ]
        )
    elif model_name == "dfine_x_obj2coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.230433, -6.6295037, -4.8339615],
                [-4.085411, -6.3280816, -4.695468],
                [-3.8968022, -6.336813, -4.67051],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.25707328, 0.54842496, 0.47624254],
                [0.76967394, 0.41272867, 0.45970756],
                [0.16882066, 0.19918433, 0.2112098],
            ]
        )
    elif model_name == "dfine_x_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-6.3844957, -3.7549126, -4.6873264],
                [-5.8433194, -3.4490552, -3.3228905],
                [-6.5314736, -3.7856622, -4.895984],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7703046, 0.41329497, 0.45932162],
                [0.16898105, 0.19876392, 0.21050783],
                [0.25134972, 0.5517619, 0.4864124],
            ]
        )
    elif model_name == "dfine_m_coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.5187078, -4.71708, -4.117749],
                [-4.513984, -4.937715, -3.829125],
                [-4.830042, -6.931682, -3.1740026],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.25851426, 0.5489963, 0.4757598],
                [0.769683, 0.41411665, 0.45988125],
                [0.16866133, 0.19921188, 0.21207744],
            ]
        )
    elif model_name == "dfine_m_obj2coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.520666, -7.6678333, -5.739887],
                [-4.5053635, -7.510611, -5.452532],
                [-4.70348, -5.6098466, -5.0199957],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2567608, 0.5485795, 0.4767465],
                [0.77035284, 0.41236404, 0.4580645],
                [0.5498525, 0.27548885, 0.05886984],
            ]
        )
    elif model_name == "dfine_m_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-5.770525, -3.1610885, -5.2807794],
                [-5.7809954, -3.768266, -5.1146393],
                [-6.180705, -3.7357295, -3.1651964],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2529114, 0.5526663, 0.48270613],
                [0.7712474, 0.41294736, 0.457174],
                [0.5497157, 0.27588123, 0.05813372],
            ]
        )
    elif model_name == "dfine_l_coco":
        expected_slice_logits = torch.tensor(
            [
                [-4.068779, -5.169955, -4.339212],
                [-3.9461594, -5.0279613, -4.0161457],
                [-4.218292, -6.196324, -5.175245],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2564867, 0.5489948, 0.4748876],
                [0.7693534, 0.4138953, 0.4598034],
                [0.16875696, 0.19875404, 0.21196914],
            ]
        )
    elif model_name == "dfine_l_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-5.7953215, -3.4901116, -5.4394145],
                [-5.7032104, -3.671125, -5.76121],
                [-6.09466, -3.1512096, -4.285499],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7693825, 0.41265628, 0.4606362],
                [0.25306237, 0.55187637, 0.4832178],
                [0.16892478, 0.19880727, 0.21115331],
            ]
        )
    elif model_name == "dfine_l_obj2coco_e25":
        expected_slice_logits = torch.tensor(
            [
                [-3.6098495, -6.633563, -5.1227236],
                [-3.682696, -6.9178205, -5.414557],
                [-4.491674, -6.0823426, -4.5718226],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7697078, 0.41368833, 0.45879585],
                [0.2573691, 0.54856044, 0.47715297],
                [0.16895264, 0.19871138, 0.2115552],
            ]
        )
    elif model_name == "dfine_n_coco":
        expected_slice_logits = torch.tensor(
            [
                [-3.7827945, -5.0889463, -4.8341026],
                [-5.3046904, -6.2801714, -2.9276395],
                [-4.497901, -5.2670407, -6.2380104],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.73334837, 0.4270624, 0.39424777],
                [0.1680235, 0.1988639, 0.21031213],
                [0.25370035, 0.5534435, 0.48496848],
            ]
        )
    elif model_name == "dfine_s_coco":
        expected_slice_logits = torch.tensor(
            [
                [-3.8097816, -4.7724586, -5.994499],
                [-5.2974715, -9.499067, -6.1653666],
                [-5.3502765, -3.9530406, -6.3630295],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7677696, 0.41479152, 0.46441072],
                [0.16912134, 0.19869131, 0.2123824],
                [0.2581653, 0.54818195, 0.47512347],
            ]
        )
    elif model_name == "dfine_s_obj2coco":
        expected_slice_logits = torch.tensor(
            [
                [-6.0208125, -7.532673, -5.0572147],
                [-3.3595953, -9.057545, -6.376975],
                [-4.3203554, -9.546032, -6.075504],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16901012, 0.19883151, 0.21121952],
                [0.76784194, 0.41266578, 0.46402973],
                [00.2563128, 0.54797643, 0.47937632],
            ]
        )
    elif model_name == "dfine_s_obj365":
        expected_slice_logits = torch.tensor(
            [
                [-6.3807316, -4.320986, -6.4775343],
                [-6.5818424, -3.5009093, -5.75824],
                [-5.748005, -4.3228016, -4.003726],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2532072, 0.5491191, 0.48222217],
                [0.76586807, 0.41175705, 0.46789962],
                [0.169111, 0.19844547, 0.21069047],
            ]
        )
    else:
        raise ValueError(f"Unknown d_fine_name: {model_name}")

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits.to(outputs.logits.device), atol=1e-3)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes.to(outputs.pred_boxes.device), atol=1e-4)

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
            repo_id=repo_id,
            commit_message="Add config from convert_d_fine_original_pytorch_checkpoint_to_hf.py",
        )
        model.push_to_hub(
            repo_id=repo_id,
            commit_message="Add model from convert_d_fine_original_pytorch_checkpoint_to_hf.py",
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
