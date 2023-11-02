# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert RT Detr checkpoints from the original repository: https://github.com/lyuwenyu/RT-DETR/issues/42"""

import argparse
from pathlib import Path
import json
import requests
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from transformers import RTDetrConfig, RTDetrImageProcessor, RTDetrModel

# TODO: (Rafael) Make this dictionary more efficient/shorter. Too many repetitive parts.
replaces = {
        "backbone.conv1.conv1_1.conv.weight": "model.backbone._backbone.conv1.0.weight",
        "backbone.conv1.conv1_1.norm.weight": "model.backbone._backbone.conv1.1.weight",
        "backbone.conv1.conv1_1.norm.bias": "model.backbone._backbone.conv1.1.bias",
        "backbone.conv1.conv1_1.norm.running_mean": "model.backbone._backbone.conv1.1.running_mean",
        "backbone.conv1.conv1_1.norm.running_var": "model.backbone._backbone.conv1.1.running_var",
        "backbone.conv1.conv1_2.conv.weight": "model.backbone._backbone.conv1.3.weight",
        "backbone.conv1.conv1_2.norm.weight": "model.backbone._backbone.conv1.4.weight",
        "backbone.conv1.conv1_2.norm.bias": "model.backbone._backbone.conv1.4.bias",
        "backbone.conv1.conv1_2.norm.running_mean": "model.backbone._backbone.conv1.4.running_mean",
        "backbone.conv1.conv1_2.norm.running_var": "model.backbone._backbone.conv1.4.running_var",
        "backbone.conv1.conv1_3.conv.weight": "model.backbone._backbone.conv1.6.weight",
        "backbone.conv1.conv1_3.norm.weight": "model.backbone._backbone.bn1.weight",
        "backbone.conv1.conv1_3.norm.bias": "model.backbone._backbone.bn1.bias",
        "backbone.conv1.conv1_3.norm.running_mean": "model.backbone._backbone.bn1.running_mean",
        "backbone.conv1.conv1_3.norm.running_var": "model.backbone._backbone.bn1.running_var",
        "backbone.res_layers.0.blocks.0.branch2a.conv.weight": "model.backbone._backbone.layer1.0.conv1.weight",
        "backbone.res_layers.0.blocks.0.branch2a.norm.weight": "model.backbone._backbone.layer1.0.bn1.weight",
        "backbone.res_layers.0.blocks.0.branch2a.norm.bias": "model.backbone._backbone.layer1.0.bn1.bias",
        "backbone.res_layers.0.blocks.0.branch2a.norm.running_mean": "model.backbone._backbone.layer1.0.bn1.running_mean",
        "backbone.res_layers.0.blocks.0.branch2a.norm.running_var": "model.backbone._backbone.layer1.0.bn1.running_var",
        "backbone.res_layers.0.blocks.0.branch2b.conv.weight": "model.backbone._backbone.layer1.0.conv2.weight",
        "backbone.res_layers.0.blocks.0.branch2b.norm.weight": "model.backbone._backbone.layer1.0.bn2.weight",
        "backbone.res_layers.0.blocks.0.branch2b.norm.bias": "model.backbone._backbone.layer1.0.bn2.bias",
        "backbone.res_layers.0.blocks.0.branch2b.norm.running_mean": "model.backbone._backbone.layer1.0.bn2.running_mean",
        "backbone.res_layers.0.blocks.0.branch2b.norm.running_var": "model.backbone._backbone.layer1.0.bn2.running_var",
        "backbone.res_layers.0.blocks.0.branch2c.conv.weight": "model.backbone._backbone.layer1.0.conv3.weight",
        "backbone.res_layers.0.blocks.0.branch2c.norm.weight": "model.backbone._backbone.layer1.0.bn3.weight",
        "backbone.res_layers.0.blocks.0.branch2c.norm.bias": "model.backbone._backbone.layer1.0.bn3.bias",
        "backbone.res_layers.0.blocks.0.branch2c.norm.running_mean": "model.backbone._backbone.layer1.0.bn3.running_mean",
        "backbone.res_layers.0.blocks.0.branch2c.norm.running_var": "model.backbone._backbone.layer1.0.bn3.running_var",
        "backbone.res_layers.0.blocks.0.short.conv.weight": "model.backbone._backbone.layer1.0.downsample.1.weight",
        "backbone.res_layers.0.blocks.0.short.norm.weight": "model.backbone._backbone.layer1.0.downsample.2.weight",
        "backbone.res_layers.0.blocks.0.short.norm.bias": "model.backbone._backbone.layer1.0.downsample.2.bias",
        "backbone.res_layers.0.blocks.0.short.norm.running_mean": "model.backbone._backbone.layer1.0.downsample.2.running_mean",
        "backbone.res_layers.0.blocks.0.short.norm.running_var": "model.backbone._backbone.layer1.0.downsample.2.running_var",
        "backbone.res_layers.0.blocks.1.branch2a.conv.weight": "model.backbone._backbone.layer1.1.conv1.weight",
        "backbone.res_layers.0.blocks.1.branch2a.norm.weight": "model.backbone._backbone.layer1.1.bn1.weight",
        "backbone.res_layers.0.blocks.1.branch2a.norm.bias": "model.backbone._backbone.layer1.1.bn1.bias",
        "backbone.res_layers.0.blocks.1.branch2a.norm.running_mean": "model.backbone._backbone.layer1.1.bn1.running_mean",
        "backbone.res_layers.0.blocks.1.branch2a.norm.running_var": "model.backbone._backbone.layer1.1.bn1.running_var",
        "backbone.res_layers.0.blocks.1.branch2b.conv.weight": "model.backbone._backbone.layer1.1.conv2.weight",
        "backbone.res_layers.0.blocks.1.branch2b.norm.weight": "model.backbone._backbone.layer1.1.bn2.weight",
        "backbone.res_layers.0.blocks.1.branch2b.norm.bias": "model.backbone._backbone.layer1.1.bn2.bias",
        "backbone.res_layers.0.blocks.1.branch2b.norm.running_mean": "model.backbone._backbone.layer1.1.bn2.running_mean",
        "backbone.res_layers.0.blocks.1.branch2b.norm.running_var": "model.backbone._backbone.layer1.1.bn2.running_var",
        "backbone.res_layers.0.blocks.1.branch2c.conv.weight": "model.backbone._backbone.layer1.1.conv3.weight",
        "backbone.res_layers.0.blocks.1.branch2c.norm.weight": "model.backbone._backbone.layer1.1.bn3.weight",
        "backbone.res_layers.0.blocks.1.branch2c.norm.bias": "model.backbone._backbone.layer1.1.bn3.bias",
        "backbone.res_layers.0.blocks.1.branch2c.norm.running_mean": "model.backbone._backbone.layer1.1.bn3.running_mean",
        "backbone.res_layers.0.blocks.1.branch2c.norm.running_var": "model.backbone._backbone.layer1.1.bn3.running_var",
        "backbone.res_layers.0.blocks.2.branch2a.conv.weight": "model.backbone._backbone.layer1.2.conv1.weight",
        "backbone.res_layers.0.blocks.2.branch2a.norm.weight": "model.backbone._backbone.layer1.2.bn1.weight",
        "backbone.res_layers.0.blocks.2.branch2a.norm.bias": "model.backbone._backbone.layer1.2.bn1.bias",
        "backbone.res_layers.0.blocks.2.branch2a.norm.running_mean": "model.backbone._backbone.layer1.2.bn1.running_mean",
        "backbone.res_layers.0.blocks.2.branch2a.norm.running_var": "model.backbone._backbone.layer1.2.bn1.running_var",
        "backbone.res_layers.0.blocks.2.branch2b.conv.weight": "model.backbone._backbone.layer1.2.conv2.weight",
        "backbone.res_layers.0.blocks.2.branch2b.norm.weight": "model.backbone._backbone.layer1.2.bn2.weight",
        "backbone.res_layers.0.blocks.2.branch2b.norm.bias": "model.backbone._backbone.layer1.2.bn2.bias",
        "backbone.res_layers.0.blocks.2.branch2b.norm.running_mean": "model.backbone._backbone.layer1.2.bn2.running_mean",
        "backbone.res_layers.0.blocks.2.branch2b.norm.running_var": "model.backbone._backbone.layer1.2.bn2.running_var",
        "backbone.res_layers.0.blocks.2.branch2c.conv.weight": "model.backbone._backbone.layer1.2.conv3.weight",
        "backbone.res_layers.0.blocks.2.branch2c.norm.weight": "model.backbone._backbone.layer1.2.bn3.weight",
        "backbone.res_layers.0.blocks.2.branch2c.norm.bias": "model.backbone._backbone.layer1.2.bn3.bias",
        "backbone.res_layers.0.blocks.2.branch2c.norm.running_mean": "model.backbone._backbone.layer1.2.bn3.running_mean",
        "backbone.res_layers.0.blocks.2.branch2c.norm.running_var": "model.backbone._backbone.layer1.2.bn3.running_var",
        "backbone.res_layers.1.blocks.0.branch2a.conv.weight": "model.backbone._backbone.layer2.0.conv1.weight",
        "backbone.res_layers.1.blocks.0.branch2a.norm.weight": "model.backbone._backbone.layer2.0.bn1.weight",
        "backbone.res_layers.1.blocks.0.branch2a.norm.bias": "model.backbone._backbone.layer2.0.bn1.bias",
        "backbone.res_layers.1.blocks.0.branch2a.norm.running_mean": "model.backbone._backbone.layer2.0.bn1.running_mean",
        "backbone.res_layers.1.blocks.0.branch2a.norm.running_var": "model.backbone._backbone.layer2.0.bn1.running_var",
        "backbone.res_layers.1.blocks.0.branch2b.conv.weight": "model.backbone._backbone.layer2.0.conv2.weight",
        "backbone.res_layers.1.blocks.0.branch2b.norm.weight": "model.backbone._backbone.layer2.0.bn2.weight",
        "backbone.res_layers.1.blocks.0.branch2b.norm.bias": "model.backbone._backbone.layer2.0.bn2.bias",
        "backbone.res_layers.1.blocks.0.branch2b.norm.running_mean": "model.backbone._backbone.layer2.0.bn2.running_mean",
        "backbone.res_layers.1.blocks.0.branch2b.norm.running_var": "model.backbone._backbone.layer2.0.bn2.running_var",
        "backbone.res_layers.1.blocks.0.branch2c.conv.weight": "model.backbone._backbone.layer2.0.conv3.weight",
        "backbone.res_layers.1.blocks.0.branch2c.norm.weight": "model.backbone._backbone.layer2.0.bn3.weight",
        "backbone.res_layers.1.blocks.0.branch2c.norm.bias": "model.backbone._backbone.layer2.0.bn3.bias",
        "backbone.res_layers.1.blocks.0.branch2c.norm.running_mean": "model.backbone._backbone.layer2.0.bn3.running_mean",
        "backbone.res_layers.1.blocks.0.branch2c.norm.running_var": "model.backbone._backbone.layer2.0.bn3.running_var",
        "backbone.res_layers.1.blocks.0.short.conv.conv.weight": "model.backbone._backbone.layer2.0.downsample.1.weight",
        "backbone.res_layers.1.blocks.0.short.conv.norm.weight": "model.backbone._backbone.layer2.0.downsample.2.weight",
        "backbone.res_layers.1.blocks.0.short.conv.norm.bias": "model.backbone._backbone.layer2.0.downsample.2.bias",
        "backbone.res_layers.1.blocks.0.short.conv.norm.running_mean": "model.backbone._backbone.layer2.0.downsample.2.running_mean",
        "backbone.res_layers.1.blocks.0.short.conv.norm.running_var": "model.backbone._backbone.layer2.0.downsample.2.running_var",
        "backbone.res_layers.1.blocks.1.branch2a.conv.weight": "model.backbone._backbone.layer2.1.conv1.weight",
        "backbone.res_layers.1.blocks.1.branch2a.norm.weight": "model.backbone._backbone.layer2.1.bn1.weight",
        "backbone.res_layers.1.blocks.1.branch2a.norm.bias": "model.backbone._backbone.layer2.1.bn1.bias",
        "backbone.res_layers.1.blocks.1.branch2a.norm.running_mean": "model.backbone._backbone.layer2.1.bn1.running_mean",
        "backbone.res_layers.1.blocks.1.branch2a.norm.running_var": "model.backbone._backbone.layer2.1.bn1.running_var",
        "backbone.res_layers.1.blocks.1.branch2b.conv.weight": "model.backbone._backbone.layer2.1.conv2.weight",
        "backbone.res_layers.1.blocks.1.branch2b.norm.weight": "model.backbone._backbone.layer2.1.bn2.weight",
        "backbone.res_layers.1.blocks.1.branch2b.norm.bias": "model.backbone._backbone.layer2.1.bn2.bias",
        "backbone.res_layers.1.blocks.1.branch2b.norm.running_mean": "model.backbone._backbone.layer2.1.bn2.running_mean",
        "backbone.res_layers.1.blocks.1.branch2b.norm.running_var": "model.backbone._backbone.layer2.1.bn2.running_var",
        "backbone.res_layers.1.blocks.1.branch2c.conv.weight": "model.backbone._backbone.layer2.1.conv3.weight",
        "backbone.res_layers.1.blocks.1.branch2c.norm.weight": "model.backbone._backbone.layer2.1.bn3.weight",
        "backbone.res_layers.1.blocks.1.branch2c.norm.bias": "model.backbone._backbone.layer2.1.bn3.bias",
        "backbone.res_layers.1.blocks.1.branch2c.norm.running_mean": "model.backbone._backbone.layer2.1.bn3.running_mean",
        "backbone.res_layers.1.blocks.1.branch2c.norm.running_var": "model.backbone._backbone.layer2.1.bn3.running_var",
        "backbone.res_layers.1.blocks.2.branch2a.conv.weight": "model.backbone._backbone.layer2.2.conv1.weight",
        "backbone.res_layers.1.blocks.2.branch2a.norm.weight": "model.backbone._backbone.layer2.2.bn1.weight",
        "backbone.res_layers.1.blocks.2.branch2a.norm.bias": "model.backbone._backbone.layer2.2.bn1.bias",
        "backbone.res_layers.1.blocks.2.branch2a.norm.running_mean": "model.backbone._backbone.layer2.2.bn1.running_mean",
        "backbone.res_layers.1.blocks.2.branch2a.norm.running_var": "model.backbone._backbone.layer2.2.bn1.running_var",
        "backbone.res_layers.1.blocks.2.branch2b.conv.weight": "model.backbone._backbone.layer2.2.conv2.weight",
        "backbone.res_layers.1.blocks.2.branch2b.norm.weight": "model.backbone._backbone.layer2.2.bn2.weight",
        "backbone.res_layers.1.blocks.2.branch2b.norm.bias": "model.backbone._backbone.layer2.2.bn2.bias",
        "backbone.res_layers.1.blocks.2.branch2b.norm.running_mean": "model.backbone._backbone.layer2.2.bn2.running_mean",
        "backbone.res_layers.1.blocks.2.branch2b.norm.running_var": "model.backbone._backbone.layer2.2.bn2.running_var",
        "backbone.res_layers.1.blocks.2.branch2c.conv.weight": "model.backbone._backbone.layer2.2.conv3.weight",
        "backbone.res_layers.1.blocks.2.branch2c.norm.weight": "model.backbone._backbone.layer2.2.bn3.weight",
        "backbone.res_layers.1.blocks.2.branch2c.norm.bias": "model.backbone._backbone.layer2.2.bn3.bias",
        "backbone.res_layers.1.blocks.2.branch2c.norm.running_mean": "model.backbone._backbone.layer2.2.bn3.running_mean",
        "backbone.res_layers.1.blocks.2.branch2c.norm.running_var": "model.backbone._backbone.layer2.2.bn3.running_var",
        "backbone.res_layers.1.blocks.3.branch2a.conv.weight": "model.backbone._backbone.layer2.3.conv1.weight",
        "backbone.res_layers.1.blocks.3.branch2a.norm.weight": "model.backbone._backbone.layer2.3.bn1.weight",
        "backbone.res_layers.1.blocks.3.branch2a.norm.bias": "model.backbone._backbone.layer2.3.bn1.bias",
        "backbone.res_layers.1.blocks.3.branch2a.norm.running_mean": "model.backbone._backbone.layer2.3.bn1.running_mean",
        "backbone.res_layers.1.blocks.3.branch2a.norm.running_var": "model.backbone._backbone.layer2.3.bn1.running_var",
        "backbone.res_layers.1.blocks.3.branch2b.conv.weight": "model.backbone._backbone.layer2.3.conv2.weight",
        "backbone.res_layers.1.blocks.3.branch2b.norm.weight": "model.backbone._backbone.layer2.3.bn2.weight",
        "backbone.res_layers.1.blocks.3.branch2b.norm.bias": "model.backbone._backbone.layer2.3.bn2.bias",
        "backbone.res_layers.1.blocks.3.branch2b.norm.running_mean": "model.backbone._backbone.layer2.3.bn2.running_mean",
        "backbone.res_layers.1.blocks.3.branch2b.norm.running_var": "model.backbone._backbone.layer2.3.bn2.running_var",
        "backbone.res_layers.1.blocks.3.branch2c.conv.weight": "model.backbone._backbone.layer2.3.conv3.weight",
        "backbone.res_layers.1.blocks.3.branch2c.norm.weight": "model.backbone._backbone.layer2.3.bn3.weight",
        "backbone.res_layers.1.blocks.3.branch2c.norm.bias": "model.backbone._backbone.layer2.3.bn3.bias",
        "backbone.res_layers.1.blocks.3.branch2c.norm.running_mean": "model.backbone._backbone.layer2.3.bn3.running_mean",
        "backbone.res_layers.1.blocks.3.branch2c.norm.running_var": "model.backbone._backbone.layer2.3.bn3.running_var",
        "backbone.res_layers.2.blocks.0.branch2a.conv.weight": "model.backbone._backbone.layer3.0.conv1.weight",
        "backbone.res_layers.2.blocks.0.branch2a.norm.weight": "model.backbone._backbone.layer3.0.bn1.weight",
        "backbone.res_layers.2.blocks.0.branch2a.norm.bias": "model.backbone._backbone.layer3.0.bn1.bias",
        "backbone.res_layers.2.blocks.0.branch2a.norm.running_mean": "model.backbone._backbone.layer3.0.bn1.running_mean",
        "backbone.res_layers.2.blocks.0.branch2a.norm.running_var": "model.backbone._backbone.layer3.0.bn1.running_var",
        "backbone.res_layers.2.blocks.0.branch2b.conv.weight": "model.backbone._backbone.layer3.0.conv2.weight",
        "backbone.res_layers.2.blocks.0.branch2b.norm.weight": "model.backbone._backbone.layer3.0.bn2.weight",
        "backbone.res_layers.2.blocks.0.branch2b.norm.bias": "model.backbone._backbone.layer3.0.bn2.bias",
        "backbone.res_layers.2.blocks.0.branch2b.norm.running_mean": "model.backbone._backbone.layer3.0.bn2.running_mean",
        "backbone.res_layers.2.blocks.0.branch2b.norm.running_var": "model.backbone._backbone.layer3.0.bn2.running_var",
        "backbone.res_layers.2.blocks.0.branch2c.conv.weight": "model.backbone._backbone.layer3.0.conv3.weight",
        "backbone.res_layers.2.blocks.0.branch2c.norm.weight": "model.backbone._backbone.layer3.0.bn3.weight",
        "backbone.res_layers.2.blocks.0.branch2c.norm.bias": "model.backbone._backbone.layer3.0.bn3.bias",
        "backbone.res_layers.2.blocks.0.branch2c.norm.running_mean": "model.backbone._backbone.layer3.0.bn3.running_mean",
        "backbone.res_layers.2.blocks.0.branch2c.norm.running_var": "model.backbone._backbone.layer3.0.bn3.running_var",
        "backbone.res_layers.2.blocks.0.short.conv.conv.weight": "model.backbone._backbone.layer3.0.downsample.1.weight",
        "backbone.res_layers.2.blocks.0.short.conv.norm.weight": "model.backbone._backbone.layer3.0.downsample.2.weight",
        "backbone.res_layers.2.blocks.0.short.conv.norm.bias": "model.backbone._backbone.layer3.0.downsample.2.bias",
        "backbone.res_layers.2.blocks.0.short.conv.norm.running_mean": "model.backbone._backbone.layer3.0.downsample.2.running_mean",
        "backbone.res_layers.2.blocks.0.short.conv.norm.running_var": "model.backbone._backbone.layer3.0.downsample.2.running_var",
        "backbone.res_layers.2.blocks.1.branch2a.conv.weight": "model.backbone._backbone.layer3.1.conv1.weight",
        "backbone.res_layers.2.blocks.1.branch2a.norm.weight": "model.backbone._backbone.layer3.1.bn1.weight",
        "backbone.res_layers.2.blocks.1.branch2a.norm.bias": "model.backbone._backbone.layer3.1.bn1.bias",
        "backbone.res_layers.2.blocks.1.branch2a.norm.running_mean": "model.backbone._backbone.layer3.1.bn1.running_mean",
        "backbone.res_layers.2.blocks.1.branch2a.norm.running_var": "model.backbone._backbone.layer3.1.bn1.running_var",
        "backbone.res_layers.2.blocks.1.branch2b.conv.weight": "model.backbone._backbone.layer3.1.conv2.weight",
        "backbone.res_layers.2.blocks.1.branch2b.norm.weight": "model.backbone._backbone.layer3.1.bn2.weight",
        "backbone.res_layers.2.blocks.1.branch2b.norm.bias": "model.backbone._backbone.layer3.1.bn2.bias",
        "backbone.res_layers.2.blocks.1.branch2b.norm.running_mean": "model.backbone._backbone.layer3.1.bn2.running_mean",
        "backbone.res_layers.2.blocks.1.branch2b.norm.running_var": "model.backbone._backbone.layer3.1.bn2.running_var",
        "backbone.res_layers.2.blocks.1.branch2c.conv.weight": "model.backbone._backbone.layer3.1.conv3.weight",
        "backbone.res_layers.2.blocks.1.branch2c.norm.weight": "model.backbone._backbone.layer3.1.bn3.weight",
        "backbone.res_layers.2.blocks.1.branch2c.norm.bias": "model.backbone._backbone.layer3.1.bn3.bias",
        "backbone.res_layers.2.blocks.1.branch2c.norm.running_mean": "model.backbone._backbone.layer3.1.bn3.running_mean",
        "backbone.res_layers.2.blocks.1.branch2c.norm.running_var": "model.backbone._backbone.layer3.1.bn3.running_var",
        "backbone.res_layers.2.blocks.2.branch2a.conv.weight": "model.backbone._backbone.layer3.2.conv1.weight",
        "backbone.res_layers.2.blocks.2.branch2a.norm.weight": "model.backbone._backbone.layer3.2.bn1.weight",
        "backbone.res_layers.2.blocks.2.branch2a.norm.bias": "model.backbone._backbone.layer3.2.bn1.bias",
        "backbone.res_layers.2.blocks.2.branch2a.norm.running_mean": "model.backbone._backbone.layer3.2.bn1.running_mean",
        "backbone.res_layers.2.blocks.2.branch2a.norm.running_var": "model.backbone._backbone.layer3.2.bn1.running_var",
        "backbone.res_layers.2.blocks.2.branch2b.conv.weight": "model.backbone._backbone.layer3.2.conv2.weight",
        "backbone.res_layers.2.blocks.2.branch2b.norm.weight": "model.backbone._backbone.layer3.2.bn2.weight",
        "backbone.res_layers.2.blocks.2.branch2b.norm.bias": "model.backbone._backbone.layer3.2.bn2.bias",
        "backbone.res_layers.2.blocks.2.branch2b.norm.running_mean": "model.backbone._backbone.layer3.2.bn2.running_mean",
        "backbone.res_layers.2.blocks.2.branch2b.norm.running_var": "model.backbone._backbone.layer3.2.bn2.running_var",
        "backbone.res_layers.2.blocks.2.branch2c.conv.weight": "model.backbone._backbone.layer3.2.conv3.weight",
        "backbone.res_layers.2.blocks.2.branch2c.norm.weight": "model.backbone._backbone.layer3.2.bn3.weight",
        "backbone.res_layers.2.blocks.2.branch2c.norm.bias": "model.backbone._backbone.layer3.2.bn3.bias",
        "backbone.res_layers.2.blocks.2.branch2c.norm.running_mean": "model.backbone._backbone.layer3.2.bn3.running_mean",
        "backbone.res_layers.2.blocks.2.branch2c.norm.running_var": "model.backbone._backbone.layer3.2.bn3.running_var",
        "backbone.res_layers.2.blocks.3.branch2a.conv.weight": "model.backbone._backbone.layer3.3.conv1.weight",
        "backbone.res_layers.2.blocks.3.branch2a.norm.weight": "model.backbone._backbone.layer3.3.bn1.weight",
        "backbone.res_layers.2.blocks.3.branch2a.norm.bias": "model.backbone._backbone.layer3.3.bn1.bias",
        "backbone.res_layers.2.blocks.3.branch2a.norm.running_mean": "model.backbone._backbone.layer3.3.bn1.running_mean",
        "backbone.res_layers.2.blocks.3.branch2a.norm.running_var": "model.backbone._backbone.layer3.3.bn1.running_var",
        "backbone.res_layers.2.blocks.3.branch2b.conv.weight": "model.backbone._backbone.layer3.3.conv2.weight",
        "backbone.res_layers.2.blocks.3.branch2b.norm.weight": "model.backbone._backbone.layer3.3.bn2.weight",
        "backbone.res_layers.2.blocks.3.branch2b.norm.bias": "model.backbone._backbone.layer3.3.bn2.bias",
        "backbone.res_layers.2.blocks.3.branch2b.norm.running_mean": "model.backbone._backbone.layer3.3.bn2.running_mean",
        "backbone.res_layers.2.blocks.3.branch2b.norm.running_var": "model.backbone._backbone.layer3.3.bn2.running_var",
        "backbone.res_layers.2.blocks.3.branch2c.conv.weight": "model.backbone._backbone.layer3.3.conv3.weight",
        "backbone.res_layers.2.blocks.3.branch2c.norm.weight": "model.backbone._backbone.layer3.3.bn3.weight",
        "backbone.res_layers.2.blocks.3.branch2c.norm.bias": "model.backbone._backbone.layer3.3.bn3.bias",
        "backbone.res_layers.2.blocks.3.branch2c.norm.running_mean": "model.backbone._backbone.layer3.3.bn3.running_mean",
        "backbone.res_layers.2.blocks.3.branch2c.norm.running_var": "model.backbone._backbone.layer3.3.bn3.running_var",
        "backbone.res_layers.2.blocks.4.branch2a.conv.weight": "model.backbone._backbone.layer3.4.conv1.weight",
        "backbone.res_layers.2.blocks.4.branch2a.norm.weight": "model.backbone._backbone.layer3.4.bn1.weight",
        "backbone.res_layers.2.blocks.4.branch2a.norm.bias": "model.backbone._backbone.layer3.4.bn1.bias",
        "backbone.res_layers.2.blocks.4.branch2a.norm.running_mean": "model.backbone._backbone.layer3.4.bn1.running_mean",
        "backbone.res_layers.2.blocks.4.branch2a.norm.running_var": "model.backbone._backbone.layer3.4.bn1.running_var",
        "backbone.res_layers.2.blocks.4.branch2b.conv.weight": "model.backbone._backbone.layer3.4.conv2.weight",
        "backbone.res_layers.2.blocks.4.branch2b.norm.weight": "model.backbone._backbone.layer3.4.bn2.weight",
        "backbone.res_layers.2.blocks.4.branch2b.norm.bias": "model.backbone._backbone.layer3.4.bn2.bias",
        "backbone.res_layers.2.blocks.4.branch2b.norm.running_mean": "model.backbone._backbone.layer3.4.bn2.running_mean",
        "backbone.res_layers.2.blocks.4.branch2b.norm.running_var": "model.backbone._backbone.layer3.4.bn2.running_var",
        "backbone.res_layers.2.blocks.4.branch2c.conv.weight": "model.backbone._backbone.layer3.4.conv3.weight",
        "backbone.res_layers.2.blocks.4.branch2c.norm.weight": "model.backbone._backbone.layer3.4.bn3.weight",
        "backbone.res_layers.2.blocks.4.branch2c.norm.bias": "model.backbone._backbone.layer3.4.bn3.bias",
        "backbone.res_layers.2.blocks.4.branch2c.norm.running_mean": "model.backbone._backbone.layer3.4.bn3.running_mean",
        "backbone.res_layers.2.blocks.4.branch2c.norm.running_var": "model.backbone._backbone.layer3.4.bn3.running_var",
        "backbone.res_layers.2.blocks.5.branch2a.conv.weight": "model.backbone._backbone.layer3.5.conv1.weight",
        "backbone.res_layers.2.blocks.5.branch2a.norm.weight": "model.backbone._backbone.layer3.5.bn1.weight",
        "backbone.res_layers.2.blocks.5.branch2a.norm.bias": "model.backbone._backbone.layer3.5.bn1.bias",
        "backbone.res_layers.2.blocks.5.branch2a.norm.running_mean": "model.backbone._backbone.layer3.5.bn1.running_mean",
        "backbone.res_layers.2.blocks.5.branch2a.norm.running_var": "model.backbone._backbone.layer3.5.bn1.running_var",
        "backbone.res_layers.2.blocks.5.branch2b.conv.weight": "model.backbone._backbone.layer3.5.conv2.weight",
        "backbone.res_layers.2.blocks.5.branch2b.norm.weight": "model.backbone._backbone.layer3.5.bn2.weight",
        "backbone.res_layers.2.blocks.5.branch2b.norm.bias": "model.backbone._backbone.layer3.5.bn2.bias",
        "backbone.res_layers.2.blocks.5.branch2b.norm.running_mean": "model.backbone._backbone.layer3.5.bn2.running_mean",
        "backbone.res_layers.2.blocks.5.branch2b.norm.running_var": "model.backbone._backbone.layer3.5.bn2.running_var",
        "backbone.res_layers.2.blocks.5.branch2c.conv.weight": "model.backbone._backbone.layer3.5.conv3.weight",
        "backbone.res_layers.2.blocks.5.branch2c.norm.weight": "model.backbone._backbone.layer3.5.bn3.weight",
        "backbone.res_layers.2.blocks.5.branch2c.norm.bias": "model.backbone._backbone.layer3.5.bn3.bias",
        "backbone.res_layers.2.blocks.5.branch2c.norm.running_mean": "model.backbone._backbone.layer3.5.bn3.running_mean",
        "backbone.res_layers.2.blocks.5.branch2c.norm.running_var": "model.backbone._backbone.layer3.5.bn3.running_var",
        "backbone.res_layers.3.blocks.0.branch2a.conv.weight": "model.backbone._backbone.layer4.0.conv1.weight",
        "backbone.res_layers.3.blocks.0.branch2a.norm.weight": "model.backbone._backbone.layer4.0.bn1.weight",
        "backbone.res_layers.3.blocks.0.branch2a.norm.bias": "model.backbone._backbone.layer4.0.bn1.bias",
        "backbone.res_layers.3.blocks.0.branch2a.norm.running_mean": "model.backbone._backbone.layer4.0.bn1.running_mean",
        "backbone.res_layers.3.blocks.0.branch2a.norm.running_var": "model.backbone._backbone.layer4.0.bn1.running_var",
        "backbone.res_layers.3.blocks.0.branch2b.conv.weight": "model.backbone._backbone.layer4.0.conv2.weight",
        "backbone.res_layers.3.blocks.0.branch2b.norm.weight": "model.backbone._backbone.layer4.0.bn2.weight",
        "backbone.res_layers.3.blocks.0.branch2b.norm.bias": "model.backbone._backbone.layer4.0.bn2.bias",
        "backbone.res_layers.3.blocks.0.branch2b.norm.running_mean": "model.backbone._backbone.layer4.0.bn2.running_mean",
        "backbone.res_layers.3.blocks.0.branch2b.norm.running_var": "model.backbone._backbone.layer4.0.bn2.running_var",
        "backbone.res_layers.3.blocks.0.branch2c.conv.weight": "model.backbone._backbone.layer4.0.conv3.weight",
        "backbone.res_layers.3.blocks.0.branch2c.norm.weight": "model.backbone._backbone.layer4.0.bn3.weight",
        "backbone.res_layers.3.blocks.0.branch2c.norm.bias": "model.backbone._backbone.layer4.0.bn3.bias",
        "backbone.res_layers.3.blocks.0.branch2c.norm.running_mean": "model.backbone._backbone.layer4.0.bn3.running_mean",
        "backbone.res_layers.3.blocks.0.branch2c.norm.running_var": "model.backbone._backbone.layer4.0.bn3.running_var",
        "backbone.res_layers.3.blocks.0.short.conv.conv.weight": "model.backbone._backbone.layer4.0.downsample.1.weight",
        "backbone.res_layers.3.blocks.0.short.conv.norm.weight": "model.backbone._backbone.layer4.0.downsample.2.weight",
        "backbone.res_layers.3.blocks.0.short.conv.norm.bias": "model.backbone._backbone.layer4.0.downsample.2.bias",
        "backbone.res_layers.3.blocks.0.short.conv.norm.running_mean": "model.backbone._backbone.layer4.0.downsample.2.running_mean",
        "backbone.res_layers.3.blocks.0.short.conv.norm.running_var": "model.backbone._backbone.layer4.0.downsample.2.running_var",
        "backbone.res_layers.3.blocks.1.branch2a.conv.weight": "model.backbone._backbone.layer4.1.conv1.weight",
        "backbone.res_layers.3.blocks.1.branch2a.norm.weight": "model.backbone._backbone.layer4.1.bn1.weight",
        "backbone.res_layers.3.blocks.1.branch2a.norm.bias": "model.backbone._backbone.layer4.1.bn1.bias",
        "backbone.res_layers.3.blocks.1.branch2a.norm.running_mean": "model.backbone._backbone.layer4.1.bn1.running_mean",
        "backbone.res_layers.3.blocks.1.branch2a.norm.running_var": "model.backbone._backbone.layer4.1.bn1.running_var",
        "backbone.res_layers.3.blocks.1.branch2b.conv.weight": "model.backbone._backbone.layer4.1.conv2.weight",
        "backbone.res_layers.3.blocks.1.branch2b.norm.weight": "model.backbone._backbone.layer4.1.bn2.weight",
        "backbone.res_layers.3.blocks.1.branch2b.norm.bias": "model.backbone._backbone.layer4.1.bn2.bias",
        "backbone.res_layers.3.blocks.1.branch2b.norm.running_mean": "model.backbone._backbone.layer4.1.bn2.running_mean",
        "backbone.res_layers.3.blocks.1.branch2b.norm.running_var": "model.backbone._backbone.layer4.1.bn2.running_var",
        "backbone.res_layers.3.blocks.1.branch2c.conv.weight": "model.backbone._backbone.layer4.1.conv3.weight",
        "backbone.res_layers.3.blocks.1.branch2c.norm.weight": "model.backbone._backbone.layer4.1.bn3.weight",
        "backbone.res_layers.3.blocks.1.branch2c.norm.bias": "model.backbone._backbone.layer4.1.bn3.bias",
        "backbone.res_layers.3.blocks.1.branch2c.norm.running_mean": "model.backbone._backbone.layer4.1.bn3.running_mean",
        "backbone.res_layers.3.blocks.1.branch2c.norm.running_var": "model.backbone._backbone.layer4.1.bn3.running_var",
        "backbone.res_layers.3.blocks.2.branch2a.conv.weight": "model.backbone._backbone.layer4.2.conv1.weight",
        "backbone.res_layers.3.blocks.2.branch2a.norm.weight": "model.backbone._backbone.layer4.2.bn1.weight",
        "backbone.res_layers.3.blocks.2.branch2a.norm.bias": "model.backbone._backbone.layer4.2.bn1.bias",
        "backbone.res_layers.3.blocks.2.branch2a.norm.running_mean": "model.backbone._backbone.layer4.2.bn1.running_mean",
        "backbone.res_layers.3.blocks.2.branch2a.norm.running_var": "model.backbone._backbone.layer4.2.bn1.running_var",
        "backbone.res_layers.3.blocks.2.branch2b.conv.weight": "model.backbone._backbone.layer4.2.conv2.weight",
        "backbone.res_layers.3.blocks.2.branch2b.norm.weight": "model.backbone._backbone.layer4.2.bn2.weight",
        "backbone.res_layers.3.blocks.2.branch2b.norm.bias": "model.backbone._backbone.layer4.2.bn2.bias",
        "backbone.res_layers.3.blocks.2.branch2b.norm.running_mean": "model.backbone._backbone.layer4.2.bn2.running_mean",
        "backbone.res_layers.3.blocks.2.branch2b.norm.running_var": "model.backbone._backbone.layer4.2.bn2.running_var",
        "backbone.res_layers.3.blocks.2.branch2c.conv.weight": "model.backbone._backbone.layer4.2.conv3.weight",
        "backbone.res_layers.3.blocks.2.branch2c.norm.weight": "model.backbone._backbone.layer4.2.bn3.weight",
        "backbone.res_layers.3.blocks.2.branch2c.norm.bias": "model.backbone._backbone.layer4.2.bn3.bias",
        "backbone.res_layers.3.blocks.2.branch2c.norm.running_mean": "model.backbone._backbone.layer4.2.bn3.running_mean",
        "backbone.res_layers.3.blocks.2.branch2c.norm.running_var": "model.backbone._backbone.layer4.2.bn3.running_var",
    }

expected_logits = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.tensor(
        [-4.159348487854004, -4.703853607177734, -5.946484565734863, -5.562824249267578, -4.7707929611206055]
    ),
}

expected_logits_shape = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.Size([1, 300, 80]),
}

expected_boxes = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.tensor(
        [
            [0.1688060760498047, 0.19992263615131378, 0.21225441992282867, 0.09384090453386307],
            [0.768376350402832, 0.41226309537887573, 0.4636859893798828, 0.7233726978302002],
            [0.25953856110572815, 0.5483334064483643, 0.4777486026287079, 0.8709195256233215],
        ]
    )
}

expected_boxes_shape = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.Size([1, 300, 4]),
}

def get_sample_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(requests.get(url, stream=True).raw)

def update_config_values(config, checkpoint_name):
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # Real values for rtdetr_r50vd_6x_coco_from_paddle.pth
    if checkpoint_name == "rtdetr_r50vd_6x_coco_from_paddle.pth":
        config.eval_spatial_size = [640, 640]
        config.feat_channels = [256, 256, 256]
    else:
        raise ValueError(f"Checkpoint {checkpoint_name} is not valid")
    
def convert_rt_detr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, repo_id):
    config = RTDetrConfig()

    checkpoint_name = Path(checkpoint_url).name
    version = Path(checkpoint_url).parts[-2]

    if version != "v0.1":
        raise ValueError(f"Given checkpoint version ({version}) is not supported.")

    # Update config values based on the checkpoint
    update_config_values(config, checkpoint_name)

    # Load model with the updated config
    model = RTDetrModel(config)
    
    # Load checkpoints from url
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["ema"]["module"]
    # For RTDetrObjectDetection:
    # state_dict_for_object_detection = {f"model.{k}": v for k, v in state_dict.items()}
    # For RTDetrModel
    state_dict_for_object_detection = {f"{k}": v for k, v in state_dict.items()}
    
    # Apply mapping
    for old_key, new_key in replaces.items():
        new_val = state_dict_for_object_detection.pop(old_key)
        new_key = new_key.replace("model.","")
        state_dict_for_object_detection[new_key] = new_val

    # Transfer mapped weights
    model.load_state_dict(state_dict_for_object_detection)
    model.eval()
    
    # Prepare image
    img = get_sample_img()
    image_processor = RTDetrImageProcessor()
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    outputs = model(pixel_values)

    # Verify boxes
    output_boxes = outputs.pred_boxes
    assert (
        output_boxes.shape == expected_boxes_shape[checkpoint_name]
    ), f"Shapes of output boxes do not match {checkpoint_name} {version}"
    expected = expected_boxes[checkpoint_name].to(device)
    assert torch.allclose(
        output_boxes[0, :3, :], expected, atol=1e-5
    ), f"Output boxes do not match for {checkpoint_name} {version}"

    # Verify logits
    output_logits = outputs.logits.cpu()
    original_logits = torch.tensor(
        [
            [
                [-4.64763879776001, -5.001153945922852, -4.978509902954102],
                [-4.159348487854004, -4.703853607177734, -5.946484565734863],
                [-4.437461853027344, -4.65836238861084, -6.235235691070557],
            ]
        ]
    )
    assert torch.allclose(output_logits[0, :3, :3], original_logits[0, :3, :3], atol=1e-4)

    if push_to_hub:
        model.push_to_hub(repo_id=repo_id, commit_message="Add model")
        image_processor.push_to_hub(repo_id=repo_id, commit_message="Add model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
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

    convert_rt_detr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
