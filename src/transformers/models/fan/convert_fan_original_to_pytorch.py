# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert FAN checkpoints."""


import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import (
    FANConfig,
    SegformerFeatureExtractor,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def replace_linear_keys(key):
    match = re.match(r"linear_c\d{1}", key)
    if match:
        start, end = match.span()
        output = key[start : end - 1] + "." + str(int(key[end - 1]) - 1) + key[end:]
        return output
    return key


def fix_linear_fuse(key):
    match_conv = re.match(r"linear_fuse\.conv", key)
    match_bn = re.match(r"linear_fuse\.bn", key)
    match_clf = re.match(r"linear_pred", key)
    if match_conv:
        return key.replace("linear_fuse.conv", "linear_fuse")
    if match_bn:
        return key.replace("linear_fuse.bn", "batch_norm")
    if match_clf:
        return key.replace("linear_pred", "classifier")
    return key


def remap_state(state_dict):
    new_state_dict = {f"fan.encoder.{k}": v for k, v in state_dict.items() if k.split(".")[0] not in ("norm", "head")}
    new_state_dict.update({f"head.{k}": v for k, v in state_dict.items() if k.split(".")[0] in ("norm", "head")})
    new_state_dict = {k.replace("gamma", "weight"): v for k, v in new_state_dict.items()}
    return new_state_dict
