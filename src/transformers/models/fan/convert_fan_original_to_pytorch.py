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
import functools
import json
import re
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

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


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def remap_patch_embed(k):
    return k.replace("patch_embed", "patch_embeddings")


def remap_embeddings(k):
    if "embed" in k:
        return f"fan.embeddings.{k}"
    return k


def remap_gamma(k):
    return k.replace("gamma", "weight")


def remap_head(k):
    if k.split(".")[0] in ("norm", "head"):
        return f"head.{k}"
    return k


def remap_encoder(k):
    if any(x in k for x in ["fan", "head"]):
        return k
    return f"fan.encoder.{k}"


def remap_blocks(k):
    pattern = "([a-z\.]*blocks\.\d*\.)"
    if re.match(pattern, k):
        return re.sub(pattern, "\\1block.", k)
    return k


def remap_segmentation_linear(k):
    if "decode_head.linear_fuse.conv" in k:
        return k.replace("decode_head.linear_fuse.conv", "decode_head.linear_fuse")
    if "decode_head.linear_fuse.bn" in k:
        return k.replace("decode_head.linear_fuse.bn", "decode_head.batch_norm")
    if "decode_head.linear_pred" in k:
        return k.replace("decode_head.linear_pred", "decode_head.classifier")
    return k


def remap_linear_fuse(k):
    for num in range(4):
        if f"decode_head.linear_c{num+1}" in k:
            return k.replace(f"decode_head.linear_c{num+1}", f"decode_head.linear_c.{num}")
    return k


remap_fn = compose(
    remap_segmentation_linear,
    remap_linear_fuse,
    remap_blocks,
    remap_encoder,
    remap_gamma,
    remap_head,
    remap_embeddings,
    remap_patch_embed,
)


def remap_state(state_dict):
    # remap_fn = compose(remap_embeddings, remap_gamma, remap_head, remap_encoder)
    return {remap_fn(k): v for k, v in state_dict.items()}
