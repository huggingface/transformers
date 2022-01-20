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
"""Convert PoolFormer checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import (
    PoolFormerConfig,
    PoolFormerFeatureExtractor,
    PoolFormerForImageClassification,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def rename_keys(state_dict, encoder_only=False):
    new_state_dict = OrderedDict()
    total_embed_found, patch_emb_offset = 0, 0
    for key, value in state_dict.items():
        if "layer_scale" in key:
            # Since we aren't training the network, we can skip layer_scale
            continue
        if encoder_only and not key.startswith("head"):
            key = "segformer.encoder." + key
        if key.startswith("network"):
            key = key.replace("network", "poolformer.encoder")
        if "proj" in key:
                # works for the first embedding as well as the internal embedding layers
                if key.endswith("bias") and "patch_embed" not in key:
                    # if it's 
                    patch_emb_offset += 1
                to_replace = key[:key.find("proj")]
                key = key.replace(to_replace, f"patch_embeddings.{total_embed_found}.")
                if key.endswith("bias"):
                    total_embed_found += 1
        if "mlp.fc1" in key:
            orig_block_num = int(key[:key.find("mlp")][-4])
            new_block_num = orig_block_num - patch_emb_offset
            layer_num = int(key[:key.find("mlp")][-2])
            key = key.replace(f"{orig_block_num}.{layer_num}.mlp.fc1", f"block.{new_block_num}.{layer_num}.output.conv1")

        if "mlp.fc2" in key:
            orig_block_num = int(key[:key.find("mlp")][-4])
            new_block_num = orig_block_num - patch_emb_offset
            layer_num = int(key[:key.find("mlp")][-2])
            key = key.replace(f"{orig_block_num}.{layer_num}.mlp.fc2", f"block.{new_block_num}.{layer_num}.output.conv2")
        if "norm1" in key:
            orig_block_num = int(key[:key.find("norm1")][-4])
            new_block_num = orig_block_num - patch_emb_offset
            layer_num = int(key[:key.find("norm1")][-2])
            key = key.replace(f"{orig_block_num}.{layer_num}.norm1", f"block.{new_block_num}.{layer_num}.output.before_norm")
        if "norm2" in key:
            orig_block_num = int(key[:key.find("norm2")][-4])
            new_block_num = orig_block_num - patch_emb_offset
            layer_num = int(key[:key.find("norm2")][-2])
            key = key.replace(f"{orig_block_num}.{layer_num}.norm2", f"block.{new_block_num}.{layer_num}.output.after_norm")
        if "head" in key:
            key = key.replace("head", "classifier")

        # Debug
        print(key)