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
"""Convert MobileViTv2 checkpoints from the ml-cvnets library."""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileViTv2Config,
    MobileViTv2ImageProcessor,
    MobileViTv2ForImageClassification,
    MobileViTv2ForSemanticSegmentation,
)
from transformers.utils import logging

import collections
import yaml
import copy

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_orig_config_file(orig_cfg_file):
    print("Loading config file...")
    
    def flatten_yaml_as_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    config = argparse.Namespace()
    with open(orig_cfg_file, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                    setattr(config, k, v)
        except yaml.YAMLError as exc:
            logger.error(
                "Error while loading config file: {}. Error message: {}".format(
                    orig_cfg_file, str(exc)
                )
            )
    return (config)


def get_mobilevitv2_config(task_name, orig_cfg_file):
    config = MobileViTv2Config()

    # dataset
    if task_name.startswith('imagenet1k_'):
        config.num_labels = 1000
        if int(task_name.strip().split('_')[-1])==384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-1k-id2label.json"
    elif task_name.startswith('imagenet21k_to_1k_'):
        config.num_labels = 21000
        if int(task_name.strip().split('_')[-1])==384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-22k-id2label.json"
    elif task_name.startswith('coco_'):
        config.num_labels = 91
        config.image_size = 320
        filename = "coco-detection-id2label.json"
    elif task_name.startswith('ade20k_'):
        config.num_labels = 151
        config.image_size = 512
        filename = "ade20k-id2label.json"
    elif task_name.startswith('voc_'):
        config.num_labels = 21
        config.image_size = 512
        filename = "pascal-voc-id2label.json"
    
    
    # orig_config
    orig_config = load_orig_config_file(orig_cfg_file)
    assert getattr(orig_config, 'model.classification.name', -1)=='mobilevit_v2', "Invalid model"
    config.width_multiplier = getattr(orig_config, 'model.classification.mitv2.width_multiplier', 1.0)
    assert getattr(orig_config, 'model.classification.mitv2.attn_norm_layer', -1)=='layer_norm_2d', "Norm layers other than layer_norm_2d is not supported"
    config.hidden_act = getattr(orig_config, 'model.classification.activation.name', 'swish')
    config.conv_init = getattr(orig_config, 'model.layer.conv_init', 'kaiming_normal')
    config.conv_init_std_dev = getattr(orig_config, 'model.layer.conv_init_std_dev', 0.02)
    config.linear_init = getattr(orig_config, 'model.layer.linear_init', 'trunc_normal')
    config.linear_init_std_dev =getattr(orig_config, 'model.layer.linear_init_std_dev', 0.02)
    # config.image_size == getattr(orig_config,  'sampler.bs.crop_size_width', 256)
    
    # id2label
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    
    return config


new_keys_list = [
"{model_prefix}.conv_stem.convolution.weight",
"{model_prefix}.conv_stem.normalization.weight",
"{model_prefix}.conv_stem.normalization.bias",
"{model_prefix}.conv_stem.normalization.running_mean",
"{model_prefix}.conv_stem.normalization.running_var",
"{model_prefix}.conv_stem.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.0.layer.0.expand_1x1.convolution.weight",
"{model_prefix}.encoder.layer.0.layer.0.expand_1x1.normalization.weight",
"{model_prefix}.encoder.layer.0.layer.0.expand_1x1.normalization.bias",
"{model_prefix}.encoder.layer.0.layer.0.expand_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.0.layer.0.expand_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.0.layer.0.expand_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.0.layer.0.conv_3x3.convolution.weight",
"{model_prefix}.encoder.layer.0.layer.0.conv_3x3.normalization.weight",
"{model_prefix}.encoder.layer.0.layer.0.conv_3x3.normalization.bias",
"{model_prefix}.encoder.layer.0.layer.0.conv_3x3.normalization.running_mean",
"{model_prefix}.encoder.layer.0.layer.0.conv_3x3.normalization.running_var",
"{model_prefix}.encoder.layer.0.layer.0.conv_3x3.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.0.layer.0.reduce_1x1.convolution.weight",
"{model_prefix}.encoder.layer.0.layer.0.reduce_1x1.normalization.weight",
"{model_prefix}.encoder.layer.0.layer.0.reduce_1x1.normalization.bias",
"{model_prefix}.encoder.layer.0.layer.0.reduce_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.0.layer.0.reduce_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.0.layer.0.reduce_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.1.layer.0.expand_1x1.convolution.weight",
"{model_prefix}.encoder.layer.1.layer.0.expand_1x1.normalization.weight",
"{model_prefix}.encoder.layer.1.layer.0.expand_1x1.normalization.bias",
"{model_prefix}.encoder.layer.1.layer.0.expand_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.1.layer.0.expand_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.1.layer.0.expand_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.1.layer.0.conv_3x3.convolution.weight",
"{model_prefix}.encoder.layer.1.layer.0.conv_3x3.normalization.weight",
"{model_prefix}.encoder.layer.1.layer.0.conv_3x3.normalization.bias",
"{model_prefix}.encoder.layer.1.layer.0.conv_3x3.normalization.running_mean",
"{model_prefix}.encoder.layer.1.layer.0.conv_3x3.normalization.running_var",
"{model_prefix}.encoder.layer.1.layer.0.conv_3x3.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.1.layer.0.reduce_1x1.convolution.weight",
"{model_prefix}.encoder.layer.1.layer.0.reduce_1x1.normalization.weight",
"{model_prefix}.encoder.layer.1.layer.0.reduce_1x1.normalization.bias",
"{model_prefix}.encoder.layer.1.layer.0.reduce_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.1.layer.0.reduce_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.1.layer.0.reduce_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.1.layer.1.expand_1x1.convolution.weight",
"{model_prefix}.encoder.layer.1.layer.1.expand_1x1.normalization.weight",
"{model_prefix}.encoder.layer.1.layer.1.expand_1x1.normalization.bias",
"{model_prefix}.encoder.layer.1.layer.1.expand_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.1.layer.1.expand_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.1.layer.1.expand_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.1.layer.1.conv_3x3.convolution.weight",
"{model_prefix}.encoder.layer.1.layer.1.conv_3x3.normalization.weight",
"{model_prefix}.encoder.layer.1.layer.1.conv_3x3.normalization.bias",
"{model_prefix}.encoder.layer.1.layer.1.conv_3x3.normalization.running_mean",
"{model_prefix}.encoder.layer.1.layer.1.conv_3x3.normalization.running_var",
"{model_prefix}.encoder.layer.1.layer.1.conv_3x3.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.1.layer.1.reduce_1x1.convolution.weight",
"{model_prefix}.encoder.layer.1.layer.1.reduce_1x1.normalization.weight",
"{model_prefix}.encoder.layer.1.layer.1.reduce_1x1.normalization.bias",
"{model_prefix}.encoder.layer.1.layer.1.reduce_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.1.layer.1.reduce_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.1.layer.1.reduce_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.2.downsampling_layer.expand_1x1.convolution.weight",
"{model_prefix}.encoder.layer.2.downsampling_layer.expand_1x1.normalization.weight",
"{model_prefix}.encoder.layer.2.downsampling_layer.expand_1x1.normalization.bias",
"{model_prefix}.encoder.layer.2.downsampling_layer.expand_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.2.downsampling_layer.expand_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.2.downsampling_layer.expand_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.2.downsampling_layer.conv_3x3.convolution.weight",
"{model_prefix}.encoder.layer.2.downsampling_layer.conv_3x3.normalization.weight",
"{model_prefix}.encoder.layer.2.downsampling_layer.conv_3x3.normalization.bias",
"{model_prefix}.encoder.layer.2.downsampling_layer.conv_3x3.normalization.running_mean",
"{model_prefix}.encoder.layer.2.downsampling_layer.conv_3x3.normalization.running_var",
"{model_prefix}.encoder.layer.2.downsampling_layer.conv_3x3.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.2.downsampling_layer.reduce_1x1.convolution.weight",
"{model_prefix}.encoder.layer.2.downsampling_layer.reduce_1x1.normalization.weight",
"{model_prefix}.encoder.layer.2.downsampling_layer.reduce_1x1.normalization.bias",
"{model_prefix}.encoder.layer.2.downsampling_layer.reduce_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.2.downsampling_layer.reduce_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.2.downsampling_layer.reduce_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.2.conv_kxk.convolution.weight",
"{model_prefix}.encoder.layer.2.conv_kxk.normalization.weight",
"{model_prefix}.encoder.layer.2.conv_kxk.normalization.bias",
"{model_prefix}.encoder.layer.2.conv_kxk.normalization.running_mean",
"{model_prefix}.encoder.layer.2.conv_kxk.normalization.running_var",
"{model_prefix}.encoder.layer.2.conv_kxk.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.2.transformer.layer.0.layernorm_before.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.layernorm_before.bias",
"{model_prefix}.encoder.layer.2.conv_1x1.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.0.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.0.layernorm_after.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.layernorm_after.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.0.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.0.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.0.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.1.layernorm_before.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.1.layernorm_before.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.1.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.1.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.1.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.1.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.1.layernorm_after.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.1.layernorm_after.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.1.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.1.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.2.transformer.layer.1.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.2.transformer.layer.1.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.2.layernorm.weight",
"{model_prefix}.encoder.layer.2.layernorm.bias",
"{model_prefix}.encoder.layer.2.conv_projection.convolution.weight",
"{model_prefix}.encoder.layer.2.conv_projection.normalization.weight",
"{model_prefix}.encoder.layer.2.conv_projection.normalization.bias",
"{model_prefix}.encoder.layer.2.conv_projection.normalization.running_mean",
"{model_prefix}.encoder.layer.2.conv_projection.normalization.running_var",
"{model_prefix}.encoder.layer.2.conv_projection.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.3.downsampling_layer.expand_1x1.convolution.weight",
"{model_prefix}.encoder.layer.3.downsampling_layer.expand_1x1.normalization.weight",
"{model_prefix}.encoder.layer.3.downsampling_layer.expand_1x1.normalization.bias",
"{model_prefix}.encoder.layer.3.downsampling_layer.expand_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.3.downsampling_layer.expand_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.3.downsampling_layer.expand_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.3.downsampling_layer.conv_3x3.convolution.weight",
"{model_prefix}.encoder.layer.3.downsampling_layer.conv_3x3.normalization.weight",
"{model_prefix}.encoder.layer.3.downsampling_layer.conv_3x3.normalization.bias",
"{model_prefix}.encoder.layer.3.downsampling_layer.conv_3x3.normalization.running_var",
"{model_prefix}.encoder.layer.3.downsampling_layer.conv_3x3.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.3.downsampling_layer.reduce_1x1.convolution.weight",
"{model_prefix}.encoder.layer.3.downsampling_layer.reduce_1x1.normalization.weight",
"{model_prefix}.encoder.layer.3.downsampling_layer.reduce_1x1.normalization.bias",
"{model_prefix}.encoder.layer.3.downsampling_layer.reduce_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.3.downsampling_layer.reduce_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.3.downsampling_layer.reduce_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.3.downsampling_layer.conv_3x3.normalization.running_mean",
"{model_prefix}.encoder.layer.3.conv_kxk.convolution.weight",
"{model_prefix}.encoder.layer.3.conv_kxk.normalization.weight",
"{model_prefix}.encoder.layer.3.conv_kxk.normalization.bias",
"{model_prefix}.encoder.layer.3.conv_kxk.normalization.running_mean",
"{model_prefix}.encoder.layer.3.conv_kxk.normalization.running_var",
"{model_prefix}.encoder.layer.3.conv_kxk.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.3.conv_1x1.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.layernorm_before.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.layernorm_before.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.0.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.0.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.0.layernorm_after.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.layernorm_after.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.0.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.0.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.0.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.1.layernorm_before.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.1.layernorm_before.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.1.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.1.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.1.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.1.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.1.layernorm_after.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.1.layernorm_after.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.1.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.1.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.1.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.1.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.2.layernorm_before.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.2.layernorm_before.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.2.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.2.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.2.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.2.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.2.layernorm_after.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.2.layernorm_after.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.2.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.2.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.2.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.2.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.3.layernorm_before.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.3.layernorm_before.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.3.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.3.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.3.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.3.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.3.layernorm_after.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.3.layernorm_after.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.3.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.3.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.3.transformer.layer.3.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.3.transformer.layer.3.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.3.layernorm.weight",
"{model_prefix}.encoder.layer.3.layernorm.bias",
"{model_prefix}.encoder.layer.3.conv_projection.convolution.weight",
"{model_prefix}.encoder.layer.3.conv_projection.normalization.weight",
"{model_prefix}.encoder.layer.3.conv_projection.normalization.bias",
"{model_prefix}.encoder.layer.3.conv_projection.normalization.running_mean",
"{model_prefix}.encoder.layer.3.conv_projection.normalization.running_var",
"{model_prefix}.encoder.layer.3.conv_projection.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.4.downsampling_layer.expand_1x1.convolution.weight",
"{model_prefix}.encoder.layer.4.downsampling_layer.expand_1x1.normalization.weight",
"{model_prefix}.encoder.layer.4.downsampling_layer.expand_1x1.normalization.bias",
"{model_prefix}.encoder.layer.4.downsampling_layer.expand_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.4.downsampling_layer.expand_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.4.downsampling_layer.expand_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.4.downsampling_layer.conv_3x3.convolution.weight",
"{model_prefix}.encoder.layer.4.downsampling_layer.conv_3x3.normalization.weight",
"{model_prefix}.encoder.layer.4.downsampling_layer.conv_3x3.normalization.bias",
"{model_prefix}.encoder.layer.4.downsampling_layer.conv_3x3.normalization.running_mean",
"{model_prefix}.encoder.layer.4.downsampling_layer.conv_3x3.normalization.running_var",
"{model_prefix}.encoder.layer.4.downsampling_layer.conv_3x3.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.4.downsampling_layer.reduce_1x1.convolution.weight",
"{model_prefix}.encoder.layer.4.downsampling_layer.reduce_1x1.normalization.weight",
"{model_prefix}.encoder.layer.4.downsampling_layer.reduce_1x1.normalization.bias",
"{model_prefix}.encoder.layer.4.downsampling_layer.reduce_1x1.normalization.running_mean",
"{model_prefix}.encoder.layer.4.downsampling_layer.reduce_1x1.normalization.running_var",
"{model_prefix}.encoder.layer.4.downsampling_layer.reduce_1x1.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.4.conv_kxk.convolution.weight",
"{model_prefix}.encoder.layer.4.conv_kxk.normalization.weight",
"{model_prefix}.encoder.layer.4.conv_kxk.normalization.bias",
"{model_prefix}.encoder.layer.4.conv_kxk.normalization.running_mean",
"{model_prefix}.encoder.layer.4.conv_kxk.normalization.running_var",
"{model_prefix}.encoder.layer.4.conv_kxk.normalization.num_batches_tracked",
"{model_prefix}.encoder.layer.4.conv_1x1.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.layernorm_before.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.layernorm_before.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.0.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.0.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.0.layernorm_after.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.layernorm_after.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.0.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.0.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.0.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.1.layernorm_before.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.1.layernorm_before.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.1.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.1.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.1.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.1.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.1.layernorm_after.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.1.layernorm_after.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.1.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.1.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.1.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.1.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.2.layernorm_before.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.2.layernorm_before.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.2.attention.qkv_proj.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.2.attention.qkv_proj.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.2.attention.out_proj.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.2.attention.out_proj.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.2.layernorm_after.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.2.layernorm_after.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.2.ffn.conv1.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.2.ffn.conv1.convolution.bias",
"{model_prefix}.encoder.layer.4.transformer.layer.2.ffn.conv2.convolution.weight",
"{model_prefix}.encoder.layer.4.transformer.layer.2.ffn.conv2.convolution.bias",
"{model_prefix}.encoder.layer.4.layernorm.weight",
"{model_prefix}.encoder.layer.4.layernorm.bias",
"{model_prefix}.encoder.layer.4.conv_projection.convolution.weight",
"{model_prefix}.encoder.layer.4.conv_projection.normalization.weight",
"{model_prefix}.encoder.layer.4.conv_projection.normalization.bias",
"{model_prefix}.encoder.layer.4.conv_projection.normalization.running_mean",
"{model_prefix}.encoder.layer.4.conv_projection.normalization.running_var",
"{model_prefix}.encoder.layer.4.conv_projection.normalization.num_batches_tracked",
"classifier.weight",
"classifier.bias",
]

orig_keys_list = [
"conv_1.block.conv.weight",
"conv_1.block.norm.weight",
"conv_1.block.norm.bias",
"conv_1.block.norm.running_mean",
"conv_1.block.norm.running_var",
"conv_1.block.norm.num_batches_tracked",
"layer_1.0.block.exp_1x1.block.conv.weight",
"layer_1.0.block.exp_1x1.block.norm.weight",
"layer_1.0.block.exp_1x1.block.norm.bias",
"layer_1.0.block.exp_1x1.block.norm.running_mean",
"layer_1.0.block.exp_1x1.block.norm.running_var",
"layer_1.0.block.exp_1x1.block.norm.num_batches_tracked",
"layer_1.0.block.conv_3x3.block.conv.weight",
"layer_1.0.block.conv_3x3.block.norm.weight",
"layer_1.0.block.conv_3x3.block.norm.bias",
"layer_1.0.block.conv_3x3.block.norm.running_mean",
"layer_1.0.block.conv_3x3.block.norm.running_var",
"layer_1.0.block.conv_3x3.block.norm.num_batches_tracked",
"layer_1.0.block.red_1x1.block.conv.weight",
"layer_1.0.block.red_1x1.block.norm.weight",
"layer_1.0.block.red_1x1.block.norm.bias",
"layer_1.0.block.red_1x1.block.norm.running_mean",
"layer_1.0.block.red_1x1.block.norm.running_var",
"layer_1.0.block.red_1x1.block.norm.num_batches_tracked",
"layer_2.0.block.exp_1x1.block.conv.weight",
"layer_2.0.block.exp_1x1.block.norm.weight",
"layer_2.0.block.exp_1x1.block.norm.bias",
"layer_2.0.block.exp_1x1.block.norm.running_mean",
"layer_2.0.block.exp_1x1.block.norm.running_var",
"layer_2.0.block.exp_1x1.block.norm.num_batches_tracked",
"layer_2.0.block.conv_3x3.block.conv.weight",
"layer_2.0.block.conv_3x3.block.norm.weight",
"layer_2.0.block.conv_3x3.block.norm.bias",
"layer_2.0.block.conv_3x3.block.norm.running_mean",
"layer_2.0.block.conv_3x3.block.norm.running_var",
"layer_2.0.block.conv_3x3.block.norm.num_batches_tracked",
"layer_2.0.block.red_1x1.block.conv.weight",
"layer_2.0.block.red_1x1.block.norm.weight",
"layer_2.0.block.red_1x1.block.norm.bias",
"layer_2.0.block.red_1x1.block.norm.running_mean",
"layer_2.0.block.red_1x1.block.norm.running_var",
"layer_2.0.block.red_1x1.block.norm.num_batches_tracked",
"layer_2.1.block.exp_1x1.block.conv.weight",
"layer_2.1.block.exp_1x1.block.norm.weight",
"layer_2.1.block.exp_1x1.block.norm.bias",
"layer_2.1.block.exp_1x1.block.norm.running_mean",
"layer_2.1.block.exp_1x1.block.norm.running_var",
"layer_2.1.block.exp_1x1.block.norm.num_batches_tracked",
"layer_2.1.block.conv_3x3.block.conv.weight",
"layer_2.1.block.conv_3x3.block.norm.weight",
"layer_2.1.block.conv_3x3.block.norm.bias",
"layer_2.1.block.conv_3x3.block.norm.running_mean",
"layer_2.1.block.conv_3x3.block.norm.running_var",
"layer_2.1.block.conv_3x3.block.norm.num_batches_tracked",
"layer_2.1.block.red_1x1.block.conv.weight",
"layer_2.1.block.red_1x1.block.norm.weight",
"layer_2.1.block.red_1x1.block.norm.bias",
"layer_2.1.block.red_1x1.block.norm.running_mean",
"layer_2.1.block.red_1x1.block.norm.running_var",
"layer_2.1.block.red_1x1.block.norm.num_batches_tracked",
"layer_3.0.block.exp_1x1.block.conv.weight",
"layer_3.0.block.exp_1x1.block.norm.weight",
"layer_3.0.block.exp_1x1.block.norm.bias",
"layer_3.0.block.exp_1x1.block.norm.running_mean",
"layer_3.0.block.exp_1x1.block.norm.running_var",
"layer_3.0.block.exp_1x1.block.norm.num_batches_tracked",
"layer_3.0.block.conv_3x3.block.conv.weight",
"layer_3.0.block.conv_3x3.block.norm.weight",
"layer_3.0.block.conv_3x3.block.norm.bias",
"layer_3.0.block.conv_3x3.block.norm.running_mean",
"layer_3.0.block.conv_3x3.block.norm.running_var",
"layer_3.0.block.conv_3x3.block.norm.num_batches_tracked",
"layer_3.0.block.red_1x1.block.conv.weight",
"layer_3.0.block.red_1x1.block.norm.weight",
"layer_3.0.block.red_1x1.block.norm.bias",
"layer_3.0.block.red_1x1.block.norm.running_mean",
"layer_3.0.block.red_1x1.block.norm.running_var",
"layer_3.0.block.red_1x1.block.norm.num_batches_tracked",
"layer_3.1.local_rep.0.block.conv.weight",
"layer_3.1.local_rep.0.block.norm.weight",
"layer_3.1.local_rep.0.block.norm.bias",
"layer_3.1.local_rep.0.block.norm.running_mean",
"layer_3.1.local_rep.0.block.norm.running_var",
"layer_3.1.local_rep.0.block.norm.num_batches_tracked",
"layer_3.1.local_rep.1.block.conv.weight",
"layer_3.1.global_rep.0.pre_norm_attn.0.weight",
"layer_3.1.global_rep.0.pre_norm_attn.0.bias",
"layer_3.1.global_rep.0.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_3.1.global_rep.0.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_3.1.global_rep.0.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_3.1.global_rep.0.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_3.1.global_rep.0.pre_norm_ffn.0.weight",
"layer_3.1.global_rep.0.pre_norm_ffn.0.bias",
"layer_3.1.global_rep.0.pre_norm_ffn.1.block.conv.weight",
"layer_3.1.global_rep.0.pre_norm_ffn.1.block.conv.bias",
"layer_3.1.global_rep.0.pre_norm_ffn.3.block.conv.weight",
"layer_3.1.global_rep.0.pre_norm_ffn.3.block.conv.bias",
"layer_3.1.global_rep.1.pre_norm_attn.0.weight",
"layer_3.1.global_rep.1.pre_norm_attn.0.bias",
"layer_3.1.global_rep.1.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_3.1.global_rep.1.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_3.1.global_rep.1.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_3.1.global_rep.1.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_3.1.global_rep.1.pre_norm_ffn.0.weight",
"layer_3.1.global_rep.1.pre_norm_ffn.0.bias",
"layer_3.1.global_rep.1.pre_norm_ffn.1.block.conv.weight",
"layer_3.1.global_rep.1.pre_norm_ffn.1.block.conv.bias",
"layer_3.1.global_rep.1.pre_norm_ffn.3.block.conv.weight",
"layer_3.1.global_rep.1.pre_norm_ffn.3.block.conv.bias",
"layer_3.1.global_rep.2.weight",
"layer_3.1.global_rep.2.bias",
"layer_3.1.conv_proj.block.conv.weight",
"layer_3.1.conv_proj.block.norm.weight",
"layer_3.1.conv_proj.block.norm.bias",
"layer_3.1.conv_proj.block.norm.running_mean",
"layer_3.1.conv_proj.block.norm.running_var",
"layer_3.1.conv_proj.block.norm.num_batches_tracked",
"layer_4.0.block.exp_1x1.block.conv.weight",
"layer_4.0.block.exp_1x1.block.norm.weight",
"layer_4.0.block.exp_1x1.block.norm.bias",
"layer_4.0.block.exp_1x1.block.norm.running_mean",
"layer_4.0.block.exp_1x1.block.norm.running_var",
"layer_4.0.block.exp_1x1.block.norm.num_batches_tracked",
"layer_4.0.block.conv_3x3.block.conv.weight",
"layer_4.0.block.conv_3x3.block.norm.weight",
"layer_4.0.block.conv_3x3.block.norm.bias",
"layer_4.0.block.conv_3x3.block.norm.running_mean",
"layer_4.0.block.conv_3x3.block.norm.running_var",
"layer_4.0.block.conv_3x3.block.norm.num_batches_tracked",
"layer_4.0.block.red_1x1.block.conv.weight",
"layer_4.0.block.red_1x1.block.norm.weight",
"layer_4.0.block.red_1x1.block.norm.bias",
"layer_4.0.block.red_1x1.block.norm.running_mean",
"layer_4.0.block.red_1x1.block.norm.running_var",
"layer_4.0.block.red_1x1.block.norm.num_batches_tracked",
"layer_4.1.local_rep.0.block.conv.weight",
"layer_4.1.local_rep.0.block.norm.weight",
"layer_4.1.local_rep.0.block.norm.bias",
"layer_4.1.local_rep.0.block.norm.running_mean",
"layer_4.1.local_rep.0.block.norm.running_var",
"layer_4.1.local_rep.0.block.norm.num_batches_tracked",
"layer_4.1.local_rep.1.block.conv.weight",
"layer_4.1.global_rep.0.pre_norm_attn.0.weight",
"layer_4.1.global_rep.0.pre_norm_attn.0.bias",
"layer_4.1.global_rep.0.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_4.1.global_rep.0.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_4.1.global_rep.0.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_4.1.global_rep.0.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_4.1.global_rep.0.pre_norm_ffn.0.weight",
"layer_4.1.global_rep.0.pre_norm_ffn.0.bias",
"layer_4.1.global_rep.0.pre_norm_ffn.1.block.conv.weight",
"layer_4.1.global_rep.0.pre_norm_ffn.1.block.conv.bias",
"layer_4.1.global_rep.0.pre_norm_ffn.3.block.conv.weight",
"layer_4.1.global_rep.0.pre_norm_ffn.3.block.conv.bias",
"layer_4.1.global_rep.1.pre_norm_attn.0.weight",
"layer_4.1.global_rep.1.pre_norm_attn.0.bias",
"layer_4.1.global_rep.1.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_4.1.global_rep.1.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_4.1.global_rep.1.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_4.1.global_rep.1.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_4.1.global_rep.1.pre_norm_ffn.0.weight",
"layer_4.1.global_rep.1.pre_norm_ffn.0.bias",
"layer_4.1.global_rep.1.pre_norm_ffn.1.block.conv.weight",
"layer_4.1.global_rep.1.pre_norm_ffn.1.block.conv.bias",
"layer_4.1.global_rep.1.pre_norm_ffn.3.block.conv.weight",
"layer_4.1.global_rep.1.pre_norm_ffn.3.block.conv.bias",
"layer_4.1.global_rep.2.pre_norm_attn.0.weight",
"layer_4.1.global_rep.2.pre_norm_attn.0.bias",
"layer_4.1.global_rep.2.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_4.1.global_rep.2.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_4.1.global_rep.2.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_4.1.global_rep.2.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_4.1.global_rep.2.pre_norm_ffn.0.weight",
"layer_4.1.global_rep.2.pre_norm_ffn.0.bias",
"layer_4.1.global_rep.2.pre_norm_ffn.1.block.conv.weight",
"layer_4.1.global_rep.2.pre_norm_ffn.1.block.conv.bias",
"layer_4.1.global_rep.2.pre_norm_ffn.3.block.conv.weight",
"layer_4.1.global_rep.2.pre_norm_ffn.3.block.conv.bias",
"layer_4.1.global_rep.3.pre_norm_attn.0.weight",
"layer_4.1.global_rep.3.pre_norm_attn.0.bias",
"layer_4.1.global_rep.3.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_4.1.global_rep.3.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_4.1.global_rep.3.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_4.1.global_rep.3.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_4.1.global_rep.3.pre_norm_ffn.0.weight",
"layer_4.1.global_rep.3.pre_norm_ffn.0.bias",
"layer_4.1.global_rep.3.pre_norm_ffn.1.block.conv.weight",
"layer_4.1.global_rep.3.pre_norm_ffn.1.block.conv.bias",
"layer_4.1.global_rep.3.pre_norm_ffn.3.block.conv.weight",
"layer_4.1.global_rep.3.pre_norm_ffn.3.block.conv.bias",
"layer_4.1.global_rep.4.weight",
"layer_4.1.global_rep.4.bias",
"layer_4.1.conv_proj.block.conv.weight",
"layer_4.1.conv_proj.block.norm.weight",
"layer_4.1.conv_proj.block.norm.bias",
"layer_4.1.conv_proj.block.norm.running_mean",
"layer_4.1.conv_proj.block.norm.running_var",
"layer_4.1.conv_proj.block.norm.num_batches_tracked",
"layer_5.0.block.exp_1x1.block.conv.weight",
"layer_5.0.block.exp_1x1.block.norm.weight",
"layer_5.0.block.exp_1x1.block.norm.bias",
"layer_5.0.block.exp_1x1.block.norm.running_mean",
"layer_5.0.block.exp_1x1.block.norm.running_var",
"layer_5.0.block.exp_1x1.block.norm.num_batches_tracked",
"layer_5.0.block.conv_3x3.block.conv.weight",
"layer_5.0.block.conv_3x3.block.norm.weight",
"layer_5.0.block.conv_3x3.block.norm.bias",
"layer_5.0.block.conv_3x3.block.norm.running_mean",
"layer_5.0.block.conv_3x3.block.norm.running_var",
"layer_5.0.block.conv_3x3.block.norm.num_batches_tracked",
"layer_5.0.block.red_1x1.block.conv.weight",
"layer_5.0.block.red_1x1.block.norm.weight",
"layer_5.0.block.red_1x1.block.norm.bias",
"layer_5.0.block.red_1x1.block.norm.running_mean",
"layer_5.0.block.red_1x1.block.norm.running_var",
"layer_5.0.block.red_1x1.block.norm.num_batches_tracked",
"layer_5.1.local_rep.0.block.conv.weight",
"layer_5.1.local_rep.0.block.norm.weight",
"layer_5.1.local_rep.0.block.norm.bias",
"layer_5.1.local_rep.0.block.norm.running_mean",
"layer_5.1.local_rep.0.block.norm.running_var",
"layer_5.1.local_rep.0.block.norm.num_batches_tracked",
"layer_5.1.local_rep.1.block.conv.weight",
"layer_5.1.global_rep.0.pre_norm_attn.0.weight",
"layer_5.1.global_rep.0.pre_norm_attn.0.bias",
"layer_5.1.global_rep.0.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_5.1.global_rep.0.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_5.1.global_rep.0.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_5.1.global_rep.0.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_5.1.global_rep.0.pre_norm_ffn.0.weight",
"layer_5.1.global_rep.0.pre_norm_ffn.0.bias",
"layer_5.1.global_rep.0.pre_norm_ffn.1.block.conv.weight",
"layer_5.1.global_rep.0.pre_norm_ffn.1.block.conv.bias",
"layer_5.1.global_rep.0.pre_norm_ffn.3.block.conv.weight",
"layer_5.1.global_rep.0.pre_norm_ffn.3.block.conv.bias",
"layer_5.1.global_rep.1.pre_norm_attn.0.weight",
"layer_5.1.global_rep.1.pre_norm_attn.0.bias",
"layer_5.1.global_rep.1.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_5.1.global_rep.1.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_5.1.global_rep.1.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_5.1.global_rep.1.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_5.1.global_rep.1.pre_norm_ffn.0.weight",
"layer_5.1.global_rep.1.pre_norm_ffn.0.bias",
"layer_5.1.global_rep.1.pre_norm_ffn.1.block.conv.weight",
"layer_5.1.global_rep.1.pre_norm_ffn.1.block.conv.bias",
"layer_5.1.global_rep.1.pre_norm_ffn.3.block.conv.weight",
"layer_5.1.global_rep.1.pre_norm_ffn.3.block.conv.bias",
"layer_5.1.global_rep.2.pre_norm_attn.0.weight",
"layer_5.1.global_rep.2.pre_norm_attn.0.bias",
"layer_5.1.global_rep.2.pre_norm_attn.1.qkv_proj.block.conv.weight",
"layer_5.1.global_rep.2.pre_norm_attn.1.qkv_proj.block.conv.bias",
"layer_5.1.global_rep.2.pre_norm_attn.1.out_proj.block.conv.weight",
"layer_5.1.global_rep.2.pre_norm_attn.1.out_proj.block.conv.bias",
"layer_5.1.global_rep.2.pre_norm_ffn.0.weight",
"layer_5.1.global_rep.2.pre_norm_ffn.0.bias",
"layer_5.1.global_rep.2.pre_norm_ffn.1.block.conv.weight",
"layer_5.1.global_rep.2.pre_norm_ffn.1.block.conv.bias",
"layer_5.1.global_rep.2.pre_norm_ffn.3.block.conv.weight",
"layer_5.1.global_rep.2.pre_norm_ffn.3.block.conv.bias",
"layer_5.1.global_rep.3.weight",
"layer_5.1.global_rep.3.bias",
"layer_5.1.conv_proj.block.conv.weight",
"layer_5.1.conv_proj.block.norm.weight",
"layer_5.1.conv_proj.block.norm.bias",
"layer_5.1.conv_proj.block.norm.running_mean",
"layer_5.1.conv_proj.block.norm.running_var",
"layer_5.1.conv_proj.block.norm.num_batches_tracked",
"classifier.1.weight",
"classifier.1.bias",
]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

def create_rename_keys(state_dict, base_model=False):
    
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevitv2."
    
    rename_keys = []
    for k in state_dict.keys():
        k_new = k
        
        if ".block." in k:
            k_new = k_new.replace(".block.", ".")
        if ".conv." in k:
            k_new = k_new.replace(".conv.", ".convolution.")
        if ".norm." in k:
            k_new = k_new.replace(".norm.", ".normalization.")
                        
        if "conv_1." in k:
            k_new = k_new.replace("conv_1.", f"{model_prefix}conv_stem.")
        for i in [1,2]:
            if f"layer_{i}." in k:
                k_new = k_new.replace(f"layer_{i}.", f"{model_prefix}encoder.layer.{i-1}.layer.")
        if ".exp_1x1." in k:
            k_new = k_new.replace(".exp_1x1.", ".expand_1x1.")
        if ".red_1x1." in k:
            k_new = k_new.replace(".red_1x1.", ".reduce_1x1.")
            
        for i in [3,4,5]:
            if f"layer_{i}.0." in k:
                k_new = k_new.replace(f"layer_{i}.0." , f"{model_prefix}encoder.layer.{i-1}.downsampling_layer.")
            if f"layer_{i}.1.local_rep.0." in k:
                k_new = k_new.replace(f"layer_{i}.1.local_rep.0." , f"{model_prefix}encoder.layer.{i-1}.conv_kxk.")
            if f"layer_{i}.1.local_rep.1." in k:
                k_new = k_new.replace(f"layer_{i}.1.local_rep.1." , f"{model_prefix}encoder.layer.{i-1}.conv_1x1.")
                
        for i in [3,4,5]:
            if i==3:
                j_in = [0,1]
            elif i==4:
                j_in = [0,1,2,3]
            elif i==5:
                j_in = [0,1,2]
                
            for j in j_in:
                if f"layer_{i}.1.global_rep.{j}." in k:
                    k_new = k_new.replace(f"layer_{i}.1.global_rep.{j}." , f"{model_prefix}encoder.layer.{i-1}.transformer.layer.{j}.")
            if f"layer_{i}.1.global_rep.{j+1}." in k:
                    k_new = k_new.replace(f"layer_{i}.1.global_rep.{j+1}." , f"{model_prefix}encoder.layer.{i-1}.layernorm.")
                    
            if f"layer_{i}.1.conv_proj." in k:
                k_new = k_new.replace(f"layer_{i}.1.conv_proj." , f"{model_prefix}encoder.layer.{i-1}.conv_projection.")
                
        if "pre_norm_attn.0." in k:
            k_new = k_new.replace("pre_norm_attn.0.", "layernorm_before.")
        if "pre_norm_attn.1." in k:
            k_new = k_new.replace("pre_norm_attn.1.", "attention.")
        if "pre_norm_ffn.0." in k:
            k_new = k_new.replace("pre_norm_ffn.0.", "layernorm_after.")
        if "pre_norm_ffn.1." in k:
            k_new = k_new.replace("pre_norm_ffn.1.", "ffn.conv1.")
        if "pre_norm_ffn.3." in k:
            k_new = k_new.replace("pre_norm_ffn.3.", "ffn.conv2.")
            
        if "classifier.1." in k:
            k_new = k_new.replace("classifier.1.", "classifier.")
        
        
        rename_keys.append((k, k_new))
    return rename_keys


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_mobilevitv2_checkpoint(task_name, checkpoint_path, orig_config_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileViTv2 structure.
    """
    config = get_mobilevitv2_config(task_name, orig_config_path)

    # load original state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # load huggingface model
    if task_name.startswith("ade20k_") or task_name.startswith("voc_"):
        model = MobileViTv2ForSemanticSegmentation(config).eval()
        base_model = False
    else:
        model = MobileViTv2ForImageClassification(config).eval()
        base_model = False
    # TODO - add support for object detection model   
    
    
    # remove and rename some keys of load the original model
    state_dict = checkpoint
    rename_keys = create_rename_keys(state_dict, base_model=base_model)
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)
    
    # load modified state_dict
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by MobileViTv2FeatureExtractor
    feature_extractor = MobileViTv2ImageProcessor(crop_size=config.image_size, size=config.image_size + 32)
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits

    # TODO :
    # assert torch.allclose()

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {task_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub: 
        # model_mapping = {
        #     "mobilevitv2_s": "mobilevitv2-small",
        #     "mobilevitv2_xs": "mobilevitv2-x-small",
        #     "mobilevitv2_xxs": "mobilevitv2-xx-small",
        #     "deeplabv3_mobilevitv2_s": "deeplabv3-mobilevitv2-small",
        #     "deeplabv3_mobilevitv2_xs": "deeplabv3-mobilevitv2-x-small",
        #     "deeplabv3_mobilevitv2_xxs": "deeplabv3-mobilevitv2-xx-small",
        # }#TODO

        # print("Pushing to the hub...")
        # model_name = model_mapping[mobilevitv2_name]
        # feature_extractor.push_to_hub(model_name, organization="apple")
        # model.push_to_hub(model_name, organization="apple")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task",
        default="imagenet1k_256",
        type=str,
        help=(
            "Name of the task for which the MobileViTv2 model you'd like to convert is trained on . "
            '''
                Classification (ImageNet-1k)
                    MobileViTv2 (256x256)                                     : imagenet1k_256
                    MobileViTv2 (Trained on 256x256 and Finetuned on 384x384) : imagenet1k_384
                    MobileViTv2 (Trained on ImageNet-21k and Finetuned on ImageNet-1k 256x256) : imagenet21k_to_1k_256
                    MobileViTv2 (Trained on ImageNet-21k, Finetuned on ImageNet-1k 256x256, and Finetuned on ImageNet-1k 384x384) : imagenet21k_to_1k_384
                Object Detection (MS-COCO)
                    SSD MobileViTv2 : coco_ssd
                Segmentation
                    ADE20K Dataset : ade20k_pspnet, ade20k_deeplabv3
                    Pascal VOC 2012 Dataset: voc_pspnet, voc_deeplabv3
            '''
        ),
        choices=['imagenet1k_256', 'imagenet1k_384', 'imagenet21k_to_1k_256', 'imagenet21k_to_1k_384',
                 'coco_ssd', 'ade20k_pspnet', 'ade20k_deeplabv3', 'voc_pspnet', 'voc_deeplabv3']
    )
    
    parser.add_argument(
        "--orig_checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    parser.add_argument(
        "--orig_config_path", required=True, type=str, help="Path to the original config file."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_mobilevitv2_checkpoint(
        args.task,
        args.orig_checkpoint_path, args.orig_config_path, 
        args.pytorch_dump_folder_path, args.push_to_hub
    )
