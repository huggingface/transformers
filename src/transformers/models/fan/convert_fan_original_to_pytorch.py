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
"""Convert Fan checkpoints."""


import argparse
import functools
import json
import re

import torch

from huggingface_hub import hf_hub_download
from transformers import FanConfig, FanForImageClassification, FanImageProcessor


fan_ckpts = {
    "fan_tiny_12_p16_224": (
        "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_vit_tiny.pth.tar"
    ),
    "fan_small_12_p16_224": (
        "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_vit_small.pth.tar"
    ),
    "fan_base_18_p16_224": (
        "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_vit_base.pth.tar"
    ),
    "fan_tiny_8_p4_hybrid": (
        "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_tiny.pth.tar"
    ),
    "fan_small_12_p4_hybrid": "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_small.pth.tar",
    "fan_base_16_p4_hybrid": (
        "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_base.pth.tar"
    ),
    "fan_large_16_p4_hybrid": "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_large_in22k_1k.pth.tar",
}
# Configuration Values different from defaults
config_dict = {
    "fan_tiny_12_p16_224": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 4},
    "fan_small_12_p16_224_se_attn": {"hidden_size": 384, "num_hidden_layers": 12, "se_mlp": True},
    "fan_small_12_p16_224": {"hidden_size": 384, "num_hidden_layers": 12},
    "fan_base_18_p16_224": {},
    "fan_large_24_p16_224": {"hidden_size": 480, "num_hidden_layers": 24, "num_attention_heads": 10},
    "fan_tiny_8_p4_hybrid": {
        "hidden_size": 192,
        "num_hidden_layers": 8,
        "segmentation_in_channels": [128, 256, 192, 192],
        "out_index": 7,
    },
    "fan_small_12_p4_hybrid": {
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "segmentation_in_channels": [128, 256, 384, 384],
        "out_index": 9,
    },
    "fan_base_16_p4_hybrid": {
        "num_hidden_layers": 16,
        "segmentation_in_channels": [128, 256, 448, 448],
        "out_index": 15,
    },
    "fan_large_16_p4_hybrid": {
        "hidden_size": 480,
        "num_hidden_layers": 22,
        "num_attention_heads": 10,
        "segmentation_in_channels": [128, 256, 480, 480],
        "out_index": 18,
    },
    "fan_Xlarge_16_p4_hybrid": {
        "hidden_size": 528,
        "num_hidden_layers": 23,
        "num_attention_heads": [
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            11,
            16,
            16,
            16,
        ],
    },
}


def get_fan_config(name):
    config = FanConfig(**config_dict[name])
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config


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


def remap_patch_embed(key):
    return key.replace("patch_embed", "patch_embeddings")


def remap_embeddings(key):
    if "embed" in key:
        return f"fan.embeddings.{key}"
    return key


def remap_gamma(key):
    return key.replace("gamma", "weight")


def remap_head(key):
    if key.split(".")[0] in ("norm", "head"):
        return f"head.{key}"
    return key


def remap_encoder(key):
    if any(x in key for x in ["fan", "head"]):
        return key
    return f"fan.encoder.{key}"


def remap_blocks(key):
    pattern = "([a-z\.]*blocks\.\d*\.)"
    if re.match(pattern, key):
        return re.sub(pattern, "\\1block.", key)
    return key


def remap_proj_keys(key):
    pattern = "([a-z\.]*patch_embed\.proj\.\d*\.)"
    if re.match(pattern, key):
        stem = ".".join(key.split(".")[:-3])
        first = int(key.split(".")[-3])
        second = int(key.split(".")[-2])
        name = key.split(".")[-1]
        return f"{stem}.{first + first//2 + second}.{name}"
    return key


def remap_segmentation_linear(key):
    if "decode_head.linear_fuse.conv" in key:
        return key.replace("decode_head.linear_fuse.conv", "decode_head.linear_fuse")
    if "decode_head.linear_fuse.bn" in key:
        return key.replace("decode_head.linear_fuse.bn", "decode_head.batch_norm")
    if "decode_head.linear_pred" in key:
        return key.replace("decode_head.linear_pred", "decode_head.classifier")
    return key


def remap_linear_fuse(key):
    for num in range(4):
        if f"decode_head.linear_c{num+1}" in key:
            return key.replace(f"decode_head.linear_c{num+1}", f"decode_head.linear_c.{num}")
    return key


def remap_qkv(key):
    elements = key.split(".")
    mapping_dict = {"q": "query", "v": "value", "k": "key", "kv": "key_value"}
    return ".".join([mapping_dict.get(elem, elem) for elem in elements])


remap_fn = compose(
    remap_segmentation_linear,
    remap_linear_fuse,
    remap_blocks,
    remap_encoder,
    remap_gamma,
    remap_head,
    remap_embeddings,
    remap_patch_embed,
    remap_proj_keys,
    remap_qkv,
)


def remap_state(state_dict):
    return {remap_fn(key): weights for key, weights in state_dict.items()}


def convert_fan_checkpoint(fan_name, pytorch_dump_folder_path):
    config = get_fan_config(fan_name)
    model = FanForImageClassification(config)
    model.eval()
    if fan_name in fan_ckpts:
        new_state_dict = remap_state(torch.hub.load_state_dict_from_url(fan_ckpts[fan_name]))
        model.load_state_dict(new_state_dict)
        print(f"model {fan_name} has a checkpoint at {fan_ckpts[fan_name]}")

    image_processor = FanImageProcessor()

    print(f"Saving model {fan_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fan_name",
        default="fan_tiny_12_p16_224",
        type=str,
        help="Name of the Fan model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_fan_checkpoint(args.fan_name, args.pytorch_dump_folder_path)
