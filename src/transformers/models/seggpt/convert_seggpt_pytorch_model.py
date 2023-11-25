# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
import logging
from collections import OrderedDict

import torch

from transformers import SegGPTForInstanceSegmentation, SegGPTImageProcessor
from transformers.models.seggpt import SegGPTConfig


tiny_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_tiny.config"
small_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_small.config"
base_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_base.config"

encoder_keys = [
    "mask_token",
    "segment_token_x",
    "segment_token_y",
    "type_token_cls",
    "pos_embed",
    "patch_embed",
    "group_blocks",
    "type_token_ins",
    "norm.bias",
    "norm.weight",
]

decoder_keys = [
    "decoder_embed.weight",
    "decoder_embed.bias",
    "decoder_pred.0.weight",
    "decoder_pred.0.bias",
    "decoder_pred.1.weight",
    "decoder_pred.1.bias",
    "decoder_pred.3.weight",
    "decoder_pred.3.bias",
]


def convert_textnet_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    config = SegGPTConfig()
    model = SegGPTForInstanceSegmentation(config)

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device("cpu"))["model"]
    state_dict_changed = OrderedDict()
    num_group_blocks = config.num_group_blocks
    num_blocks_in_group = config.num_blocks_in_group

    rename_keys_for_blocks = {}
    for ix in range(0, num_blocks_in_group * num_group_blocks):
        group_to_assign = ix // num_blocks_in_group
        block_id_in_group = ix % num_blocks_in_group
        rename_keys_for_blocks[f"blocks.{ix}"] = f"group_blocks.{group_to_assign}.blocks.{block_id_in_group}"

    for key in state_dict:
        new_key = key
        for rename_key in list(rename_keys_for_blocks.keys())[::-1]:
            if rename_key in new_key:
                new_key = new_key.replace(rename_key, rename_keys_for_blocks[rename_key])
                break

        for encoder_key in encoder_keys:
            if encoder_key in new_key:
                new_key = new_key.replace(encoder_key, "encoder." + encoder_key)
                break

        for decoder_key in decoder_keys:
            if decoder_key in new_key:
                new_key = new_key.replace(decoder_key, "decoder." + decoder_key)
                break

        new_key = "seggpt_model." + new_key

        state_dict_changed[new_key] = state_dict[key]

    model.load_state_dict(state_dict_changed)
    processor = SegGPTImageProcessor(size={"shortest_edge": 448})
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)
    # textnet_image_processor.save_pretrained(pytorch_dump_folder_path)
    logging.info("The converted weights are save here : " + pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()

    convert_textnet_checkpoint(
        args.checkpoint_url,
        args.pytorch_dump_folder_path,
    )
