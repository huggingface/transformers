# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
import os

import torch

from transformers import FlavaImageCodebook, FlavaImageCodebookConfig


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def count_parameters(state_dict):
    # encoder.embeddings are double copied in original FLAVA
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


def upgrade_state_dict(state_dict):
    upgrade = {}

    group_keys = ["group_1", "group_2", "group_3", "group_4"]
    for key, value in state_dict.items():
        for group_key in group_keys:
            if group_key in key:
                key = key.replace(f"{group_key}.", f"{group_key}.group.")

        if "res_path" in key:
            key = key.replace("res_path.", "res_path.path.")

        if key.endswith(".w"):
            key = rreplace(key, ".w", ".weight", 1)
        if key.endswith(".b"):
            key = rreplace(key, ".b", ".bias", 1)

        upgrade[key] = value.float()

    return upgrade


@torch.no_grad()
def convert_dalle_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, save_checkpoint=True):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    from dall_e import Encoder

    encoder = Encoder()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
    else:
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path)

    if isinstance(ckpt, Encoder):
        ckpt = ckpt.state_dict()
    encoder.load_state_dict(ckpt)

    if config_path is not None:
        config = FlavaImageCodebookConfig.from_pretrained(config_path)
    else:
        config = FlavaImageCodebookConfig()

    hf_model = FlavaImageCodebook(config).eval()
    state_dict = encoder.state_dict()

    hf_state_dict = upgrade_state_dict(state_dict)
    hf_model.load_state_dict(hf_state_dict)
    hf_state_dict = hf_model.state_dict()
    hf_count = count_parameters(hf_state_dict)
    state_dict_count = count_parameters(state_dict)

    assert torch.allclose(hf_count, state_dict_count, atol=1e-3)

    if save_checkpoint:
        hf_model.save_pretrained(pytorch_dump_folder_path)
    else:
        return hf_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_dalle_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
