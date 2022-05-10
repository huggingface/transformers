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
"""Convert ViT checkpoints trained with the DINO method."""


import argparse
from pathlib import Path

import torch

from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None, base_model=True):
    """
    Copy/paste/tweak model's weights to our ViT structure.
    """
    weight_dict = []
    vqvae_dic = torch.load("/Users/arthur/Work/HuggingFace/jukebox/porting/vqvae.pth")
    weight_dict.append(vqvae_dic)

    for dict_name in ["up0", "up1", "up2"]:
        old_dic = torch.load(f"/Users/arthur/Work/HuggingFace/jukebox/porting/{dict_name}.pth")
        new_dic = {}
        for k in old_dic.keys():
            if k.endswith(".b"):
                new_dic[k.replace("b", "bias")] = old_dic[k]
            elif k.endswith(".w"):
                new_dic[k.replace("w", "weight")] = old_dic[k]
            elif dict_name != "up2" and "cond.model." in k:
                new_dic[k.replace(".blocks.", ".model.")] = old_dic[k]
            else:
                new_dic[k] = old_dic[k]
        weight_dict.append(new_dic)

    return weight_dict
    config = JukeboxConfig.from_pretrained(model_name)
    model = JukeboxModel(config)

    vq, *weights = weight_dict
    model.vqvae.load_state_dict(vq)
    for i in range(len(weights)):
        model.priors[i].load_state_dict(weights[i])

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    return weight_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dino_vitb16",
        type=str,
        help="Name of the model trained with DINO you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Whether to only convert the base model (no projection head weights).",
    )

    parser.set_defaults(base_model=True)
    args = parser.parse_args()
    convert_openai_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.base_model)
