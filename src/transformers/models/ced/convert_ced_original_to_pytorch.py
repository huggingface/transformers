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
"""Convert CED checkpoints from the original repository. URL: https://github.com/RicherMans/CED"""


import argparse
from pathlib import Path

import torch

from transformers import CedConfig, CedFeatureExtractor, CedForAudioClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_keys(state_dict):
    ignore_keys = [key for key in state_dict.keys() if key.startswith("front_end")]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    name = name.replace("init_bn.1", "init_bn")
    if not name.startswith("outputlayer"):
        name = f"encoder.{name}"
    return name


@torch.no_grad()
def convert_ced_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    r"""
    TODO: Add docstring
    """

    config = CedConfig(model_name)

    model_name_to_url = {
        "ced-tiny": ("https://zenodo.org/record/8275347/files/audiotransformer_tiny_mAP_4814.pt?download=1"),
        "ced-mini": ("https://zenodo.org/record/8275347/files/audiotransformer_mini_mAP_4896.pt?download=1"),
        "ced-small": ("https://zenodo.org/record/8275319/files/audiotransformer_small_mAP_4958.pt?download=1"),
        "ced-base": ("https://zenodo.org/record/8275347/files/audiotransformer_base_mAP_4999.pt?download=1"),
    }

    state_dict = torch.hub.load_state_dict_from_url(model_name_to_url[model_name], map_location="cpu")
    remove_keys(state_dict)
    new_state_dict = {rename_key(key): val for key, val in state_dict.items()}

    model = CedForAudioClassification(config)
    model.load_state_dict(new_state_dict)

    feature_extractor = CedFeatureExtractor()

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        logger.info(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"xiaomi/{model_name}")
        feature_extractor.push_to_hub(f"xiaomi/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="ced-mini",
        type=str,
        help="Name of the CED model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_ced_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
