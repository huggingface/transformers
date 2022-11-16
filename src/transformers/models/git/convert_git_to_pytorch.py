# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert GIT checkpoints from the original repository.

URL: https://github.com/microsoft/GenerativeImage2Text/tree/main"""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import GITConfig, GITForCausalLM, GITVisionConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_git_config(model_name):
    config = GITConfig(vision_config=GITVisionConfig())

    return config


def rename_key(name):
    if "downsample_layers.0.0" in name:
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our GIT structure.
    """

    model_name_to_url = {
        "git-base": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE/snapshot/model.pt",
    }

    # define GIT configuration based on model name
    config = get_git_config(model_name)
    # load original state_dict from URL
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val

    # load HuggingFace model
    model = GITForCausalLM(config)
    model.load_state_dict(state_dict)
    model.eval()

    # TODO verify results
    pixel_values = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, 100, (1, 10))

    logits = model(pixel_values, input_ids).logits
    print(logits.shape)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model to the hub...")
        model.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="git-base",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    args = parser.parse_args()
    convert_git_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
