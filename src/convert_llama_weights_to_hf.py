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
"""Convert ViT and non-distilled DeiT checkpoints from the timm library."""


import argparse
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import LlaVaConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(state_dict, old, new):
    if old in state_dict:
        val = state_dict.pop(old)
        state_dict[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_llava_checkpoint(
    checkpoint_path_llava_1, checkpoint_path_llava_2, checkpoint_path_clip, pytorch_dump_folder_path
):
    """
    Copy/paste/tweak model's weights to our ProPainter structure.
    """

    # define default ViT configuration
    config = LlaVaConfig()

    # load original models
    llava_state_dict = OrderedDict()
    llava_state_dict = torch.load(checkpoint_path_llava_1, map_location="cpu")
    llava_state_dict.update(torch.load(checkpoint_path_llava_2, map_location="cpu"))
    for i in llava_state_dict:
        i = i.replace("model", "model.text_model")

    state_dict = OrderedDict()
    state_dict.update(torch.load(checkpoint_path_clip, map_location="cpu"))

    for i in state_dict:
        i = i.replace("model", "model.vision_model")

    state_dict.update(llava_state_dict)

    model = LlavaForCausalLM(config)
    model.eval()

    # load state_dict of original model, remove and rename some keys
    # rename_keys = create_rename_keys(depths())
    # print(rename_keys)
    # for src, dest in rename_keys:
    #    rename_key(state_dict, src, dest)
    model.load_state_dict(state_dict)
    print(model)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    image_processor = LlaVaProcessor(size=config.image_size)
    # encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # pixel_values = encoding["pixel_values"]
    # outputs = model(pixel_values)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path_clip",
        default="./raft-things.pth",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_path_llava_1",
        default="./recurrent_flow_completion.pth",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_path_llava_2",
        default="./recurrent_flow_completion.pth",
        type=str,
        help="Path to the original state dict (.pth file).",
    )

    parser.add_argument(
        "--pytorch_dump_folder_path", default="./", type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_llava_checkpoint(
        args.checkpoint_path_llava_1,
        args.checkpoint_path_llava_2,
        args.checkpoint_path_clip,
        args.pytorch_dump_folder_path,
    )
