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
"""Convert BiT checkpoints from the timm library."""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
from transformers import BitConfig, BitForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_config(model_name):
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    conv_layer = "std_conv" if "bit" in model_name else False
    # for the ViT-hybrid checkpoints, one needs to additionally set config.layer_type = "bottleneck"
    # and use a different conv_layer, namely StdConv2dSame
    # and "stem_type": "same" in the data config
    config = BitConfig(
        conv_layer=conv_layer,
        num_labels=1000,
        id2label=id2label,
        label2id=label2id,
    )

    return config


def rename_key(name):
    if "stem.conv" in name:
        name = name.replace("stem.conv", "resnetv2.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "head.fc" in name:
        name = name.replace("head.fc", "classifier.1")
    if name.startswith("norm"):
        name = "resnetv2." + name
    if "resnetv2" not in name and "classifier" not in name:
        name = "resnetv2.encoder." + name

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_resnetv2_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our BiT structure.
    """

    # define default BiT configuration
    config = get_config(model_name)

    # load original model from timm
    timm_model = create_model(model_name, pretrained=True)
    timm_model.eval()

    # load state_dict of original model
    state_dict = timm_model.state_dict()
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val.squeeze() if "head" in key else val

    # load HuggingFace model
    model = BitForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)

    # TODO verify logits
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = transform(image).unsqueeze(0)

    print("Shape of pixel values:", pixel_values.shape)

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    print("Logits:", logits[0, :3])
    print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])
    # if model_name == "resnetv2_50x1_bitm":
    #     expected_slice = torch.tensor([ 0.1665, -0.2718, -1.1446])
    timm_logits = timm_model(pixel_values)
    assert timm_logits.shape == outputs.logits.shape
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    # assert torch.allclose(logits[0, :3], expected_slice, atol=1e-3)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="resnetv2_50x1_bitm",
        type=str,
        help="Name of the BiT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_resnetv2_checkpoint(args.model_name, args.pytorch_dump_folder_path)
