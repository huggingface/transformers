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
"""Convert ConvNext checkpoints from the original repository."""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import ConvNextConfig, ConvNextForImageClassification, ViTFeatureExtractor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnext_config(checkpoint_url):
    config = ConvNextConfig()

    if "tiny" in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    elif "small" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    elif "base" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    elif "large" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    elif "xlarge" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]

    if "22k" in checkpoint_url:
        num_labels = 21841
        filename = "imagenet-22k-id2label.json"
        expected_shape = (1, 21841)
    else:
        num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        expected_shape = (1, 1000)

    repo_id = "datasets/huggingface/label-files"
    config.num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    if "22k" in checkpoint_url:
        # this dataset contains 21843 labels but the model only has 21841
        # we delete the classes as mentioned in https://github.com/google-research/big_transfer/issues/18
        del id2label[9205]
        del id2label[15027]
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths

    return config, expected_shape


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys():
    rename_keys = [
        ("norm.weight", "layernorm.weight"),
        ("norm.bias", "layernorm.bias"),
        ("head.weight", "classifier.weight"),
        ("head.bias", "classifier.bias"),
    ]

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_convnext_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ConvNext structure.
    """

    # define ConvNext configuration based on URL
    config, expected_shape = get_convnext_config(checkpoint_url)
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    rename_keys = create_rename_keys()
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # add prefix
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "convnext." + key
        state_dict[key] = val

    # load HuggingFace model
    model = ConvNextForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor()
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    outputs = model(pixel_values)
    logits = outputs.logits

    # TODO assert values
    print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])
    print("Shape of logits:", logits.shape)
    assert outputs.logits.shape == torch.Size(expected_shape)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        type=str,
        help="URL of the ConvNext original checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )

    args = parser.parse_args()
    convert_convnext_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)