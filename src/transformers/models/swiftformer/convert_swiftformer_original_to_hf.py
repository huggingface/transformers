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
"""Convert SwiftFormer checkpoints from the original implementation."""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SwiftFormerConfig,
    SwiftFormerForImageClassification,
    ViTImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

device = torch.device("cpu")


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_expected_output(swiftformer_name):
    if swiftformer_name == "swiftformer_xs":
        return torch.tensor([-2.1703e00, 2.1107e00, -2.0811e00, 8.8685e-01, 2.4360e-01])

    elif swiftformer_name == "swiftformer_s":
        return torch.tensor([3.9636e-01, 2.3478e-01, -1.6963e00, -1.7381e00, -8.6337e-01])

    elif swiftformer_name == "swiftformer_l1":
        return torch.tensor([-4.2768e-01, -4.7429e-01, -1.0897e00, -1.0248e00, 3.5523e-02])

    elif swiftformer_name == "swiftformer_l3":
        return torch.tensor([-2.5330e-01, 2.4211e-01, -6.0185e-01, -8.2789e-01, -6.0446e-02])


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def create_rename_keys(state_dict):
    rename_keys = []
    for k in state_dict.keys():
        k_new = k
        if ".pwconv" in k:
            k_new = k_new.replace(".pwconv", ".point_wise_conv")
        if ".dwconv" in k:
            k_new = k_new.replace(".dwconv", ".depth_wise_conv")
        if ".Proj." in k:
            k_new = k_new.replace(".Proj.", ".proj.")
        if "patch_embed" in k_new:
            k_new = k_new.replace("patch_embed", "swiftformer.patch_embed.patch_embedding")
        if "network" in k_new:
            ls = k_new.split(".")
            if ls[2].isdigit():
                k_new = "swiftformer.encoder.network." + ls[1] + ".blocks." + ls[2] + "." + ".".join(ls[3:])
            else:
                k_new = k_new.replace("network", "swiftformer.encoder.network")
        rename_keys.append((k, k_new))
    return rename_keys


@torch.no_grad()
def convert_swiftformer_checkpoint(swiftformer_name, pytorch_dump_folder_path, original_ckpt):
    """
    Copy/paste/tweak model's weights to our SwiftFormer structure.
    """

    # define default SwiftFormer configuration
    config = SwiftFormerConfig()

    # dataset (ImageNet-21k only or also fine-tuned on ImageNet 2012), patch_size and image_size
    config.num_labels = 1000
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # size of the architecture
    if swiftformer_name == "swiftformer_xs":
        config.depths = [3, 3, 6, 4]
        config.embed_dims = [48, 56, 112, 220]

    elif swiftformer_name == "swiftformer_s":
        config.depths = [3, 3, 9, 6]
        config.embed_dims = [48, 64, 168, 224]

    elif swiftformer_name == "swiftformer_l1":
        config.depths = [4, 3, 10, 5]
        config.embed_dims = [48, 96, 192, 384]

    elif swiftformer_name == "swiftformer_l3":
        config.depths = [4, 4, 12, 6]
        config.embed_dims = [64, 128, 320, 512]

    # load state_dict of original model, remove and rename some keys
    if original_ckpt:
        if original_ckpt.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(original_ckpt, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(original_ckpt, map_location="cpu")
    state_dict = checkpoint

    rename_keys = create_rename_keys(state_dict)
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)

    # load HuggingFace model
    hf_model = SwiftFormerForImageClassification(config).eval()
    hf_model.load_state_dict(state_dict)

    # prepare test inputs
    image = prepare_img()
    processor = ViTImageProcessor.from_pretrained("preprocessor_config")
    inputs = processor(images=image, return_tensors="pt")

    # compare outputs from both models
    timm_logits = get_expected_output(swiftformer_name)
    hf_logits = hf_model(inputs["pixel_values"]).logits

    assert hf_logits.shape == torch.Size([1, 1000])
    assert torch.allclose(hf_logits[0, 0:5], timm_logits, atol=1e-3)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {swiftformer_name} to {pytorch_dump_folder_path}")
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--swiftformer_name",
        default="swiftformer_xs",
        choices=["swiftformer_xs", "swiftformer_s", "swiftformer_l1", "swiftformer_l3"],
        type=str,
        help="Name of the SwiftFormer model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="./converted_outputs/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--original_ckpt", default=None, type=str, help="Path to the original model checkpoint.")

    args = parser.parse_args()
    convert_swiftformer_checkpoint(args.swiftformer_name, args.pytorch_dump_folder_path, args.original_ckpt)
