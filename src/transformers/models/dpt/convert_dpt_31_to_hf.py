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
"""Convert DPT 3.1 checkpoints from the MiDaS repository. URL: https://github.com/isl-org/MiDaS"""


import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config():
    backbone_config = BeitConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

    config = DPTConfig(backbone_config=backbone_config)

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys():
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("backbone.downsample_layers.0.0.weight", "backbone.embeddings.patch_embeddings.weight"))
    rename_keys.append(("backbone.downsample_layers.0.0.bias", "backbone.embeddings.patch_embeddings.bias"))
    rename_keys.append(("backbone.downsample_layers.0.1.weight", "backbone.embeddings.layernorm.weight"))
    rename_keys.append(("backbone.downsample_layers.0.1.bias", "backbone.embeddings.layernorm.bias"))
    
    return rename_keys


def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    name_to_url = {
        "dpt-beit-large-512": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
    }

    # define DPT configuration based on URL
    checkpoint_url = name_to_url[model_name]
    config = get_dpt_config()
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # remove certain keys
    remove_ignore_keys_(state_dict)
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # read in qkv matrices
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = DPTForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Check outputs on an image
    size = 384
    processor = DPTImageProcessor(size={"height": size, "width": size})

    image = prepare_img()
    encoding = processor(image, return_tensors="pt")

    # forward pass
    outputs = model(**encoding).predicted_depth

    # TODO assert logits
    # expected_slice = torch.tensor([[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]])
    # if "ade" in checkpoint_url:
    #     expected_slice = torch.tensor([[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]])
    # assert outputs.shape == torch.Size(expected_shape)
    # assert (
    #     torch.allclose(outputs[0, 0, :3, :3], expected_slice, atol=1e-4)
    #     if "ade" in checkpoint_url
    #     else torch.allclose(outputs[0, :3, :3], expected_slice)
    # )
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(repo_id=f"nielsr/{model_name}")
        processor.push_to_hub(repo_id=f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dpt-large",
        type=str,
        choices=["dpt-large", "dpt-large-ade"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)