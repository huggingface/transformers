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
"""Convert EfficientNet checkpoints from the original repository.

URL: https://github.com/facebookresearch/ConvNeXt"""


import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import *
from transformers import EfficientNetImageProcessor, EfficientNetConfig, EfficientNetForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

model_classes = {
    'b0': EfficientNetB0,
    'b1': EfficientNetB1,
    'b2': EfficientNetB2,
    'b3': EfficientNetB3,
    'b4': EfficientNetB4,
    'b5': EfficientNetB5,
    'b6': EfficientNetB6,
    'b7': EfficientNetB7,
}

CONFIG_MAP = {
    'b0': {"width_coef": 1.0, "depth_coef": 1.0, "image_size": 224, "dropout_rate": 0.2},
    'b1': {"width_coef": 1.0, "depth_coef": 1.1, "image_size": 240, "dropout_rate": 0.2},
    'b2': {"width_coef": 1.1, "depth_coef": 1.2, "image_size": 260, "dropout_rate": 0.3},
    'b3': {"width_coef": 1.2, "depth_coef": 1.4, "image_size": 300, "dropout_rate": 0.3},
    'b4': {"width_coef": 1.4, "depth_coef": 1.8, "image_size": 380, "dropout_rate": 0.4},
    'b5': {"width_coef": 1.6, "depth_coef": 2.2, "image_size": 456, "dropout_rate": 0.4},
    'b6': {"width_coef": 1.8, "depth_coef": 2.6, "image_size": 528, "dropout_rate": 0.5},
    'b7': {"width_coef": 2.0, "depth_coef": 3.1, "image_size": 600, "dropout_rate": 0.5},
}


def get_efficientnet_config(model_name):
    config = EfficientNetConfig()
    config.width_coefficient = CONFIG_MAP[model_name]["width_coef"]
    config.depth_coefficient = CONFIG_MAP[model_name]["depth_coef"]
    config.image_size = CONFIG_MAP[model_name]["image_size"]
    config.dropout_rate = CONFIG_MAP[model_name]["dropout_rate"]

    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    config.num_labels = 1000
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths
    return config


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def convert_image_processor(model_name):
    size = CONFIG_MAP[model_name]["image_size"]
    preprocessor = EfficientNetImageProcessor(
        size={"height": size, "width": size}, 
        do_center_crop=False,
    )
    return preprocessor

# here we list all keys to be renamed (original name on the left, our name on the right)
def rename_keys(original_param_names):
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = sorted(list(set(block_names)))
    num_blocks = len(block_names)
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    rename_keys = []
    rename_keys.append("stem_conv/kernel:0", "embeddings.convolution.weight")
    rename_keys.append("stem_bn/gamma:0", "embeddings.batchnorm.weight")
    rename_keys.append("stem_bn/beta:0", "embeddings.batchnorm.bias")

    for b in range(block_names):
        hf_b = block_name_mapping[b]
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.bias"))

    key_mapping = {item[0]: item[1] for item in rename_keys}
    return key_mapping


@torch.no_grad()
def convert_efficientnet_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our EfficientNet structure.
    """
    # Load original model
    original_model = model_classes[model_name](
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    vars = original_model.trainable_variables
    var_names = [v.name for v in if vars]
    key_mapping = rename_keys(var_names)

    # Define EfficientNet configuration based on URL
    config, expected_shape = get_efficientnet_config(checkpoint_url)
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # Add prefix to all keys expect classifier head
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "efficientnet." + key
        state_dict[key] = val

    # Load HuggingFace model
    config = get_efficientnet_config(model_name)
    model = EfficientNetForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize preprocessor and preprocess input image
    preprocessor = convert_image_processor(model_name)
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")

    logits = model(pixel_values).logits

    # note: the logits below were obtained without center cropping
    if model_name == "b0":
        expected_logits = torch.tensor([-0.1210, -0.6605, 0.1918])
    elif model_name == "b1":
        expected_logits = torch.tensor([-0.4473, -0.1847, -0.6365])
    elif model_name == "b2":
        expected_logits = torch.tensor([0.4525, 0.7539, 0.0308])
    elif model_name == "b3":
        expected_logits = torch.tensor([0.3561, 0.6350, -0.0384])
    elif model_name == "b4":
        expected_logits = torch.tensor([0.4174, -0.0989, 0.1489])
    elif model_name == "b5":
        expected_logits = torch.tensor([0.2513, -0.1349, -0.1613])
    elif model_name == "b6":
        expected_logits = torch.tensor([1.2980, 0.3631, -0.1198])
    elif model_name == "b7":
        expected_logits = torch.tensor([1.2963, 0.1227, 0.1723])
    else:
        raise ValueError(f"Unknown version: {model_name}")

    assert torch.allclose(logits[0, :3], expected_logits, atol=1e-3)
    assert logits.shape == expected_shape

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    print("Pushing model to the hub...")
    model_name = "efficientnet-" + model_name

    model.push_to_hub(
        repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
        organization="adirik",
        commit_message="Add model",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="b0",
        type=str,
        help="Version name of the EfficientNet model you want to convert, select from [b0, b1, b2, b3, b4, b5, b6, b7].",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )

    args = parser.parse_args()
    convert_efficientnet_checkpoint(args.model_name, args.pytorch_dump_folder_path)
