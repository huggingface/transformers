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

URL: https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/efficientnet.py"""

import argparse
import json
import os

import numpy as np
import PIL
import requests
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image

from transformers import (
    EfficientNetConfig,
    EfficientNetForImageClassification,
    EfficientNetImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

model_classes = {
    "b0": efficientnet.EfficientNetB0,
    "b1": efficientnet.EfficientNetB1,
    "b2": efficientnet.EfficientNetB2,
    "b3": efficientnet.EfficientNetB3,
    "b4": efficientnet.EfficientNetB4,
    "b5": efficientnet.EfficientNetB5,
    "b6": efficientnet.EfficientNetB6,
    "b7": efficientnet.EfficientNetB7,
}

CONFIG_MAP = {
    "b0": {
        "hidden_dim": 1280,
        "width_coef": 1.0,
        "depth_coef": 1.0,
        "image_size": 224,
        "dropout_rate": 0.2,
        "dw_padding": [],
    },
    "b1": {
        "hidden_dim": 1280,
        "width_coef": 1.0,
        "depth_coef": 1.1,
        "image_size": 240,
        "dropout_rate": 0.2,
        "dw_padding": [16],
    },
    "b2": {
        "hidden_dim": 1408,
        "width_coef": 1.1,
        "depth_coef": 1.2,
        "image_size": 260,
        "dropout_rate": 0.3,
        "dw_padding": [5, 8, 16],
    },
    "b3": {
        "hidden_dim": 1536,
        "width_coef": 1.2,
        "depth_coef": 1.4,
        "image_size": 300,
        "dropout_rate": 0.3,
        "dw_padding": [5, 18],
    },
    "b4": {
        "hidden_dim": 1792,
        "width_coef": 1.4,
        "depth_coef": 1.8,
        "image_size": 380,
        "dropout_rate": 0.4,
        "dw_padding": [6],
    },
    "b5": {
        "hidden_dim": 2048,
        "width_coef": 1.6,
        "depth_coef": 2.2,
        "image_size": 456,
        "dropout_rate": 0.4,
        "dw_padding": [13, 27],
    },
    "b6": {
        "hidden_dim": 2304,
        "width_coef": 1.8,
        "depth_coef": 2.6,
        "image_size": 528,
        "dropout_rate": 0.5,
        "dw_padding": [31],
    },
    "b7": {
        "hidden_dim": 2560,
        "width_coef": 2.0,
        "depth_coef": 3.1,
        "image_size": 600,
        "dropout_rate": 0.5,
        "dw_padding": [18],
    },
}


def get_efficientnet_config(model_name):
    config = EfficientNetConfig()
    config.hidden_dim = CONFIG_MAP[model_name]["hidden_dim"]
    config.width_coefficient = CONFIG_MAP[model_name]["width_coef"]
    config.depth_coefficient = CONFIG_MAP[model_name]["depth_coef"]
    config.image_size = CONFIG_MAP[model_name]["image_size"]
    config.dropout_rate = CONFIG_MAP[model_name]["dropout_rate"]
    config.depthwise_padding = CONFIG_MAP[model_name]["dw_padding"]

    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    config.num_labels = 1000
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
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
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.47853944, 0.4732864, 0.47434163],
        do_center_crop=False,
    )
    return preprocessor


# here we list all keys to be renamed (original name on the left, our name on the right)
def rename_keys(original_param_names):
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = sorted(set(block_names))
    num_blocks = len(block_names)
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    rename_keys = []
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))

    for b in block_names:
        hf_b = block_name_mapping[b]
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        rename_keys.append(
            (f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var")
        )
        rename_keys.append(
            (f"block{b}_dwconv/depthwise_kernel:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_conv.weight")
        )
        rename_keys.append((f"block{b}_bn/gamma:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.weight"))
        rename_keys.append((f"block{b}_bn/beta:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.bias"))
        rename_keys.append(
            (f"block{b}_bn/moving_mean:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_mean")
        )
        rename_keys.append(
            (f"block{b}_bn/moving_variance:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_var")
        )

        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        rename_keys.append(
            (f"block{b}_project_conv/kernel:0", f"encoder.blocks.{hf_b}.projection.project_conv.weight")
        )
        rename_keys.append((f"block{b}_project_bn/gamma:0", f"encoder.blocks.{hf_b}.projection.project_bn.weight"))
        rename_keys.append((f"block{b}_project_bn/beta:0", f"encoder.blocks.{hf_b}.projection.project_bn.bias"))
        rename_keys.append(
            (f"block{b}_project_bn/moving_mean:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_project_bn/moving_variance:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_var")
        )

    rename_keys.append(("top_conv/kernel:0", "encoder.top_conv.weight"))
    rename_keys.append(("top_bn/gamma:0", "encoder.top_bn.weight"))
    rename_keys.append(("top_bn/beta:0", "encoder.top_bn.bias"))
    rename_keys.append(("top_bn/moving_mean:0", "encoder.top_bn.running_mean"))
    rename_keys.append(("top_bn/moving_variance:0", "encoder.top_bn.running_var"))

    key_mapping = {}
    for item in rename_keys:
        if item[0] in original_param_names:
            key_mapping[item[0]] = "efficientnet." + item[1]

    key_mapping["predictions/kernel:0"] = "classifier.weight"
    key_mapping["predictions/bias:0"] = "classifier.bias"
    return key_mapping


def replace_params(hf_params, tf_params, key_mapping):
    for key, value in tf_params.items():
        if "normalization" in key:
            continue

        hf_key = key_mapping[key]
        if "_conv" in key and "kernel" in key:
            new_hf_value = torch.from_numpy(value).permute(3, 2, 0, 1)
        elif "depthwise_kernel" in key:
            new_hf_value = torch.from_numpy(value).permute(2, 3, 0, 1)
        elif "kernel" in key:
            new_hf_value = torch.from_numpy(np.transpose(value))
        else:
            new_hf_value = torch.from_numpy(value)

        # Replace HF parameters with original TF model parameters
        assert hf_params[hf_key].shape == new_hf_value.shape
        hf_params[hf_key].copy_(new_hf_value)


@torch.no_grad()
def convert_efficientnet_checkpoint(model_name, pytorch_dump_folder_path, save_model, push_to_hub):
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

    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())

    # Load HuggingFace model
    config = get_efficientnet_config(model_name)
    hf_model = EfficientNetForImageClassification(config).eval()
    hf_params = hf_model.state_dict()

    # Create src-to-dst parameter name mapping dictionary
    print("Converting parameters...")
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)

    # Initialize preprocessor and preprocess input image
    preprocessor = convert_image_processor(model_name)
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")

    # HF model inference
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)
    hf_logits = outputs.logits.detach().numpy()

    # Original model inference
    original_model.trainable = False
    image_size = CONFIG_MAP[model_name]["image_size"]
    img = prepare_img().resize((image_size, image_size), resample=PIL.Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    original_logits = original_model.predict(x)

    # Check whether original and HF model outputs match  -> np.allclose
    assert np.allclose(original_logits, hf_logits, atol=1e-3), "The predicted logits are not the same."
    print("Model outputs match!")

    if save_model:
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # Save converted model and image processor
        hf_model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Push model and image processor to hub
        print(f"Pushing converted {model_name} to the hub...")
        model_name = f"efficientnet-{model_name}"
        preprocessor.push_to_hub(model_name)
        hf_model.push_to_hub(model_name)


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
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")

    args = parser.parse_args()
    convert_efficientnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
