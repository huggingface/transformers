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
"""Convert Omnivore checkpoints from timm."""


import argparse
import json
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import OmnivoreConfig, OmnivoreFeatureExtractor, OmnivoreForVisionClassification
from transformers.utils import logging


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

logging.set_verbosity_info()
logger = logging.get_logger()


def convert_weight_and_push(config: OmnivoreConfig, name: str, save_directory: Path, names_to_save_names: Dict):
    print(f"Converting {name}...")
    device = "cpu"
    src_model = torch.hub.load("facebookresearch/omnivore:main", model=name).eval()
    src_model = src_model.to(device)
    src_keys = list(src_model.state_dict().keys())
    src_weights = src_model.state_dict()

    dest_model = OmnivoreForVisionClassification(config).to(device)
    dest_keys = list(dest_model.state_dict().keys())

    assert len(src_keys) == len(dest_keys), "The models are not exactly same"
    number_of_keys = len(src_keys)
    print("Number of Keys: ", number_of_keys)
    new_weights = OrderedDict()
    for i in range(number_of_keys):
        new_weights[dest_keys[i]] = src_weights[src_keys[i]]

    dest_model.load_state_dict(new_weights)
    dest_model.eval()

    x = torch.randn(2, 3, 6, 224, 224)
    out1 = src_model(x, "video")
    out2 = dest_model(x, "video").logits
    assert torch.allclose(out1, out2), "The model logits don't match the original one for videos"

    feature_extractor = OmnivoreFeatureExtractor()
    inputs = feature_extractor(images=image, return_tensors="pt")
    out1 = src_model(inputs["pixel_values"].unsqueeze(2), "image")
    out2 = dest_model(inputs["pixel_values"], "image").logits
    assert torch.allclose(out1, out2), "The model logits don't match the original one for images"

    x = torch.randn(2, 4, 1, 224, 224)
    out1 = src_model(x, "rgbd")
    out2 = dest_model(x, "rgbd").logits
    assert torch.allclose(out1, out2), "The model logits don't match the original one for rgbd"

    checkpoint_name = names_to_save_names[name]
    print(checkpoint_name)

    dest_model.save_pretrained(save_directory / checkpoint_name)
    feature_extractor = OmnivoreFeatureExtractor()
    feature_extractor.save_pretrained(save_directory / checkpoint_name)
    print(f"Pushed {checkpoint_name}\n\n")


def convert_weights_and_push(save_directory: Path, model_name: str = None):
    filename = "imagenet-1k-id2label.json"
    image_num_labels = 1000
    expected_shape = (1, image_num_labels)

    repo_id = "datasets/huggingface/label-files"
    image_id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    image_id2label = {int(k): v for k, v in image_id2label.items()}
    image_label2id = {v: k for k, v in image_id2label.items()}

    filename = "kinetics_classnames.json"
    video_num_labels = 400
    expected_shape = (1, video_num_labels)
    video_id2label = json.load(open(filename, "r"))
    video_id2label = {int(v): str(k).replace('"', "") for k, v in video_id2label.items()}
    video_label2id = {v: k for k, v in video_id2label.items()}

    filename = "sunrgbd_classnames.json"
    rgbd_num_labels = 19
    expected_shape = (1, rgbd_num_labels)
    rgbd_id2label = json.load(open(filename, "r"))
    rgbd_id2label = {int(k): v for k, v in rgbd_id2label.items()}
    rgbd_label2id = {v: k for k, v in rgbd_id2label.items()}

    OmnivorePreTrainedConfig = partial(
        OmnivoreConfig,
        image_num_labels=image_num_labels,
        image_id2label=image_id2label,
        image_label2id=image_label2id,
        video_id2label=video_id2label,
        video_label2id=video_label2id,
        rgbd_id2label=rgbd_id2label,
        rgbd_label2id=rgbd_label2id,
    )
    names_to_config = {
        "omnivore_swinT": OmnivorePreTrainedConfig(
            patch_size=[2, 4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[8, 7, 7],
            drop_path_rate=0.2,
            patch_norm=True,
            depth_mode="summed_rgb_d_tokens",
        ),
        "omnivore_swinB": OmnivorePreTrainedConfig(
            patch_size=[2, 4, 4],
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=[16, 7, 7],
            drop_path_rate=0.3,
            patch_norm=True,
            depth_mode="summed_rgb_d_tokens",
        ),
        "omnivore_swinB_imagenet21k": OmnivorePreTrainedConfig(
            patch_size=[2, 4, 4],
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=[16, 7, 7],
            drop_path_rate=0.3,
            patch_norm=True,
            depth_mode="summed_rgb_d_tokens",
        ),
        "omnivore_swinS": OmnivorePreTrainedConfig(
            patch_size=[2, 4, 4],
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[8, 7, 7],
            drop_path_rate=0.3,
            patch_norm=True,
            depth_mode="summed_rgb_d_tokens",
        ),
        "omnivore_swinL_imagenet21k": OmnivorePreTrainedConfig(
            patch_size=[2, 4, 4],
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=[8, 7, 7],
            drop_path_rate=0.3,
            patch_norm=True,
            depth_mode="summed_rgb_d_tokens",
        ),
    }

    names_to_save_names = {
        "omnivore_swinT": "omnivore-swinT",
        "omnivore_swinS": "omnivore-swinS",
        "omnivore_swinB": "omnivore-swinB",
        "omnivore_swinB_imagenet21k": "omnivore-swinB-in21k",
        "omnivore_swinL_imagenet21k": "omnivore-swinL-in21k",
    }

    if model_name:
        convert_weight_and_push(names_to_config[model_name], model_name, save_directory, names_to_save_names)
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(names_to_config[model_name], model_name, save_directory, names_to_save_names)
    return config, expected_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="The name of the model you wish to convert, it must be one of the supported Omnivore* architecture,",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="omnivore_dump/",
        type=Path,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    args = parser.parse_args()
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name)
