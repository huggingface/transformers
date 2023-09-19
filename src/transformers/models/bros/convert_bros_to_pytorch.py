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
"""Convert Bros checkpoints."""

import argparse

import bros  # original repo
import torch

from transformers import BrosConfig, BrosModel, BrosProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_configs(model_name):
    bros_config = BrosConfig.from_pretrained(model_name)
    return bros_config


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "embeddings.bbox_sinusoid_emb.inv_freq",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    if name == "embeddings.bbox_projection.weight":
        name = "bbox_embeddings.bbox_projection.weight"

    if name == "embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq"

    if name == "embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq"

    return name


def convert_state_dict(orig_state_dict, model):
    # rename keys
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        orig_state_dict[rename_key(key)] = val

    # remove ignore keys
    remove_ignore_keys_(orig_state_dict)

    return orig_state_dict


def convert_bros_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    # load original model
    original_model = bros.BrosModel.from_pretrained(model_name).eval()

    # load HuggingFace Model
    bros_config = get_configs(model_name)
    model = BrosModel.from_pretrained(model_name, config=bros_config)
    model.eval()

    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # verify results

    # original BROS model require 4 points (8 float values) for each bbox, prepare bbox with [batch_size, seq_len, 8] shape
    bbox = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.4396, 0.6720, 0.4659, 0.6720, 0.4659, 0.6850, 0.4396, 0.6850],
                [0.4698, 0.6720, 0.4843, 0.6720, 0.4843, 0.6850, 0.4698, 0.6850],
                [0.4698, 0.6720, 0.4843, 0.6720, 0.4843, 0.6850, 0.4698, 0.6850],
                [0.2047, 0.6870, 0.2730, 0.6870, 0.2730, 0.7000, 0.2047, 0.7000],
                [0.2047, 0.6870, 0.2730, 0.6870, 0.2730, 0.7000, 0.2047, 0.7000],
                [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            ]
        ]
    )

    processor = BrosProcessor.from_pretrained(model_name)

    encoding = processor("His name is Rocco.", return_tensors="pt")
    encoding["bbox"] = bbox

    original_hidden_states = original_model(**encoding).last_hidden_state
    # pixel_values = processor(image, return_tensors="pt").pixel_values

    last_hidden_states = model(**encoding).last_hidden_state

    assert torch.allclose(original_hidden_states, last_hidden_states, atol=1e-4)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub("jinho8345/" + model_name.split("/")[-1], commit_message="Update model")
        processor.push_to_hub("jinho8345/" + model_name.split("/")[-1], commit_message="Update model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="naver-clova-ocr/bros-base-uncased",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_bros_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
