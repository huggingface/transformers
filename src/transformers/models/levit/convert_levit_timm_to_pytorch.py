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
"""Convert LeViT checkpoints from timm."""

import argparse
import json
from collections import OrderedDict
from functools import partial
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download

from transformers import LevitConfig, LevitForImageClassificationWithTeacher, LevitImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger()


def convert_weight_and_push(
    hidden_sizes: int, name: str, config: LevitConfig, save_directory: Path, push_to_hub: bool = True
):
    print(f"Converting {name}...")

    with torch.no_grad():
        if hidden_sizes == 128:
            if name[-1] == "S":
                from_model = timm.create_model("levit_128s", pretrained=True)
            else:
                from_model = timm.create_model("levit_128", pretrained=True)
        if hidden_sizes == 192:
            from_model = timm.create_model("levit_192", pretrained=True)
        if hidden_sizes == 256:
            from_model = timm.create_model("levit_256", pretrained=True)
        if hidden_sizes == 384:
            from_model = timm.create_model("levit_384", pretrained=True)

        from_model.eval()
        our_model = LevitForImageClassificationWithTeacher(config).eval()
        huggingface_weights = OrderedDict()

        weights = from_model.state_dict()
        og_keys = list(from_model.state_dict().keys())
        new_keys = list(our_model.state_dict().keys())
        print(len(og_keys), len(new_keys))
        for i in range(len(og_keys)):
            huggingface_weights[new_keys[i]] = weights[og_keys[i]]
        our_model.load_state_dict(huggingface_weights)

        x = torch.randn((2, 3, 224, 224))
        out1 = from_model(x)
        out2 = our_model(x).logits

    assert torch.allclose(out1, out2), "The model logits don't match the original one."

    checkpoint_name = name
    print(checkpoint_name)

    if push_to_hub:
        our_model.save_pretrained(save_directory / checkpoint_name)
        image_processor = LevitImageProcessor()
        image_processor.save_pretrained(save_directory / checkpoint_name)

        print(f"Pushed {checkpoint_name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    ImageNetPreTrainedConfig = partial(LevitConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_hidden_sizes = {
        "levit-128S": 128,
        "levit-128": 128,
        "levit-192": 192,
        "levit-256": 256,
        "levit-384": 384,
    }

    names_to_config = {
        "levit-128S": ImageNetPreTrainedConfig(
            hidden_sizes=[128, 256, 384],
            num_attention_heads=[4, 6, 8],
            depths=[2, 3, 4],
            key_dim=[16, 16, 16],
            drop_path_rate=0,
        ),
        "levit-128": ImageNetPreTrainedConfig(
            hidden_sizes=[128, 256, 384],
            num_attention_heads=[4, 8, 12],
            depths=[4, 4, 4],
            key_dim=[16, 16, 16],
            drop_path_rate=0,
        ),
        "levit-192": ImageNetPreTrainedConfig(
            hidden_sizes=[192, 288, 384],
            num_attention_heads=[3, 5, 6],
            depths=[4, 4, 4],
            key_dim=[32, 32, 32],
            drop_path_rate=0,
        ),
        "levit-256": ImageNetPreTrainedConfig(
            hidden_sizes=[256, 384, 512],
            num_attention_heads=[4, 6, 8],
            depths=[4, 4, 4],
            key_dim=[32, 32, 32],
            drop_path_rate=0,
        ),
        "levit-384": ImageNetPreTrainedConfig(
            hidden_sizes=[384, 512, 768],
            num_attention_heads=[6, 9, 12],
            depths=[4, 4, 4],
            key_dim=[32, 32, 32],
            drop_path_rate=0.1,
        ),
    }

    if model_name:
        convert_weight_and_push(
            names_to_hidden_sizes[model_name], model_name, names_to_config[model_name], save_directory, push_to_hub
        )
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(names_to_hidden_sizes[model_name], model_name, config, save_directory, push_to_hub)
    return config, expected_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="The name of the model you wish to convert, it must be one of the supported Levit* architecture,",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="levit-dump-folder/",
        type=Path,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and image processor to the hub",
    )

    args = parser.parse_args()
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
