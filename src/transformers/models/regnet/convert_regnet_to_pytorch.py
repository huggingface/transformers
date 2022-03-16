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
"""Convert RegNet checkpoints from timm."""


import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

import timm
from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoFeatureExtractor, RegNetConfig, RegNetForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger()


@dataclass
class Tracker:
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")


def convert_weight_and_push(
    name: str, from_name: str, config: RegNetConfig, save_directory: Path, push_to_hub: bool = True
):
    print(f"Converting {name}...")
    with torch.no_grad():
        from_model = timm.create_model(from_name, pretrained=True).eval()
        our_model = RegNetForImageClassification(config).eval()
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)

    assert torch.allclose(from_model(x), our_model(x).logits), "The model logits don't match the original one."

    if push_to_hub:
        our_model.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # we can use the convnext one
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-base-224-22k-1k")
        feature_extractor.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add feature extractor",
            use_temp_dir=True,
        )

        print(f"Pushed {name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    repo_id = "datasets/huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_config = {
        "regnetx-002": ImageNetPreTrainedConfig(
            depths=[1, 1, 4, 7], hidden_sizes=[24, 56, 152, 368], groups_width=8, layer_type="x"
        ),
        "regnetx-004": ImageNetPreTrainedConfig(
            depths=[1, 2, 7, 12], hidden_sizes=[32, 64, 160, 384], groups_width=16, layer_type="x"
        ),
        "regnetx-006": ImageNetPreTrainedConfig(
            depths=[1, 3, 5, 7], hidden_sizes=[48, 96, 240, 528], groups_width=24, layer_type="x"
        ),
        "regnetx-008": ImageNetPreTrainedConfig(
            depths=[1, 3, 7, 5], hidden_sizes=[64, 128, 288, 672], groups_width=16, layer_type="x"
        ),
        "regnetx-016": ImageNetPreTrainedConfig(
            depths=[2, 4, 10, 2], hidden_sizes=[72, 168, 408, 912], groups_width=24, layer_type="x"
        ),
        "regnetx-032": ImageNetPreTrainedConfig(
            depths=[2, 6, 15, 2], hidden_sizes=[96, 192, 432, 1008], groups_width=48, layer_type="x"
        ),
        "regnetx-040": ImageNetPreTrainedConfig(
            depths=[2, 5, 14, 2], hidden_sizes=[80, 240, 560, 1360], groups_width=40, layer_type="x"
        ),
        "regnetx-064": ImageNetPreTrainedConfig(
            depths=[2, 4, 10, 1], hidden_sizes=[168, 392, 784, 1624], groups_width=56, layer_type="x"
        ),
        "regnetx-080": ImageNetPreTrainedConfig(
            depths=[2, 5, 15, 1], hidden_sizes=[80, 240, 720, 1920], groups_width=120, layer_type="x"
        ),
        "regnetx-120": ImageNetPreTrainedConfig(
            depths=[2, 5, 11, 1], hidden_sizes=[224, 448, 896, 2240], groups_width=112, layer_type="x"
        ),
        "regnetx-160": ImageNetPreTrainedConfig(
            depths=[2, 6, 13, 1], hidden_sizes=[256, 512, 896, 2048], groups_width=128, layer_type="x"
        ),
        "regnetx-320": ImageNetPreTrainedConfig(
            depths=[2, 7, 13, 1], hidden_sizes=[336, 672, 1344, 2520], groups_width=168, layer_type="x"
        ),
        # y variant
        "regnety-002": ImageNetPreTrainedConfig(depths=[1, 1, 4, 7], hidden_sizes=[24, 56, 152, 368], groups_width=8),
        "regnety-004": ImageNetPreTrainedConfig(depths=[1, 3, 6, 6], hidden_sizes=[48, 104, 208, 440], groups_width=8),
        "regnety-006": ImageNetPreTrainedConfig(
            depths=[1, 3, 7, 4], hidden_sizes=[48, 112, 256, 608], groups_width=16
        ),
        "regnety-008": ImageNetPreTrainedConfig(
            depths=[1, 3, 8, 2], hidden_sizes=[64, 128, 320, 768], groups_width=16
        ),
        "regnety-016": ImageNetPreTrainedConfig(
            depths=[2, 6, 17, 2], hidden_sizes=[48, 120, 336, 888], groups_width=24
        ),
        "regnety-032": ImageNetPreTrainedConfig(
            depths=[2, 5, 13, 1], hidden_sizes=[72, 216, 576, 1512], groups_width=24
        ),
        "regnety-040": ImageNetPreTrainedConfig(
            depths=[2, 6, 12, 2], hidden_sizes=[128, 192, 512, 1088], groups_width=64
        ),
        "regnety-064": ImageNetPreTrainedConfig(
            depths=[2, 7, 14, 2], hidden_sizes=[144, 288, 576, 1296], groups_width=72
        ),
        "regnety-080": ImageNetPreTrainedConfig(
            depths=[2, 4, 10, 1], hidden_sizes=[168, 448, 896, 2016], groups_width=56
        ),
        "regnety-120": ImageNetPreTrainedConfig(
            depths=[2, 5, 11, 1], hidden_sizes=[224, 448, 896, 2240], groups_width=112
        ),
        "regnety-160": ImageNetPreTrainedConfig(
            depths=[2, 4, 11, 1], hidden_sizes=[224, 448, 1232, 3024], groups_width=112
        ),
        "regnety-320": ImageNetPreTrainedConfig(
            depths=[2, 5, 12, 1], hidden_sizes=[232, 696, 1392, 3712], groups_width=232
        ),
    }

    names_to_timm = {k: k.replace("-", "_") for k in names_to_config.keys()}

    if model_name:
        convert_weight_and_push(
            model_name, names_to_timm[model_name], names_to_config[model_name], save_directory, push_to_hub
        )
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(model_name, names_to_timm[model_name], config, save_directory, push_to_hub)
    return config, expected_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="The name of the model you wish to convert, it must be one of the supported regnet* architecture, currently: regnetx-*, regnety-*. If `None`, all of them will the converted.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and feature extractor to the hub.",
    )

    args = parser.parse_args()
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
