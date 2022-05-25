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
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from levit import LeViT_192, LeViT_128, LeViT_128S, LeViT_256, LeViT_384
from huggingface_hub import hf_hub_download
from transformers import LevitFeatureExtractor, LevitConfig, LevitForImageClassification
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
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")


def convert_weight_and_push(embed_dim: int, name: str, config: LevitConfig, save_directory: Path, push_to_hub: bool = True):
    print(f"Converting {name}...")
    with torch.no_grad():
        if embed_dim == 128:
            if name[-1] == "S":
                from_model = LeViT_128S().eval()
                weights = "https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth"
                checkpoint = torch.hub.load_state_dict_from_url( weights, map_location='cpu')
                from_model.load_state_dict(checkpoint['model'])
            else: 
                from_model = LeViT_128().eval()
                weights = "https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth"
                checkpoint = torch.hub.load_state_dict_from_url( weights, map_location='cpu')
                from_model.load_state_dict(checkpoint['model'])
        if embed_dim == 192:
            from_model = LeViT_192().eval()
            weights = "https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth"
            checkpoint = torch.hub.load_state_dict_from_url( weights, map_location='cpu')
            from_model.load_state_dict(checkpoint['model'])
        if embed_dim == 256:
            from_model = LeViT_256().eval()
            weights = "https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth"
            checkpoint = torch.hub.load_state_dict_from_url( weights, map_location='cpu')
            from_model.load_state_dict(checkpoint['model'])
        if embed_dim == 384:
            from_model = LeViT_384().eval()
            weights = "https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth"
            checkpoint = torch.hub.load_state_dict_from_url( weights, map_location='cpu')
            from_model.load_state_dict(checkpoint['model'])



        our_model = LevitForImageClassification(config).eval()
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        x = torch.randn((2, 3, 224, 224))
        print(x.shape)
        module_transfer(x)

    assert torch.allclose(from_model(x), our_model(x).logits), "The model logits don't match the original one."

    checkpoint_name = name
    print(checkpoint_name)

    if push_to_hub:
        our_model.save_pretrained(save_directory / checkpoint_name)

        # we can use the convnext one
        feature_extractor = LevitFeatureExtractor.from_pretrained("anugunj/levit-384")
        feature_extractor.save_pretrained(save_directory / checkpoint_name)

        print(f"Pushed {checkpoint_name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    repo_id = "datasets/huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    ImageNetPreTrainedConfig = partial(LevitConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_embed_dim = {
        "levit-128S": 128,
        "levit-128": 128,
        "levit-192": 192,
        "levit-256": 256,
        "levit-384": 384,
    }

    names_to_config = {
        "levit-128S": ImageNetPreTrainedConfig(
            embed_dim = [128, 256, 384], num_heads = [4, 6, 8], depth = [2, 3, 4], key_dim = [16, 16, 16],
            drop_path_rate = 0
        ),
        "levit-128": ImageNetPreTrainedConfig(
            embed_dim = [128, 256, 384], num_heads = [4, 8, 12], depth = [4, 4, 4], key_dim = [16, 16, 16],
            drop_path_rate = 0
        ),
        "levit-192": ImageNetPreTrainedConfig(
            embed_dim = [192, 288, 384], num_heads = [3, 5, 6], depth = [4, 4, 4], key_dim = [32, 32, 32],
            drop_path_rate = 0
        ),
        "levit-256": ImageNetPreTrainedConfig(
            embed_dim = [256, 384, 512], num_heads = [4, 6, 8], depth = [4, 4, 4], key_dim = [32, 32, 32],
            drop_path_rate = 0
        ),
        "levit-384": ImageNetPreTrainedConfig(
            embed_dim = [384, 512, 768], num_heads = [6, 9, 12], depth = [4, 4, 4], key_dim = [32, 32, 32],
            drop_path_rate = 0.1
        ),
    }

    if model_name:
        convert_weight_and_push(names_to_embed_dim[model_name], model_name, names_to_config[model_name], save_directory, push_to_hub)
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(names_to_embed_dim[model_name], model_name, config, save_directory, push_to_hub)
    return config, expected_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported Levit* architecture,"
        ),
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
