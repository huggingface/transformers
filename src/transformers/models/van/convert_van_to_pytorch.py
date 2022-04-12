# coding=utf-8
# Copyright 2022 BNRist (Tsinghua University), TKLNDST (Nankai University) and The HuggingFace Inc. team. All rights reserved.
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
"""Convert VAN checkpoints from the original repository.

URL: https://github.com/Visual-Attention-Network/VAN-Classification"""


import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from huggingface_hub import cached_download, hf_hub_download
from transformers import AutoFeatureExtractor, VanConfig, VanForImageClassification
from transformers.models.van.modeling_van import VanLayerScaling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


@dataclass
class Tracker:
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            if not isinstance(m, VanLayerScaling):
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


def copy_parameters(from_model: nn.Module, our_model: nn.Module) -> nn.Module:
    # nn.Parameter cannot be tracked by the Tracker, thus we need to manually convert them
    from_state_dict = from_model.state_dict()
    our_state_dict = our_model.state_dict()
    config = our_model.config
    all_keys = []
    for stage_idx in range(len(config.hidden_sizes)):
        for block_id in range(config.depths[stage_idx]):
            from_key = f"block{stage_idx + 1}.{block_id}.layer_scale_1"
            to_key = f"van.encoder.stages.{stage_idx}.layers.{block_id}.attention_scaling.weight"

            all_keys.append((from_key, to_key))
            from_key = f"block{stage_idx + 1}.{block_id}.layer_scale_2"
            to_key = f"van.encoder.stages.{stage_idx}.layers.{block_id}.mlp_scaling.weight"

            all_keys.append((from_key, to_key))

    for from_key, to_key in all_keys:
        our_state_dict[to_key] = from_state_dict.pop(from_key)

    our_model.load_state_dict(our_state_dict)
    return our_model


def convert_weight_and_push(
    name: str,
    config: VanConfig,
    checkpoint: str,
    from_model: nn.Module,
    save_directory: Path,
    push_to_hub: bool = True,
):
    print(f"Downloading weights for {name}...")
    checkpoint_path = cached_download(checkpoint)
    print(f"Converting {name}...")
    from_state_dict = torch.load(checkpoint_path)["state_dict"]
    from_model.load_state_dict(from_state_dict)
    from_model.eval()
    with torch.no_grad():
        our_model = VanForImageClassification(config).eval()
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)
        our_model = copy_parameters(from_model, our_model)

    if not torch.allclose(from_model(x), our_model(x).logits):
        raise ValueError("The model logits don't match the original one.")

    checkpoint_name = name
    print(checkpoint_name)

    if push_to_hub:
        our_model.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # we can use the convnext one
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-base-224-22k-1k")
        feature_extractor.push_to_hub(
            repo_path_or_name=save_directory / checkpoint_name,
            commit_message="Add feature extractor",
            use_temp_dir=True,
        )

        print(f"Pushed {checkpoint_name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000

    repo_id = "datasets/huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    ImageNetPreTrainedConfig = partial(VanConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_config = {
        "van-tiny": ImageNetPreTrainedConfig(
            hidden_sizes=[32, 64, 160, 256],
            depths=[3, 3, 5, 2],
            mlp_ratios=[8, 8, 4, 4],
        ),
        "van-small": ImageNetPreTrainedConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
        ),
        "van-base": ImageNetPreTrainedConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 3, 12, 3],
            mlp_ratios=[8, 8, 4, 4],
        ),
        "van-large": ImageNetPreTrainedConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 5, 27, 3],
            mlp_ratios=[8, 8, 4, 4],
        ),
    }

    names_to_original_models = {
        "van-tiny": van_tiny,
        "van-small": van_small,
        "van-base": van_base,
        "van-large": van_large,
    }

    names_to_original_checkpoints = {
        "van-tiny": "https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar",
        "van-small": "https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar",
        "van-base": "https://huggingface.co/Visual-Attention-Network/VAN-Base-original/resolve/main/van_base_828.pth.tar",
        "van-large": "https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar",
    }

    if model_name:
        convert_weight_and_push(
            model_name,
            names_to_config[model_name],
            checkpoint=names_to_original_checkpoints[model_name],
            from_model=names_to_original_models[model_name](),
            save_directory=save_directory,
            push_to_hub=push_to_hub,
        )
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(
                model_name,
                config,
                checkpoint=names_to_original_checkpoints[model_name],
                from_model=names_to_original_models[model_name](),
                save_directory=save_directory,
                push_to_hub=push_to_hub,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default=None,
        type=str,
        help="The name of the model you wish to convert, it must be one of the supported resnet* architecture, currently: van-tiny/small/base/large. If `None`, all of them will the converted.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--van_dir",
        required=True,
        type=Path,
        help="A path to VAN's original implementation directory. You can download from here: https://github.com/Visual-Attention-Network/VAN-Classification",
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
    van_dir = args.van_dir
    # append the path to the parents to maskformer dir
    sys.path.append(str(van_dir.parent))
    from van.models.van import van_base, van_large, van_small, van_tiny

    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
