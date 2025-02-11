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
"""Convert RegNet checkpoints from timm and vissl."""

import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import hf_hub_download
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs

from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
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
        [x.remove() for x in self.handles]
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 1
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)
    raise_if_mismatch: bool = True

    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced) and self.raise_if_mismatch:
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")


class FakeRegNetVisslWrapper(nn.Module):
    """
    Fake wrapper for RegNet that mimics what vissl does without the need to pass a config file.
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        feature_blocks: List[Tuple[str, nn.Module]] = []
        # - get the stem
        feature_blocks.append(("conv1", model.stem))
        # - get all the feature blocks
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x: Tensor):
        return get_trunk_forward_outputs(
            x,
            out_feat_keys=None,
            feature_blocks=self._feature_blocks,
        )


class NameToFromModelFuncMap(dict):
    """
    A Dictionary with some additional logic to return a function that creates the correct original model.
    """

    def convert_name_to_timm(self, x: str) -> str:
        x_split = x.split("-")
        return x_split[0] + x_split[1] + "_" + "".join(x_split[2:])

    def __getitem__(self, x: str) -> Callable[[], Tuple[nn.Module, Dict]]:
        # default to timm!
        if x not in self:
            x = self.convert_name_to_timm(x)
            val = partial(lambda: (timm.create_model(x, pretrained=True).eval(), None))

        else:
            val = super().__getitem__(x)

        return val


class NameToOurModelFuncMap(dict):
    """
    A Dictionary with some additional logic to return the correct hugging face RegNet class reference.
    """

    def __getitem__(self, x: str) -> Callable[[], nn.Module]:
        if "seer" in x and "in1k" not in x:
            val = RegNetModel
        else:
            val = RegNetForImageClassification
        return val


def manually_copy_vissl_head(from_state_dict, to_state_dict, keys: List[Tuple[str, str]]):
    for from_key, to_key in keys:
        to_state_dict[to_key] = from_state_dict[from_key].clone()
        print(f"Copied key={from_key} to={to_key}")
    return to_state_dict


def convert_weight_and_push(
    name: str,
    from_model_func: Callable[[], nn.Module],
    our_model_func: Callable[[], nn.Module],
    config: RegNetConfig,
    save_directory: Path,
    push_to_hub: bool = True,
):
    print(f"Converting {name}...")
    with torch.no_grad():
        from_model, from_state_dict = from_model_func()
        our_model = our_model_func(config).eval()
        module_transfer = ModuleTransfer(src=from_model, dest=our_model, raise_if_mismatch=False)
        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)

    if from_state_dict is not None:
        keys = []
        # for seer - in1k finetuned we have to manually copy the head
        if "seer" in name and "in1k" in name:
            keys = [("0.clf.0.weight", "classifier.1.weight"), ("0.clf.0.bias", "classifier.1.bias")]
        to_state_dict = manually_copy_vissl_head(from_state_dict, our_model.state_dict(), keys)
        our_model.load_state_dict(to_state_dict)

    our_outputs = our_model(x, output_hidden_states=True)
    our_output = (
        our_outputs.logits if isinstance(our_model, RegNetForImageClassification) else our_outputs.last_hidden_state
    )

    from_output = from_model(x)
    from_output = from_output[-1] if isinstance(from_output, list) else from_output

    # now since I don't want to use any config files, vissl seer model doesn't actually have an head, so let's just check the last hidden state
    if "seer" in name and "in1k" in name:
        our_output = our_outputs.hidden_states[-1]

    assert torch.allclose(from_output, our_output), "The model logits don't match the original one."

    if push_to_hub:
        our_model.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        size = 224 if "seer" not in name else 384
        # we can use the convnext one
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k", size=size)
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        print(f"Pushed {name}")


def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_config = {
        "regnet-x-002": ImageNetPreTrainedConfig(
            depths=[1, 1, 4, 7], hidden_sizes=[24, 56, 152, 368], groups_width=8, layer_type="x"
        ),
        "regnet-x-004": ImageNetPreTrainedConfig(
            depths=[1, 2, 7, 12], hidden_sizes=[32, 64, 160, 384], groups_width=16, layer_type="x"
        ),
        "regnet-x-006": ImageNetPreTrainedConfig(
            depths=[1, 3, 5, 7], hidden_sizes=[48, 96, 240, 528], groups_width=24, layer_type="x"
        ),
        "regnet-x-008": ImageNetPreTrainedConfig(
            depths=[1, 3, 7, 5], hidden_sizes=[64, 128, 288, 672], groups_width=16, layer_type="x"
        ),
        "regnet-x-016": ImageNetPreTrainedConfig(
            depths=[2, 4, 10, 2], hidden_sizes=[72, 168, 408, 912], groups_width=24, layer_type="x"
        ),
        "regnet-x-032": ImageNetPreTrainedConfig(
            depths=[2, 6, 15, 2], hidden_sizes=[96, 192, 432, 1008], groups_width=48, layer_type="x"
        ),
        "regnet-x-040": ImageNetPreTrainedConfig(
            depths=[2, 5, 14, 2], hidden_sizes=[80, 240, 560, 1360], groups_width=40, layer_type="x"
        ),
        "regnet-x-064": ImageNetPreTrainedConfig(
            depths=[2, 4, 10, 1], hidden_sizes=[168, 392, 784, 1624], groups_width=56, layer_type="x"
        ),
        "regnet-x-080": ImageNetPreTrainedConfig(
            depths=[2, 5, 15, 1], hidden_sizes=[80, 240, 720, 1920], groups_width=120, layer_type="x"
        ),
        "regnet-x-120": ImageNetPreTrainedConfig(
            depths=[2, 5, 11, 1], hidden_sizes=[224, 448, 896, 2240], groups_width=112, layer_type="x"
        ),
        "regnet-x-160": ImageNetPreTrainedConfig(
            depths=[2, 6, 13, 1], hidden_sizes=[256, 512, 896, 2048], groups_width=128, layer_type="x"
        ),
        "regnet-x-320": ImageNetPreTrainedConfig(
            depths=[2, 7, 13, 1], hidden_sizes=[336, 672, 1344, 2520], groups_width=168, layer_type="x"
        ),
        # y variant
        "regnet-y-002": ImageNetPreTrainedConfig(depths=[1, 1, 4, 7], hidden_sizes=[24, 56, 152, 368], groups_width=8),
        "regnet-y-004": ImageNetPreTrainedConfig(
            depths=[1, 3, 6, 6], hidden_sizes=[48, 104, 208, 440], groups_width=8
        ),
        "regnet-y-006": ImageNetPreTrainedConfig(
            depths=[1, 3, 7, 4], hidden_sizes=[48, 112, 256, 608], groups_width=16
        ),
        "regnet-y-008": ImageNetPreTrainedConfig(
            depths=[1, 3, 8, 2], hidden_sizes=[64, 128, 320, 768], groups_width=16
        ),
        "regnet-y-016": ImageNetPreTrainedConfig(
            depths=[2, 6, 17, 2], hidden_sizes=[48, 120, 336, 888], groups_width=24
        ),
        "regnet-y-032": ImageNetPreTrainedConfig(
            depths=[2, 5, 13, 1], hidden_sizes=[72, 216, 576, 1512], groups_width=24
        ),
        "regnet-y-040": ImageNetPreTrainedConfig(
            depths=[2, 6, 12, 2], hidden_sizes=[128, 192, 512, 1088], groups_width=64
        ),
        "regnet-y-064": ImageNetPreTrainedConfig(
            depths=[2, 7, 14, 2], hidden_sizes=[144, 288, 576, 1296], groups_width=72
        ),
        "regnet-y-080": ImageNetPreTrainedConfig(
            depths=[2, 4, 10, 1], hidden_sizes=[168, 448, 896, 2016], groups_width=56
        ),
        "regnet-y-120": ImageNetPreTrainedConfig(
            depths=[2, 5, 11, 1], hidden_sizes=[224, 448, 896, 2240], groups_width=112
        ),
        "regnet-y-160": ImageNetPreTrainedConfig(
            depths=[2, 4, 11, 1], hidden_sizes=[224, 448, 1232, 3024], groups_width=112
        ),
        "regnet-y-320": ImageNetPreTrainedConfig(
            depths=[2, 5, 12, 1], hidden_sizes=[232, 696, 1392, 3712], groups_width=232
        ),
        # models created by SEER -> https://arxiv.org/abs/2202.08360
        "regnet-y-320-seer": RegNetConfig(depths=[2, 5, 12, 1], hidden_sizes=[232, 696, 1392, 3712], groups_width=232),
        "regnet-y-640-seer": RegNetConfig(depths=[2, 5, 12, 1], hidden_sizes=[328, 984, 1968, 4920], groups_width=328),
        "regnet-y-1280-seer": RegNetConfig(
            depths=[2, 7, 17, 1], hidden_sizes=[528, 1056, 2904, 7392], groups_width=264
        ),
        "regnet-y-2560-seer": RegNetConfig(
            depths=[3, 7, 16, 1], hidden_sizes=[640, 1696, 2544, 5088], groups_width=640
        ),
        "regnet-y-10b-seer": ImageNetPreTrainedConfig(
            depths=[2, 7, 17, 1], hidden_sizes=[2020, 4040, 11110, 28280], groups_width=1010
        ),
        # finetuned on imagenet
        "regnet-y-320-seer-in1k": ImageNetPreTrainedConfig(
            depths=[2, 5, 12, 1], hidden_sizes=[232, 696, 1392, 3712], groups_width=232
        ),
        "regnet-y-640-seer-in1k": ImageNetPreTrainedConfig(
            depths=[2, 5, 12, 1], hidden_sizes=[328, 984, 1968, 4920], groups_width=328
        ),
        "regnet-y-1280-seer-in1k": ImageNetPreTrainedConfig(
            depths=[2, 7, 17, 1], hidden_sizes=[528, 1056, 2904, 7392], groups_width=264
        ),
        "regnet-y-2560-seer-in1k": ImageNetPreTrainedConfig(
            depths=[3, 7, 16, 1], hidden_sizes=[640, 1696, 2544, 5088], groups_width=640
        ),
        "regnet-y-10b-seer-in1k": ImageNetPreTrainedConfig(
            depths=[2, 7, 17, 1], hidden_sizes=[2020, 4040, 11110, 28280], groups_width=1010
        ),
    }

    names_to_ours_model_map = NameToOurModelFuncMap()
    names_to_from_model_map = NameToFromModelFuncMap()
    # add seer weights logic

    def load_using_classy_vision(checkpoint_url: str, model_func: Callable[[], nn.Module]) -> Tuple[nn.Module, Dict]:
        files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location="cpu")
        model = model_func()
        # check if we have a head, if yes add it
        model_state_dict = files["classy_state_dict"]["base_model"]["model"]
        state_dict = model_state_dict["trunk"]
        model.load_state_dict(state_dict)
        return model.eval(), model_state_dict["heads"]

    # pretrained
    names_to_from_model_map["regnet-y-320-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY32gf()),
    )

    names_to_from_model_map["regnet-y-640-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY64gf()),
    )

    names_to_from_model_map["regnet-y-1280-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY128gf()),
    )

    names_to_from_model_map["regnet-y-10b-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet10B/model_iteration124500_conso.torch",
        lambda: FakeRegNetVisslWrapper(
            RegNet(RegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
        ),
    )

    # IN1K finetuned
    names_to_from_model_map["regnet-y-320-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY32gf()),
    )

    names_to_from_model_map["regnet-y-640-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY64gf()),
    )

    names_to_from_model_map["regnet-y-1280-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY128gf()),
    )

    names_to_from_model_map["regnet-y-10b-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_10b_finetuned_in1k_model_phase28_conso.torch",
        lambda: FakeRegNetVisslWrapper(
            RegNet(RegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
        ),
    )

    if model_name:
        convert_weight_and_push(
            model_name,
            names_to_from_model_map[model_name],
            names_to_ours_model_map[model_name],
            names_to_config[model_name],
            save_directory,
            push_to_hub,
        )
    else:
        for model_name, config in names_to_config.items():
            convert_weight_and_push(
                model_name,
                names_to_from_model_map[model_name],
                names_to_ours_model_map[model_name],
                config,
                save_directory,
                push_to_hub,
            )
    return config, expected_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported regnet* architecture,"
            " currently: regnetx-*, regnety-*. If `None`, all of them will the converted."
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
        help="If True, push model and image processor to the hub.",
    )

    args = parser.parse_args()

    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
