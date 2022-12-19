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

"""Convert EfficientFormer checkpoints from the original repository."""

import argparse
import re
from pathlib import Path

import torch

from transformers import EfficientFormerConfig, EfficientFormerFeatureExtractor, EfficientFormerForImageClassification


def rename_key(old_name):
    new_name = old_name

    if "patch_embed" in old_name:
        _, layer, param = old_name.split(".")

        if layer == "0":
            new_name = old_name.replace("0", "convolution1")
        elif layer == "1":
            new_name = old_name.replace("1", "batchnorm_before")
        elif layer == "3":
            new_name = old_name.replace("3", "convolution2")
        else:
            new_name = old_name.replace("4", "batchnorm_after")

    if "network" in old_name and re.search("\d\.\d", old_name):
        match = re.search("\d\.\d", old_name).group()
        new_name = old_name.replace(match, match[:2] + "blocks." + match[2:])

    if "proj" in old_name:
        new_name = new_name.replace("proj", "projection")

    if "dist_head" in new_name:
        pass
    elif "head" in new_name:
        new_name = new_name.replace("head", "classifier")
    elif "patch_embed" in new_name or new_name == "norm.weight" or new_name == "norm.bias":
        new_name = "efficientformer." + new_name
    else:
        new_name = "efficientformer.encoder." + new_name

    return new_name


def convert_torch_checkpoint(checkpoint):
    for key in checkpoint.copy().keys():
        val = checkpoint.pop(key)
        checkpoint[rename_key(key)] = val

    return checkpoint


@torch.no_grad
def convert_efficientformer_checkpoint(checkpoint_path, efficientformer_config_file, pytorch_dump_path):
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    config = EfficientFormerConfig.from_json_file(efficientformer_config_file)
    model = EfficientFormerForImageClassification(config)

    new_state_dict = convert_torch_checkpoint(orig_state_dict)

    model.load_state_dict(new_state_dict)
    model.eval()

    feature_extractor = EfficientFormerFeatureExtractor(config.size)

    # Save Checkpoints
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_path)
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
    feature_extractor.save_pretrained(pytorch_dump_path)
    print(f"Feature extractor successfuly saved at {pytorch_dump_path}")

    print("Pushing model to the hub...")

    model_name = "EfficientFormer"
    repo_name = Path(pytorch_dump_path, model_name)

    model.push_to_hub(
        repo_path_or_name=repo_name,
        commit_message="Add model",
        use_temp_dir=True,
    )
    feature_extractor.push_to_hub(
        repo_path_or_name=repo_name,
        commit_message="Add feature extractor",
        use_temp_dir=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to EfficientFormer pytorch checkpoint.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for EfficientFormer model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_efficientformer_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
