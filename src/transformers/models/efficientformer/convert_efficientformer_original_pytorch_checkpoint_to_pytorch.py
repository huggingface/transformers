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

from transformers import (
    EfficientFormerConfig,
    EfficientFormerForImageClassificationWithTeacher,
    EfficientFormerImageProcessor,
)


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
        match = re.search("\d\.\d.", old_name).group()
        if int(match[0]) < 6:
            trimmed_name = old_name.replace(match, "")
            trimmed_name = trimmed_name.replace("network", match[0] + ".meta4D_layers.blocks." + match[2])
            new_name = "intermediate_stages." + trimmed_name
        else:
            trimmed_name = old_name.replace(match, "")
            if int(match[2]) < 4:
                trimmed_name = trimmed_name.replace("network", "meta4D_layers.blocks." + match[2])
            else:
                layer_index = str(int(match[2]) - 4)
                trimmed_name = trimmed_name.replace("network", "meta3D_layers.blocks." + layer_index)
                if "norm1" in old_name:
                    trimmed_name = trimmed_name.replace("norm1", "layernorm")
                elif "norm2" in old_name:
                    trimmed_name = trimmed_name.replace("norm2", "layernorm")
                elif "fc1" in old_name:
                    trimmed_name = trimmed_name.replace("fc1", "linear_in")
                elif "fc2" in old_name:
                    trimmed_name = trimmed_name.replace("fc2", "linear_out")

            new_name = "last_stage." + trimmed_name

    elif "network" in old_name and re.search(".\d.", old_name):
        new_name = old_name.replace("network", "intermediate_stages")

    if "fc" in new_name:
        new_name = new_name.replace("fc", "convolution")
    elif "norm1" in new_name:
        new_name = new_name.replace("norm1", "batchnorm_before")
    elif "norm2" in new_name:
        new_name = new_name.replace("norm2", "batchnorm_after")
    if "proj" in new_name:
        new_name = new_name.replace("proj", "projection")
    if "dist_head" in new_name:
        new_name = new_name.replace("dist_head", "distillation_classifier")
    elif "head" in new_name:
        new_name = new_name.replace("head", "classifier")
    elif "patch_embed" in new_name:
        new_name = "efficientformer." + new_name
    elif new_name == "norm.weight" or new_name == "norm.bias":
        new_name = new_name.replace("norm", "layernorm")
        new_name = "efficientformer." + new_name
    else:
        new_name = "efficientformer.encoder." + new_name

    return new_name


def convert_torch_checkpoint(checkpoint):
    for key in checkpoint.copy().keys():
        val = checkpoint.pop(key)
        checkpoint[rename_key(key)] = val

    return checkpoint


def convert_efficientformer_checkpoint(
    checkpoint_path: Path, efficientformer_config_file: Path, pytorch_dump_path: Path, push_to_hub: bool
):
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    config = EfficientFormerConfig.from_json_file(efficientformer_config_file)
    model = EfficientFormerForImageClassificationWithTeacher(config)

    new_state_dict = convert_torch_checkpoint(orig_state_dict)

    model.load_state_dict(new_state_dict)
    model.eval()

    feature_extractor = EfficientFormerImageProcessor(size=config.input_size[-1])

    # Save Checkpoints
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_path)
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
    feature_extractor.save_pretrained(pytorch_dump_path)
    print(f"Feature extractor successfuly saved at {pytorch_dump_path}")

    if push_to_hub:
        print("Pushing model to the hub...")

        model.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add model",
            use_temp_dir=True,
        )
        feature_extractor.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
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

    parser.add_argument("--push_to_hub", action="store_true", help="Push model and feature extractor to the hub")
    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and feature extractor to the hub",
    )
    parser.set_defaults(push_to_hub=True)

    args = parser.parse_args()
    convert_efficientformer_checkpoint(
        checkpoint_path=args.pytorch_model_path,
        efficientformer_config_file=args.config_file,
        pytorch_dump_path=args.pytorch_dump_path,
        push_to_hub=args.push_to_hub,
    )
