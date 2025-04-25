# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import argparse

import torch

from transformers import UnivNetConfig, UnivNetModel, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.univnet")


def get_kernel_predictor_key_mapping(config: UnivNetConfig, old_prefix: str = "", new_prefix: str = ""):
    mapping = {}
    # Initial conv layer
    mapping[f"{old_prefix}.input_conv.0.weight_g"] = f"{new_prefix}.input_conv.weight_g"
    mapping[f"{old_prefix}.input_conv.0.weight_v"] = f"{new_prefix}.input_conv.weight_v"
    mapping[f"{old_prefix}.input_conv.0.bias"] = f"{new_prefix}.input_conv.bias"

    # Kernel predictor resnet blocks
    for i in range(config.kernel_predictor_num_blocks):
        mapping[f"{old_prefix}.residual_convs.{i}.1.weight_g"] = f"{new_prefix}.resblocks.{i}.conv1.weight_g"
        mapping[f"{old_prefix}.residual_convs.{i}.1.weight_v"] = f"{new_prefix}.resblocks.{i}.conv1.weight_v"
        mapping[f"{old_prefix}.residual_convs.{i}.1.bias"] = f"{new_prefix}.resblocks.{i}.conv1.bias"

        mapping[f"{old_prefix}.residual_convs.{i}.3.weight_g"] = f"{new_prefix}.resblocks.{i}.conv2.weight_g"
        mapping[f"{old_prefix}.residual_convs.{i}.3.weight_v"] = f"{new_prefix}.resblocks.{i}.conv2.weight_v"
        mapping[f"{old_prefix}.residual_convs.{i}.3.bias"] = f"{new_prefix}.resblocks.{i}.conv2.bias"

    # Kernel output conv
    mapping[f"{old_prefix}.kernel_conv.weight_g"] = f"{new_prefix}.kernel_conv.weight_g"
    mapping[f"{old_prefix}.kernel_conv.weight_v"] = f"{new_prefix}.kernel_conv.weight_v"
    mapping[f"{old_prefix}.kernel_conv.bias"] = f"{new_prefix}.kernel_conv.bias"

    # Bias output conv
    mapping[f"{old_prefix}.bias_conv.weight_g"] = f"{new_prefix}.bias_conv.weight_g"
    mapping[f"{old_prefix}.bias_conv.weight_v"] = f"{new_prefix}.bias_conv.weight_v"
    mapping[f"{old_prefix}.bias_conv.bias"] = f"{new_prefix}.bias_conv.bias"

    return mapping


def get_key_mapping(config: UnivNetConfig):
    mapping = {}

    # NOTE: inital conv layer keys are the same

    # LVC Residual blocks
    for i in range(len(config.resblock_stride_sizes)):
        # LVCBlock initial convt layer
        mapping[f"res_stack.{i}.convt_pre.1.weight_g"] = f"resblocks.{i}.convt_pre.weight_g"
        mapping[f"res_stack.{i}.convt_pre.1.weight_v"] = f"resblocks.{i}.convt_pre.weight_v"
        mapping[f"res_stack.{i}.convt_pre.1.bias"] = f"resblocks.{i}.convt_pre.bias"

        # Kernel predictor
        kernel_predictor_mapping = get_kernel_predictor_key_mapping(
            config, old_prefix=f"res_stack.{i}.kernel_predictor", new_prefix=f"resblocks.{i}.kernel_predictor"
        )
        mapping.update(kernel_predictor_mapping)

        # LVC Residual blocks
        for j in range(len(config.resblock_dilation_sizes[i])):
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.weight_g"] = f"resblocks.{i}.resblocks.{j}.conv.weight_g"
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.weight_v"] = f"resblocks.{i}.resblocks.{j}.conv.weight_v"
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.bias"] = f"resblocks.{i}.resblocks.{j}.conv.bias"

    # Output conv layer
    mapping["conv_post.1.weight_g"] = "conv_post.weight_g"
    mapping["conv_post.1.weight_v"] = "conv_post.weight_v"
    mapping["conv_post.1.bias"] = "conv_post.bias"

    return mapping


def rename_state_dict(state_dict, keys_to_modify, keys_to_remove):
    model_state_dict = {}
    for key, value in state_dict.items():
        if key in keys_to_remove:
            continue

        if key in keys_to_modify:
            new_key = keys_to_modify[key]
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    return model_state_dict


def convert_univnet_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
    safe_serialization=False,
):
    model_state_dict_base = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Get the generator's state dict
    state_dict = model_state_dict_base["model_g"]

    if config_path is not None:
        config = UnivNetConfig.from_pretrained(config_path)
    else:
        config = UnivNetConfig()

    keys_to_modify = get_key_mapping(config)
    keys_to_remove = set()
    hf_state_dict = rename_state_dict(state_dict, keys_to_modify, keys_to_remove)

    model = UnivNetModel(config)
    # Apply weight norm since the original checkpoint has weight norm applied
    model.apply_weight_norm()
    model.load_state_dict(hf_state_dict)
    # Remove weight norm in preparation for inference
    model.remove_weight_norm()

    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", help="Whether to save the model using `safetensors`."
    )

    args = parser.parse_args()

    convert_univnet_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
        args.safe_serialization,
    )


if __name__ == "__main__":
    main()
