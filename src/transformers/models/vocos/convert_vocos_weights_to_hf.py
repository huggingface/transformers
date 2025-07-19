# coding=utf-8
# Copyright 2024 Descript and The HuggingFace Inc. team. All rights reserved.
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
import io

import torch

from transformers import (
    VocosConfig,
    VocosFeatureExtractor,
    VocosModel,
    logging,
)


def convert_old_keys_to_new_keys(original_state_dict) -> dict[str, torch.Tensor]:
    converted_checkpoint = {}
    for key, value in original_state_dict.items():
        if key.startswith("feature_extractor."):
            continue
        key = key.replace("head.out.", "head.out_proj.")
        key = key.replace("backbone.convnext.", "backbone.layers.")
        key = key.replace(".gamma", ".layer_scale_parameter")
        converted_checkpoint[key] = value
    return converted_checkpoint


def safe_load(path: str) -> dict[str, torch.Tensor]:
    """
    Load only the tensor objects from a checkpoint, skipping any BytesIO
    """
    shard = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in shard.items() if not isinstance(v, io.BytesIO)}


@torch.no_grad()
def convert_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, push_to_hub=None):
    if config_path is not None:
        config = VocosConfig.from_pretrained(config_path)
    else:
        config = VocosConfig()

    with torch.device("meta"):
        model = VocosModel(config)

    original_state_dict = safe_load(checkpoint_path)

    new_state_dict = convert_old_keys_to_new_keys(original_state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False, assign=True)  # , strict=False)

    if len(unexpected_keys) != 0:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    if len(missing_keys) != 0:
        raise ValueError(f"missing keys found: {missing_keys}")

    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)
    feature_extractor = VocosFeatureExtractor()
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        feature_extractor.push_to_hub(push_to_hub)
        logging.info(f"Pushed model to {push_to_hub}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )
