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
    EncodecModel,
    VocosWithEncodecConfig,
    VocosWithEncodecModel,
)
from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch import recursively_load_weights


BACKBONE_MAPPING = {
    ".gamma": ".layer_scale_parameter",
    ".norm.scale.weight": ".norm.weight",
    ".norm.shift.weight": ".norm.bias",
    "backbone.norm.scale.weight": "backbone.norm.weight",
    "backbone.norm.shift.weight": "backbone.norm.bias",
    "backbone.convnext.*": "backbone.layers.*",
}

HEAD_MAPPING = {
    "head.out.weight": "head.out_proj.weight",
    "head.out.bias": "head.out_proj.bias",
}


def _rewrite_weight_norm(key):
    key = key.replace("weight_g", "parametrizations.weight.original0")
    key = key.replace("weight_v", "parametrizations.weight.original1")
    return key


def _remap_key(key, mapping_dict):
    while True:
        new_key = key
        for old, new in mapping_dict.items():
            if "*" in old:
                prefix, suffix = old.split("*")
                if new_key.startswith(prefix) and new_key.endswith(suffix):
                    idx = new_key[len(prefix) :] if not suffix else new_key[len(prefix) : -len(suffix)]
                    new_key = new.replace("*", idx)
                    break
            elif old in new_key:
                new_key = new_key.replace(old, new)
                break
        if new_key == key:
            return new_key
        key = new_key


def convert_old_keys_to_new_keys(original_state_dict: dict, model_name: str = "encodec_24khz") -> dict:
    converted_checkpoint = {}
    original_encodec = {}
    for old_key, value in original_state_dict.items():
        if not old_key.startswith("feature_extractor.encodec."):
            continue
        encodec_key = old_key[len("feature_extractor.encodec.") :]
        encodec_key = encodec_key.replace(".conv.conv.", ".conv.").replace(".convtr.convtr.", ".conv.")
        original_encodec[_rewrite_weight_norm(encodec_key)] = value

    hf_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()  # {model_name}").eval()
    recursively_load_weights(original_encodec, hf_encodec, model_name)
    for key, value in hf_encodec.state_dict().items():
        converted_checkpoint[f"encodec_model.{key}"] = value

    converted_checkpoint["codebook_weights"] = original_state_dict["feature_extractor.codebook_weights"]

    for old_key, value in original_state_dict.items():
        if old_key.startswith("backbone."):
            converted_checkpoint[_remap_key(old_key, BACKBONE_MAPPING)] = value
        elif old_key.startswith("head.out."):
            converted_checkpoint[_remap_key(old_key, HEAD_MAPPING)] = value
        elif old_key == "head.istft.window":
            converted_checkpoint["head.window"] = value

    return converted_checkpoint


def safe_load(path: str) -> dict[str, torch.Tensor]:
    """
    Load only the tensor objects from a checkpoint, skipping any BytesIO
    """
    shard = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in shard.items() if not isinstance(v, io.BytesIO)}


@torch.no_grad()
def convert_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, push_to_hub=None):
    if config_path is not None:
        config = VocosWithEncodecConfig.from_pretrained(config_path)
    else:
        config = VocosWithEncodecConfig()

    with torch.device("meta"):
        model = VocosWithEncodecModel(config)

    original_state_dict = safe_load(checkpoint_path)
    new_state_dict = convert_old_keys_to_new_keys(original_state_dict, model_name="encodec_24khz")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False, assign=True)  # , strict=False)

    if len(unexpected_keys) != 0:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    if len(missing_keys) != 0:
        raise ValueError(f"missing keys found: {missing_keys}")

    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(push_to_hub)
        print(f"Pushed model to {push_to_hub}")


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
