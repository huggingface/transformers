# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import gc
import os
import re
from typing import Dict, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AIMv2Config, AIMv2Model, AutoProcessor


NEW_MODEL_KEY_MAPPING = {
    # Embeddings
    r"preprocessor.patchifier.proj": r"embeddings.patch_embed",
    r"preprocessor.pos_embed": r"embeddings.position_embeddings.weight",
    r"preprocessor.patchifier.norm.weight": r"embeddings.rms_norm.weight",
    # Encoder Layers
    r"trunk.blocks.(\d+).attn.qkv": r"encoder.layers.\1.attention.qkv",
    r"trunk.blocks.(\d+).attn.proj": r"encoder.layers.\1.attention.proj_out",
    r"trunk.blocks.(\d+).mlp.fc1": r"encoder.layers.\1.ffn.fc1",
    r"trunk.blocks.(\d+).mlp.fc2": r"encoder.layers.\1.ffn.fc2",
    r"trunk.blocks.(\d+).mlp.fc3": r"encoder.layers.\1.ffn.fc3",
    # Normalization Layers
    r"trunk.blocks.(\d+).norm_1": r"encoder.layers.\1.rms_norm1",
    r"trunk.blocks.(\d+).norm_2": r"encoder.layers.\1.rms_norm2",
    # Final Norm
    r"trunk.post_trunk_norm": r"rms_norm",
}


def load_original_state_dict(model_id: str, revision: Optional[str] = None) -> Dict[str, torch.Tensor]:
    # Download only the model.safetensors file
    directory_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["model.safetensors"],
    )

    original_state_dict = {}
    safetensor_path = f"{directory_path}/model.safetensors"

    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """Converts state dict keys from the old format to the new format."""

    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in NEW_MODEL_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_qkv_tensor(key, tensor):
    """Splits a qkv tensor into separate q, k, v tensors and updates the key accordingly."""

    new_keys = ["q_proj", "k_proj", "v_proj"]
    split_size = tensor.shape[0] // 3
    split_tensors = torch.split(tensor, split_size, dim=0)

    return {key.replace("qkv", new_key): split_tensors[i] for i, new_key in enumerate(new_keys)}


def write_model(
    hf_repo_id: str,
    output_dir: str,
    safe_serialization: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # create config
    config = AIMv2Config.from_pretrained(hf_repo_id)

    # Load original model state dict
    original_state_dict = load_original_state_dict("apple/aimv2-large-patch14-224")

    print("Converting model...")
    state_dict = {}
    result = convert_old_keys_to_new_keys(original_state_dict)
    all_keys = list(original_state_dict.keys())

    for key in all_keys:
        value = original_state_dict[key]
        new_key = result.pop(key)

        if "qkv" in new_key:
            qkv_state_dict = split_qkv_tensor(new_key, value)
            state_dict.update(qkv_state_dict)
        else:
            state_dict[new_key] = value

    state_dict["embeddings.position_embeddings.weight"] = state_dict["embeddings.position_embeddings.weight"].squeeze(
        0
    )

    print("Loading the checkpoint in a DepthPro model.")
    model = AIMv2Model(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")

    print("Saving the model.")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = AIMv2Model.from_pretrained(output_dir, device_map="auto")
    print("Model reloaded successfully.")
    return model


def write_image_processor(hf_repo_id: str, output_dir: str):
    image_processor = AutoProcessor.from_pretrained(hf_repo_id, use_fast=True)
    image_processor.save_pretrained(output_dir)
    return image_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default="apple/aimv2-large-patch14-224",
        help="Location of official weights from apple on HF",
    )
    parser.add_argument(
        "--output_dir",
        default="aimv2_model",
        help="Location to write the converted model and processor",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--push_to_hub",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to push the converted model to the huggingface hub.",
    )
    parser.add_argument(
        "--hub_repo_id",
        default=None,
        help="Huggingface hub repo to write the converted model and processor",
    )
    args = parser.parse_args()

    model = write_model(
        hf_repo_id=args.hf_repo_id,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    image_processor = write_image_processor(
        hf_repo_id=args.hf_repo_id,
        output_dir=args.output_dir,
    )

    if args.push_to_hub:
        print("Pushing to hub...")
        model.push_to_hub(args.hub_repo_id)
        image_processor.push_to_hub(args.hub_repo_id)


if __name__ == "__main__":
    main()

# python src/transformers/models/aimv2/convert_aimv2_original_pytorch_to_hf.py.py --hf_repo_id apple/aimv2-large-patch14-224 --output_dir tmp/aimv2 --safe_serialization
