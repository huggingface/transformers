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

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import (
    Aimv2Config,
    Aimv2Model,
    Aimv2VisionConfig,
    Aimv2VisionModel,
    AutoImageProcessor,
    AutoProcessor,
)


ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION_MODEL = {
    # Embeddings
    r"preprocessor.patchifier.proj": r"embeddings.patch_embed",
    r"preprocessor.pos_embed": r"embeddings.position_embedding.weight",
    r"preprocessor.patchifier.norm.weight": r"embeddings.rms_norm.weight",
    # Encoder Layers
    r"trunk.blocks.(\d+).attn.qkv": r"encoder.layers.\1.attention.qkv",
    r"trunk.blocks.(\d+).attn.proj": r"encoder.layers.\1.attention.out_proj",
    r"trunk.blocks.(\d+).mlp.fc1": r"encoder.layers.\1.ffn.gate_proj",
    r"trunk.blocks.(\d+).mlp.fc2": r"encoder.layers.\1.ffn.down_proj",
    r"trunk.blocks.(\d+).mlp.fc3": r"encoder.layers.\1.ffn.up_proj",
    # Normalization Layers
    r"trunk.blocks.(\d+).norm_1": r"encoder.layers.\1.rms_norm1",
    r"trunk.blocks.(\d+).norm_2": r"encoder.layers.\1.rms_norm2",
    # Final Norm
    r"trunk.post_trunk_norm": r"rms_norm",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision Embeddings
    r"image_encoder.preprocessor.patchifier.proj": r"vision_model.embeddings.patch_embed",
    r"image_encoder.preprocessor.pos_embed": r"vision_model.embeddings.position_embedding.weight",
    r"image_encoder.preprocessor.patchifier.norm.weight": r"vision_model.embeddings.rms_norm.weight",
    # Vision Encoder Layers
    r"image_encoder.trunk.blocks.(\d+).attn.qkv": r"vision_model.encoder.layers.\1.attention.qkv",
    r"image_encoder.trunk.blocks.(\d+).attn.proj": r"vision_model.encoder.layers.\1.attention.out_proj",
    r"image_encoder.trunk.blocks.(\d+).mlp.fc1": r"vision_model.encoder.layers.\1.ffn.gate_proj",
    r"image_encoder.trunk.blocks.(\d+).mlp.fc2": r"vision_model.encoder.layers.\1.ffn.down_proj",
    r"image_encoder.trunk.blocks.(\d+).mlp.fc3": r"vision_model.encoder.layers.\1.ffn.up_proj",
    # Normalization Layers
    r"image_encoder.trunk.blocks.(\d+).norm_1": r"vision_model.encoder.layers.\1.rms_norm1",
    r"image_encoder.trunk.blocks.(\d+).norm_2": r"vision_model.encoder.layers.\1.rms_norm2",
    r"image_encoder.trunk.post_trunk_norm": r"vision_model.rms_norm",
    r"image_projector": r"visual_projection",
    # Vision Head
    r"image_encoder.head.cls_token": r"vision_model.head.cls_token",
    r"image_encoder.head.k": r"vision_model.head.k_proj",
    r"image_encoder.head.v": r"vision_model.head.v_proj",
    r"image_encoder.head.linear": r"vision_model.head.output_proj",
    # Text Embeddings
    r"text_encoder.preprocessor.text_embedding.weight": r"text_model.embeddings.token_embedding.weight",
    r"text_encoder.preprocessor.positional_embedding": r"text_model.embeddings.position_embedding.weight",
    # Text Encoder Layers
    r"text_encoder.trunk.blocks.(\d+).attn.qkv": r"text_model.encoder.layers.\1.attention.qkv",
    r"text_encoder.trunk.blocks.(\d+).attn.proj": r"text_model.encoder.layers.\1.attention.out_proj",
    r"text_encoder.trunk.blocks.(\d+).mlp.fc1": r"text_model.encoder.layers.\1.ffn.gate_proj",
    r"text_encoder.trunk.blocks.(\d+).mlp.fc2": r"text_model.encoder.layers.\1.ffn.down_proj",
    r"text_encoder.trunk.blocks.(\d+).mlp.fc3": r"text_model.encoder.layers.\1.ffn.up_proj",
    # Text Normalization Layers
    r"text_encoder.trunk.blocks.(\d+).norm_1": r"text_model.encoder.layers.\1.rms_norm1",
    r"text_encoder.trunk.blocks.(\d+).norm_2": r"text_model.encoder.layers.\1.rms_norm2",
    r"text_encoder.trunk.post_trunk_norm": r"text_model.rms_norm",
    r"text_projector": r"text_projection",
    r"log_logit_scale": r"logit_scale",
}


def load_original_state_dict(model_id: str, revision: str | None = None) -> dict[str, torch.Tensor]:
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


def convert_old_keys_to_new_keys(state_dict_keys: dict, ORIGINAL_TO_CONVERTED_KEY_MAPPING: dict):
    """Converts state dict keys from the old format to the new format."""

    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
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


def get_model_config_mapping(model_id: str):
    """Determines the correct model, config, and key mappings based on the checkpoint name."""

    if model_id == "apple/aimv2-large-patch14-224-lit":
        return Aimv2Model, Aimv2Config, ORIGINAL_TO_CONVERTED_KEY_MAPPING
    else:
        return Aimv2VisionModel, Aimv2VisionConfig, ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION_MODEL


def write_model(
    hf_repo_id: str,
    output_dir: str,
):
    """
    Converts a model checkpoint to Hugging Face format and saves it.

    Args:
        hf_repo_id (str): The Hugging Face repo ID to load from.
        output_dir (str): The directory to save the converted model.

    Returns:
        model: The reloaded Hugging Face model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the appropriate model, config, and key mapping
    model_class, config_class, key_mapping = get_model_config_mapping(hf_repo_id)

    # Load config and original state dict
    config = config_class.from_pretrained(hf_repo_id)

    # Checkpoint `apple/aimv2-large-patch14-224-lit` uses AttentionPoolingHead hence set the required attr in config.
    if hf_repo_id != "apple/aimv2-large-patch14-224-lit":
        config.use_head = False

    if hf_repo_id == "apple/aimv2-large-patch14-native":
        config.is_native = True

    original_state_dict = load_original_state_dict(hf_repo_id)

    print("Converting model...")

    state_dict = {}
    result = convert_old_keys_to_new_keys(original_state_dict, key_mapping)
    all_keys = list(original_state_dict.keys())

    for key in all_keys:
        value = original_state_dict[key]
        new_key = result.pop(key)

        if "qkv" in new_key:
            qkv_state_dict = split_qkv_tensor(new_key, value)
            state_dict.update(qkv_state_dict)
        else:
            state_dict[new_key] = value

        # Check if position embeddings exist before squeezing
        if new_key.endswith("position_embedding.weight"):
            state_dict[new_key] = value.squeeze(0)

    print(f"Loading the checkpoint in a {model_class.__name__}.")
    model = model_class(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")

    print("Saving the model.")
    model.save_pretrained(output_dir)
    del state_dict, model
    gc.collect()

    print("Reloading the model to check if it's saved correctly.")
    model = model_class.from_pretrained(output_dir, device_map="auto")
    print("Model reloaded successfully.")
    return model


def write_image_processor(hf_repo_id: str, output_dir: str):
    if hf_repo_id == "apple/aimv2-large-patch14-224-lit":
        image_processor = AutoProcessor.from_pretrained(hf_repo_id, use_fast=True)
    else:
        image_processor = AutoImageProcessor.from_pretrained(hf_repo_id, use_fast=True)
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
