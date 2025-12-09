# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import json
import os
import re
from typing import Optional

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

from transformers import EomtConfig, EomtForUniversalSegmentation, EomtImageProcessorFast


# fmt: off
MAPPINGS = {
    # Embeddings
    r"network.encoder.backbone.cls_token"                : r"embeddings.cls_token",
    r"network.encoder.backbone.reg_token"                : r"embeddings.register_tokens",
    r"network.encoder.backbone.pos_embed"                : r"embeddings.position_embeddings.weight",
    r"network.encoder.backbone.patch_embed.proj"         : r"embeddings.patch_embeddings.projection",

    # Encoder Block
    r"network.encoder.backbone.blocks.(\d+).norm1"       : r"layers.\1.norm1",
    r"network.encoder.backbone.blocks.(\d+).attn.proj"   : r"layers.\1.attention.out_proj",
    r"network.encoder.backbone.blocks.(\d+).ls1.gamma"   : r"layers.\1.layer_scale1.lambda1",
    r"network.encoder.backbone.blocks.(\d+).norm2"       : r"layers.\1.norm2",
    r"network.encoder.backbone.blocks.(\d+).ls2.gamma"   : r"layers.\1.layer_scale2.lambda1",
    r"network.encoder.backbone.blocks.(\d+).attn"        : r"layers.\1.attention",

    # Others
    r"network.q.weight"                                  : r"query.weight",
    r"network.class_head"                                : r"class_predictor",
    r"network.upscale.(\d+).conv1"                       : r"upscale_block.block.\1.conv1",
    r"network.upscale.(\d+).conv2"                       : r"upscale_block.block.\1.conv2",
    r"network.upscale.(\d+).norm"                        : r"upscale_block.block.\1.layernorm2d",
    r"network.mask_head.0"                               : r"mask_head.fc1",
    r"network.mask_head.2"                               : r"mask_head.fc2",
    r"network.mask_head.4"                               : r"mask_head.fc3",
    r"network.encoder.backbone.norm"                     : r"layernorm",
    r"network.attn_mask_probs"                           : r"attn_mask_probs",
}
# fmt: on

# Mappings for MLP layers, depending on the type of MLP used in ckpts.
MLP_MAPPINGS = {
    "swiglu_ffn": {
        r"network.encoder.backbone.blocks.(\d+).mlp.fc1": r"layers.\1.mlp.weights_in",
        r"network.encoder.backbone.blocks.(\d+).mlp.fc2": r"layers.\1.mlp.weights_out",
    },
    "vanilla_mlp": {
        r"network.encoder.backbone.blocks.(\d+).mlp": r"layers.\1.mlp",
    },
}


def convert_old_keys_to_new_keys(state_dict):
    keys_as_text = "\n".join(state_dict.keys())
    new_keys_as_text = keys_as_text
    for old, repl in MAPPINGS.items():
        if repl is None:
            new_keys_as_text = re.sub(old, "", new_keys_as_text)
        else:
            new_keys_as_text = re.sub(old, repl, new_keys_as_text)
    output_dict = dict(zip(keys_as_text.split("\n"), new_keys_as_text.split("\n")))
    return output_dict


def split_qkv_tensor(key, tensor):
    """Splits a qkv tensor into separate q, k, v tensors and updates the key accordingly."""

    new_keys = ["q_proj", "k_proj", "v_proj"]
    split_size = tensor.shape[0] // 3
    split_tensors = torch.split(tensor, split_size, dim=0)

    return {key.replace("qkv", new_key): split_tensors[i] for i, new_key in enumerate(new_keys)}


def convert_state_dict_to_hf(state_dict):
    """Convert state dict keys to HF format."""
    conversion_dict = convert_old_keys_to_new_keys(state_dict)
    converted_state_dict = {}

    for old_key, new_key in conversion_dict.items():
        if new_key:
            if "qkv" in new_key:  # Detect merged attention keys and split them.
                qkv_split_dict = split_qkv_tensor(new_key, state_dict[old_key])
                converted_state_dict.update(qkv_split_dict)
            else:
                converted_state_dict[new_key] = state_dict[old_key]

    for i in [
        "network.encoder.pixel_mean",
        "network.encoder.pixel_std",
    ]:
        converted_state_dict.pop(i)

    # Embeddings will not have initial dimension
    pos_embed_key = "embeddings.position_embeddings.weight"
    converted_state_dict[pos_embed_key] = converted_state_dict[pos_embed_key].squeeze(0)

    return converted_state_dict


def ensure_model_downloaded(
    repo_id: Optional[str] = None, revision: Optional[str] = None, local_dir: Optional[str] = None
) -> str:
    """
    Ensures model files are downloaded locally, downloads them if not.
    Returns path to local files.

    Args:
        repo_id: The Hugging Face model repo ID (required if local_dir not provided)
        revision: Optional git revision to use
        local_dir: Optional local directory path where model files should be stored/found
    """
    if local_dir is not None:
        if os.path.exists(local_dir):
            print(f"Using provided local directory: {local_dir}")
        else:
            # Create the local directory if it doesn't exist
            os.makedirs(local_dir, exist_ok=True)
            print(f"Created local directory: {local_dir}")

    if repo_id is None:
        raise ValueError("Either repo_id or local_dir must be provided")

    print(f"Ensuring {repo_id} (revision: {revision or 'latest'}) is downloaded...")

    try:
        # First try to find files locally
        download_dir = snapshot_download(repo_id, revision=revision, local_files_only=True, local_dir=local_dir)
        print(f"Found model files locally at {download_dir}")
        return download_dir
    except Exception:
        # If files not found locally, download them
        print(f"Downloading model files for {repo_id}...")
        download_dir = snapshot_download(repo_id, revision=revision, local_files_only=False, local_dir=local_dir)
        print(f"Downloaded model files to {download_dir}")
        return download_dir


def load_model_state_dict(input_path: str) -> dict:
    """
    Load model state dict, handling both single and sharded files.
    """
    index_path = os.path.join(input_path, "pytorch_model.bin.index.json")
    single_file_path = os.path.join(input_path, "pytorch_model.bin")

    # Check if we have a sharded model
    if os.path.exists(index_path):
        print("Loading sharded model...")
        state_dict = {}
        with open(index_path, "r") as f:
            index = json.load(f)

        # Get unique shard files and load each one only once
        unique_shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in unique_shard_files:
            print(f"Loading shard {shard_file}...")
            shard_path = os.path.join(input_path, shard_file)
            shard_dict = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_dict)

        return state_dict

    # Single file model
    elif os.path.exists(single_file_path):
        print("Loading single file model...")
        return torch.load(single_file_path, map_location="cpu")

    else:
        raise ValueError(f"No model files found in {input_path}")


def convert_model(
    repo_id=None,
    local_dir=None,
    output_dir=None,
    output_hub_path=None,
    revision=None,
):
    """Convert and save the model weights, processor, and configuration."""
    if output_dir is None and output_hub_path is None:
        raise ValueError("At least one of output_dir or output_hub_path must be specified")

    if repo_id is None and local_dir is None:
        raise ValueError("Either repo_id or local_dir must be specified")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/verified output directory: {output_dir}")

    torch.set_default_dtype(torch.float16)

    # Download or locate model files
    input_path = ensure_model_downloaded(repo_id=repo_id, revision=revision, local_dir=local_dir)

    with open(os.path.join(input_path, "config.json"), "r") as f:
        config_data = json.load(f)
    # Pop off unwanted keys
    _ = config_data.pop("backbone", None)

    config = EomtConfig(
        **{
            **config_data,
            "layerscale_value": 1e-5,
        }
    )

    if "semantic" in repo_id.split("_"):
        size = {"shortest_edge": config.image_size, "longest_edge": None}
        do_split_image = True
        do_pad = False
    else:
        size = {"shortest_edge": config.image_size, "longest_edge": config.image_size}
        do_split_image = False
        do_pad = True

    if "giant" in repo_id.split("_"):
        config.use_swiglu_ffn = True
        config.hidden_size = 1536
        config.num_hidden_layers = 40
        config.num_attention_heads = 24
        # Update MAPPINGS for ckpts depending on the MLP type
        MAPPINGS.update(MLP_MAPPINGS["swiglu_ffn"])
    else:
        MAPPINGS.update(MLP_MAPPINGS["vanilla_mlp"])

    processor = EomtImageProcessorFast(size=size, do_split_image=do_split_image, do_pad=do_pad)

    # Save the config and processor
    if output_dir:
        config.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    if output_hub_path:
        config.push_to_hub(output_hub_path)
        processor.push_to_hub(output_hub_path)

    # Initialize model with empty weights
    print("Creating empty model...")
    with init_empty_weights():
        model = EomtForUniversalSegmentation(config)

    # Load and convert state dict
    print("Loading state dict...")
    state_dict = load_model_state_dict(input_path)
    state_dict = convert_state_dict_to_hf(state_dict)

    # Load converted state dict
    print("Loading converted weights into model...")
    model.load_state_dict(state_dict, strict=True, assign=True)

    # Save the model
    if output_dir:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
    if output_hub_path:
        print(f"Pushing model to hub at {output_hub_path}...")
        model.push_to_hub(output_hub_path)

    del state_dict, model
    gc.collect()

    # Validate the saved model if saved locally
    if output_dir:
        print("Reloading the local model to check if it's saved correctly...")
        EomtForUniversalSegmentation.from_pretrained(output_dir, device_map="auto")
        print("Local model reloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        help="HuggingFace Hub repo ID for the model",
        default=None,
    )
    parser.add_argument(
        "--local_dir",
        help="Local directory containing the model files",
        default=None,
    )
    parser.add_argument(
        "--revision",
        help="Specific revision to download from the Hub",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model locally",
        default=None,
    )
    parser.add_argument(
        "--output_hub_path",
        help="Repository ID to push model to hub (e.g. 'username/model-name')",
        default=None,
    )
    args = parser.parse_args()

    if args.output_dir is None and args.output_hub_path is None:
        raise ValueError("At least one of --output_dir or --output_hub_path must be specified")

    if args.hf_repo_id is None and args.local_dir is None:
        raise ValueError("Either --hf_repo_id or --local_dir must be specified")

    convert_model(
        repo_id=args.hf_repo_id,
        local_dir=args.local_dir,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
