# coding=utf-8
# Copyright 2025 The Resemble AI and HuggingFace Inc. team. All rights reserved.
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
"""Converts a Chatterbox model in Resemble AI format to Hugging Face format."""

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from transformers import ChatterboxConfig, ChatterboxModel


def convert_chatterbox_model_to_hf(checkpoint_path, pytorch_dump_folder_path, verbose=False):
    """
    Converts a Chatterbox model in Resemble AI format to Hugging Face format.
    """
    # Download from HF Hub
    checkpoint_dir = snapshot_download(repo_id=checkpoint_path)
    print(f"Downloaded checkpoint from Hugging Face Hub: {checkpoint_dir}")

    # Load original checkpoints
    print("Loading original checkpoints...")
    s3gen_path = os.path.join(checkpoint_dir, "s3gen.pt")
    t3_path = os.path.join(checkpoint_dir, "t3.pt")
    ve_path = os.path.join(checkpoint_dir, "ve.pt")

    # Check if files exist
    if not os.path.exists(t3_path):
        # Fallback to other possible names
        t3_path = os.path.join(checkpoint_dir, "t3_cfg.pt")
        if not os.path.exists(t3_path):
            # Maybe it is in a different snapshot or just t3_23lang.safetensors
            t3_path = next(Path(checkpoint_dir).glob("t3*.safetensors"), None) or next(
                Path(checkpoint_dir).glob("t3*.pt"), None
            )
            if t3_path:
                t3_path = str(t3_path)

    print(f"Using T3 path: {t3_path}")
    print(f"Using S3Gen path: {s3gen_path}")
    print(f"Using VE path: {ve_path}")

    s3gen_sd = torch.load(s3gen_path, map_location="cpu")
    t3_sd = (
        torch.load(t3_path, map_location="cpu") if t3_path.endswith(".pt") else None
    )  # Will handle safetensors later if needed
    if t3_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        t3_sd = load_file(t3_path)

    ve_sd = torch.load(ve_path, map_location="cpu") if ve_path.endswith(".pt") else None
    if ve_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        ve_sd = load_file(ve_path)

    # Handle 'model' wrapper in T3 if present
    if "model" in t3_sd and len(t3_sd) == 1:
        print("Unwrapping T3 state dict from 'model' key...")
        t3_sd = t3_sd["model"]
        if isinstance(t3_sd, list):
            t3_sd = t3_sd[0]

    # Initialize HF model
    # Detect if multilingual from t3_path name or vocab size if possible
    is_multilingual = "23lang" in t3_path or "multilingual" in t3_path

    config = ChatterboxConfig(is_multilingual=is_multilingual)
    model = ChatterboxModel(config)

    hf_sd = model.state_dict()
    new_sd = {}

    if verbose:
        print(f"HF model has {len(hf_sd)} keys")
        # print("First 20 HF keys:", sorted(list(hf_sd.keys()))[:20])

    print("Mapping T3 weights...")
    # T3 Mapping
    for key, value in t3_sd.items():
        # original T3: tfmr.layers.0... -> HF Chatterbox: t3.layers.0...
        new_key = key
        if new_key.startswith("tfmr."):
            new_key = new_key.replace("tfmr.", "t3.", 1)
        elif not new_key.startswith("t3."):
            new_key = "t3." + new_key

        if new_key in hf_sd:
            new_sd[new_key] = value
        elif new_key.replace("t3.model.", "t3.") in hf_sd:
            new_sd[new_key.replace("t3.model.", "t3.")] = value
        else:
            if verbose:
                print(f"Skipping T3 key: {key} (mapped to {new_key})")

    print("Mapping S3Gen weights...")
    # S3Gen Mapping
    for key, value in s3gen_sd.items():
        # original S3Gen: flow..., mel2wav..., tokenizer... -> HF Chatterbox: s3gen.flow..., s3gen.mel2wav..., s3gen.tokenizer...
        new_key = "s3gen." + key

        # Handle S3Tokenizer mapping difference
        if new_key.startswith("s3gen.tokenizer."):
            new_key = new_key.replace("s3gen.tokenizer.", "s3gen.tokenizer.s3_model.", 1)

        if new_key in hf_sd:
            new_sd[new_key] = value
        else:
            if verbose:
                print(f"Skipping S3Gen key: {key} (mapped to {new_key})")

    print("Mapping Voice Encoder weights...")
    # VE Mapping
    for key, value in ve_sd.items():
        # original VE: lstm..., proj... -> HF Chatterbox: t3.voice_encoder.lstm..., t3.voice_encoder.proj...
        new_key = "t3.voice_encoder." + key
        if new_key in hf_sd:
            new_sd[new_key] = value
        else:
            if verbose:
                print(f"Skipping VE key: {key} (mapped to {new_key})")

    # Check for missing keys
    missing_keys = set(hf_sd.keys()) - set(new_sd.keys())
    # Filter out known computed/buffer keys that might not be in checkpoints
    missing_keys = {
        k
        for k in missing_keys
        if not any(x in k for x in ["inv_freq", "stft_window", "trim_fade", "window", "_mel_filters", "embed_tokens"])
    }

    if missing_keys:
        print(f"Warning: Missing keys in new state dict: {len(missing_keys)}")
        if verbose:
            for k in sorted(missing_keys)[:20]:
                print(f"  Missing: {k}")

    # Load state dict
    model.load_state_dict(new_sd, strict=False)

    # Remove weight norm for saving
    print("Removing weight norm...")
    try:
        if hasattr(model.s3gen, "mel2wav") and hasattr(model.s3gen.mel2wav, "remove_weight_norm"):
            model.s3gen.mel2wav.remove_weight_norm()
    except Exception as e:
        print(f"Warning: Could not remove weight norm from s3gen.mel2wav: {e}")

    # Save model
    print(f"Saving model to {pytorch_dump_folder_path}...")
    model.save_pretrained(pytorch_dump_folder_path)

    # Copy tokenizer.json if it exists
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        import shutil

        shutil.copy(tokenizer_path, os.path.join(pytorch_dump_folder_path, "tokenizer.json"))
        print(f"Copied tokenizer.json to {pytorch_dump_folder_path}")

    print("Conversion completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="ResembleAI/chatterbox",
        help="Path to the downloaded checkpoints",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="chatterbox-hf",
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether or not to log information during conversion.",
    )
    args = parser.parse_args()

    convert_chatterbox_model_to_hf(args.checkpoint_path, args.pytorch_dump_folder_path, args.verbose)
