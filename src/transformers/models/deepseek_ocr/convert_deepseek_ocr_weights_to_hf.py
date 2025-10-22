# Copyright 2025 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
import json
from pathlib import Path

import torch

from transformers import DeepSeekOCRConfig, DeepSeekOCRForCausalLM


def convert_state_dict(original_state_dict):
    """
    Convert original state dict to HuggingFace format.

    This function performs minimal key renaming to match the new modular structure.
    """
    new_state_dict = {}

    for key, value in original_state_dict.items():
        new_key = key

        if key.startswith("model.sam_model."):
            new_key = key.replace("model.sam_model.", "model.sam_model.encoder.")

        elif key.startswith("model.vision_model."):
            new_key = key.replace("model.vision_model.", "model.clip_model.model.vision_model.")

        elif key.startswith("model.projector.layers."):
            new_key = key

        elif key == "model.image_newline":
            new_key = key

        elif key == "model.view_seperator":
            new_key = "model.view_separator"

        elif key.startswith("model.") and not any(
            prefix in key for prefix in ["sam_model", "vision_model", "projector", "image_newline", "view_separator"]
        ):
            new_key = key.replace("model.", "model.language_model.")

        new_state_dict[new_key] = value

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSeek OCR weights to HuggingFace format")
    parser.add_argument(
        "--original_checkpoint_path",
        type=str,
        required=True,
        help="Path to the original checkpoint file (PyTorch .pt or .pth file)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the converted model",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config.json file. If not provided, will use default config.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Repository ID for pushing to the Hub (required if --push_to_hub is set)",
    )

    args = parser.parse_args()

    print(f"Loading original checkpoint from {args.original_checkpoint_path}")
    original_state_dict = torch.load(args.original_checkpoint_path, map_location="cpu")

    if "model" in original_state_dict:
        original_state_dict = original_state_dict["model"]

    print("Converting state dict...")
    converted_state_dict = convert_state_dict(original_state_dict)

    if args.config_path:
        print(f"Loading config from {args.config_path}")
        with open(args.config_path, "r") as f:
            config_dict = json.load(f)
        config = DeepSeekOCRConfig(**config_dict)
    else:
        print("Using default config")
        config = DeepSeekOCRConfig()

    print("Creating model...")
    model = DeepSeekOCRForCausalLM(config)

    print("Loading converted state dict into model...")
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print(f"Saving converted model to {args.output_path}")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(args.output_path)
    config.save_pretrained(args.output_path)

    print("Conversion complete!")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo_id must be provided when --push_to_hub is set")

        print(f"Pushing model to Hub: {args.repo_id}")
        model.push_to_hub(args.repo_id)
        print("Model pushed successfully!")


if __name__ == "__main__":
    main()
