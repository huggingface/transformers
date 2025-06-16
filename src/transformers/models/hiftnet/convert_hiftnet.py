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
import os
import re

from safetensors.torch import load_file

from transformers import (
    HiFTNetConfig,
    HiFTNetModel,
)
from transformers.utils.hub import cached_file


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^source_downs\.(\d+)\.(weight|bias)$":                        r"layers.\1.noise_conv.\2",
    r"^ups\.(\d+)\.(.+)$":                                          r"layers.\1.up.\2",

    r"^source_resblocks\.(\d+)\.convs([12])\.(\d+)":                lambda match: f"layers.{match.group(1)}.noise_res.layers.{match.group(3)}.conv{match.group(2)}",
    r"^source_resblocks\.(\d+)\.activations([12]).([012])\.alpha$": lambda match: f"layers.{match.group(1)}.noise_res.layers.{match.group(3)}.alpha{match.group(2)}",

    r"resblocks\.(\d+)\.convs([12])\.(\d+)":                        lambda match: f"layers.{int(match.group(1)) // 3}.resblocks.{int(match.group(1)) % 3}.layers.{match.group(3)}.conv{match.group(2)}",
    r"resblocks\.(\d+)\.activations([12]).([012])\.alpha$":         lambda match: f"layers.{int(match.group(1)) // 3}.resblocks.{int(match.group(1)) % 3}.layers.{match.group(3)}.alpha{match.group(2)}",
}
# fmt: on

STATE_DICT_PREFIX = "mel2wav"
TO_SKIP_PREFIX = "f0_predictor"


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def write_model(
    input_path_or_repo,
    model_name,
    output_dir,
    safe_serialization=True,
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    # Load the configuration for HiFTNet
    config = HiFTNetConfig()

    # Fetch the model path from the cache
    model_path = cached_file(input_path_or_repo, model_name)
    print(f"Fetching all parameters from the checkpoint at {model_path}...")

    # Load the model parameters from the file
    loaded = load_file(model_path, device="cpu")

    # -----------------------
    # Convert parameter names
    # -----------------------

    print("Filtering state dict to keep only HiFTNet weights and converting parameter names...")
    state_dict = {
        convert_key(k[len(STATE_DICT_PREFIX) + 1 :], ORIGINAL_TO_CONVERTED_KEY_MAPPING): v
        for k, v in loaded.items()
        if k.startswith(STATE_DICT_PREFIX) and not k[len(STATE_DICT_PREFIX) + 1 :].startswith(TO_SKIP_PREFIX)
    }

    # -------------------------
    # Load the weights and save
    # -------------------------

    # Initialize the HiFTNet model with the configuration
    model = HiFTNetModel(config)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully.")

    # Save the model to the specified output directory
    print("Saving the model...")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    print(f"Model saved at {output_dir}!")

    # Safety check: reload the converted model
    print("Reloading the model to check if it's saved correctly.")
    HiFTNetModel.from_pretrained(output_dir)
    print("Model reloaded successfully.")


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro weights to HuggingFace format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing Kokoro weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model in input_path_or_repo",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Path to vocab file",
    )
    parser.add_argument(
        "--voice_presets_path_or_repo",
        type=str,
        required=True,
        help="Path or repo that will be the base diretory of paths in voice_to_path_path",
    )
    parser.add_argument(
        "--voice_to_path_path",
        type=str,
        required=True,
        help="Path to voice to path mapping is the voice_presets_path_or_repo directory.",
    )
    args = parser.parse_args()

    write_model(
        args.input_path_or_repo,
        args.model_name,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    # main()
    write_model(
        "ResembleAI/chatterbox",
        "s3gen.safetensors",
        output_dir="./HiFTNet-HF",
        safe_serialization=True,
    )
