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
import re

import torch

from transformers import (
    DacConfig,
    DacFeatureExtractor,
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
)
from transformers.utils.hub import cached_file


INNER_LAYER_NAMES = ["snake1", "conv1", "snake2", "conv2"]

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Encoder: initial conv, final snake + conv
    r"^encoder\.block\.0":                                                           "acoustic_encoder.conv1",
    r"^encoder\.block\.6":                                                          "acoustic_encoder.snake1",
    r"^encoder\.block\.7":                                                           "acoustic_encoder.conv2",

    # Encoder: res_unit inner layers (block.M res_unit index 0-2 → 1-3, block.K layer index 0-3 → snake1/conv1/snake2/conv2)
    r"^encoder\.block\.(\d+)\.block\.([012])\.block\.([0123])": lambda m: f"acoustic_encoder.block.{int(m[1])-1}.res_unit{int(m[2])+1}.{INNER_LAYER_NAMES[int(m[3])]}",
    # Encoder: block-level snake + downsampling conv
    r"^encoder\.block\.(\d+)\.block\.3":                        lambda m: f"acoustic_encoder.block.{int(m[1])-1}.snake1",
    r"^encoder\.block\.(\d+)\.block\.4":                        lambda m: f"acoustic_encoder.block.{int(m[1])-1}.conv1",

    # Decoder: initial conv, final snake + conv
    r"^decoder_2\.model\.0":                                                         "acoustic_decoder.conv1",
    r"^decoder_2\.model\.6":                                                        "acoustic_decoder.snake1",
    r"^decoder_2\.model\.7":                                                         "acoustic_decoder.conv2",

    # Decoder: block-level snake + upsample conv_t
    r"^decoder_2\.model\.(\d+)\.block\.0":                      lambda m: f"acoustic_decoder.block.{int(m[1])-1}.snake1",
    r"^decoder_2\.model\.(\d+)\.block\.1":                      lambda m: f"acoustic_decoder.block.{int(m[1])-1}.conv_t1",
    # Decoder: res_unit inner layers (block.M res_unit index 2-4 → 1-3, block.K layer index 0-3 → snake1/conv1/snake2/conv2)
    r"^decoder_2\.model\.(\d+)\.block\.([234])\.block\.([0123])": lambda m: f"acoustic_decoder.block.{int(m[1])-1}.res_unit{int(m[2])-1}.{INNER_LAYER_NAMES[int(m[3])]}",

    # Quantizer
    r"^quantizer\.vq\.layers":                                                "quantizer.quantizers",
    r"\._codebook\.":                                                             ".codebook.",

    # FC layers
    r"^fc_prior\.":                                                                         "fc.",
    r"^fc_post1\.":                                                                        "fc1.",
    r"^fc_post2\.":                                                                        "fc2.",

    # Semantic encoder/decoder: unwrap nested conv modules
    r"\.conv\.conv\.":                                                                ".conv.",
    r"\.conv1\.conv\.":                                                              ".conv1.",
    r"\.conv2\.conv\.":                                                              ".conv2.",
}
# fmt: on


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def compute_weight_from_weight_norm(weight_v, weight_g):
    """Combine weight_v and weight_g from weight normalization into a plain weight."""
    dims = list(range(1, weight_v.dim()))
    norm = weight_v.norm(dim=dims, keepdim=True)
    return weight_g * weight_v / norm


def convert_model(input_path_or_repo, revision=None):
    print("Converting the model.")

    config = HiggsAudioV2TokenizerConfig(
        acoustic_model_config=DacConfig(
            encoder_hidden_size=64,
            downsampling_ratios=[8, 5, 4, 2, 3],
            decoder_hidden_size=1024,
            upsampling_ratios=[8, 5, 4, 2, 3],
            hidden_size=256,
        ),
    )

    model_path = cached_file(input_path_or_repo, "model.pth", revision=revision)
    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)

    print("Converting model...")

    # -----------------------------------------
    # Preprocess: merge weight_norm into weight
    # -----------------------------------------

    preprocessed = {}
    for key, value in loaded.items():
        if key.endswith(".weight_g"):
            base = key.removesuffix(".weight_g")
            weight = compute_weight_from_weight_norm(loaded[base + ".weight_v"], value)
            preprocessed[base + ".weight"] = weight
        elif key.endswith(".weight_v"):
            continue  # already handled with weight_g
        else:
            preprocessed[key] = value

    del loaded
    gc.collect()

    # -----------------------
    # Convert parameter names
    # -----------------------

    state_dict = {}
    for key, value in preprocessed.items():
        # fc1 is not used in the forward pass
        if key.startswith("fc1."):
            continue
        # masked_spec_embed is not used in inference
        if key == "semantic_model.masked_spec_embed":
            continue

        new_key = convert_key(key, ORIGINAL_TO_CONVERTED_KEY_MAPPING)
        state_dict[new_key] = value

    del preprocessed
    gc.collect()

    # -------------------------
    # Load the weights
    # -------------------------

    print("Loading the checkpoint in a HiggsAudioV2TokenizerModel.")
    with torch.device("meta"):
        model = HiggsAudioV2TokenizerModel(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Model converted successfully.")
    del model.config._name_or_path

    return model


def create_feature_extractor():
    feature_extractor = DacFeatureExtractor(
        feature_size=1,
        hop_length=960,
        padding_side="right",
        padding_value=0.0,
        return_attention_mask=True,
        sampling_rate=24000,
    )
    return feature_extractor


def main():
    parser = argparse.ArgumentParser(description="Convert HiggsAudioV2Tokenizer weights to HuggingFace format")
    parser.add_argument("--input_path_or_repo", type=str, default="bosonai/higgs-audio-v2-tokenizer")
    parser.add_argument("--input_revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub_path", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None and args.push_to_hub_path is None:
        raise ValueError("Either --output_dir or --push_to_hub_path must be provided.")

    model = convert_model(args.input_path_or_repo, revision=args.input_revision)
    feature_extractor = create_feature_extractor()

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        feature_extractor.save_pretrained(args.output_dir)
        print(f"Model and feature extractor saved to {args.output_dir}")

    if args.push_to_hub_path is not None:
        model.push_to_hub(args.push_to_hub_path)
        feature_extractor.push_to_hub(args.push_to_hub_path)
        print(f"Model and feature extractor pushed to {args.push_to_hub_path}")


if __name__ == "__main__":
    main()
