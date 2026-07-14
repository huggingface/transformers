# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""
Convert an original WavTokenizer checkpoint (https://github.com/jishengpeng/WavTokenizer) into a
transformers WavTokenizerModel.

The reference checkpoint for Apertus 1.5 is the 40 tokens/s, 24 kHz variant:
    novateur/WavTokenizer-large-unify-40token / wavtokenizer_large_unify_600_24k.ckpt

Example:
    python src/transformers/models/wavtokenizer/convert_wavtokenizer_checkpoint.py \
        --checkpoint_path wavtokenizer_large_unify_600_24k.ckpt \
        --output_dir ./wavtokenizer-large-unify-40token-hf
"""

import argparse
import re

import torch

from transformers import WavTokenizerConfig, WavTokenizerFeatureExtractor, WavTokenizerModel, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Training-only / unused branches of the original Lightning checkpoint.
IGNORE_PREFIXES = (
    "multiperioddisc.",
    "multiresddisc.",
    "dac.",
    "dacdiscriminator.",
    "melspec_loss.",
    # The original SEANet *decoder* is bypassed at inference (the Vocos backbone + ISTFT head replace it).
    "feature_extractor.encodec.decoder.",
)

# Buffers that transformers re-creates (non-persistent) instead of loading.
IGNORE_KEYS = ("head.istft.window",)


def remap_key(key: str) -> str | None:
    """Map an original state-dict key to the transformers WavTokenizerModel naming; None to drop."""
    if key in IGNORE_KEYS or any(key.startswith(prefix) for prefix in IGNORE_PREFIXES):
        return None

    # Encoder: feature_extractor.encodec.encoder.model.{i}... -> encoder.layers.{i}...
    key = key.replace("feature_extractor.encodec.encoder.model.", "encoder.layers.")
    # Quantizer: ...quantizer.vq.layers.0._codebook.* -> quantizer.codebook.*
    key = key.replace("feature_extractor.encodec.quantizer.vq.layers.0._codebook.", "quantizer.codebook.")
    # ISTFT head: head.out.* -> head.linear.*
    key = key.replace("head.out.", "head.linear.")

    # Original SConv1d nests the conv as `.conv.conv` with legacy weight-norm keys `weight_g`/`weight_v`;
    # transformers EncodecConv1d-style modules use `.conv` with torch parametrized weight norm.
    key = re.sub(r"\.conv\.conv\.weight_g$", ".conv.parametrizations.weight.original0", key)
    key = re.sub(r"\.conv\.conv\.weight_v$", ".conv.parametrizations.weight.original1", key)
    key = re.sub(r"\.conv\.conv\.bias$", ".conv.bias", key)

    return key


def convert_checkpoint(checkpoint_path: str, output_dir: str, push_to_hub: str | None = None):
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = original.get("state_dict", original)

    converted = {}
    for key, value in state_dict.items():
        new_key = remap_key(key)
        if new_key is not None:
            converted[new_key] = value

    config = WavTokenizerConfig()
    model = WavTokenizerModel(config)
    model.load_state_dict(converted, strict=True)
    model.eval()

    feature_extractor = WavTokenizerFeatureExtractor(sampling_rate=config.sampling_rate, hop_length=config.hop_length)

    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    logger.info(f"Saved converted model + feature extractor to {output_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        feature_extractor.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the original .ckpt file")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push to")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_path, args.output_dir, args.push_to_hub)
