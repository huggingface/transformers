#!/usr/bin/env python
# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Convert the matcher weights from an official LoMa checkpoint.

This script converts the LoMa matcher into a Transformers model configured with a native SuperPoint detector. The
reference checkpoints contain DaD and DeDoDe weights as well; those weights are intentionally not converted because
those models are not part of the initial Transformers integration.

Example:

    python src/transformers/models/loma/convert_loma_to_hf.py \
        --checkpoint_path /path/to/loma_B.pt \
        --variant loma_b \
        --output_dir /tmp/loma-b
"""

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

import torch

from transformers import LoMaConfig, LoMaForKeypointMatching


VARIANT_CONFIGS = {
    "loma_b": {"descriptor_dim": 256, "num_attention_heads": 4},
    "loma_l": {"descriptor_dim": 512, "num_attention_heads": 8},
    "loma_g": {"descriptor_dim": 1024, "num_attention_heads": 16},
}


def _rename_matcher_key(key: str, num_hidden_layers: int) -> str | None:
    if key == "posenc.Wr.weight":
        return "positional_encoder.projector.weight"
    if key.startswith("input_proj."):
        return key.replace("input_proj", "input_projection", 1)

    transformer_match = re.fullmatch(r"transformers\.(\d+)\.(self_attn|cross_attn)\.(.+)", key)
    if transformer_match is not None:
        layer_index, attention_type, suffix = transformer_match.groups()
        if attention_type == "self_attn":
            replacements = {
                "Wqkv": "self_attention.qkv",
                "out_proj": "self_attention.output",
                "ffn.0": "self_attention.mlp.layers.0",
                "ffn.1": "self_attention.mlp.layers.1",
                "ffn.3": "self_attention.mlp.layers.3",
            }
        else:
            replacements = {
                "to_qk": "cross_attention.query_key",
                "to_v": "cross_attention.value",
                "to_out": "cross_attention.output",
                "ffn.0": "cross_attention.mlp.layers.0",
                "ffn.1": "cross_attention.mlp.layers.1",
                "ffn.3": "cross_attention.mlp.layers.3",
            }
        for source_prefix, destination_prefix in replacements.items():
            if suffix.startswith(source_prefix + "."):
                return f"transformer_layers.{layer_index}.{suffix.replace(source_prefix, destination_prefix, 1)}"
        return None

    assignment_match = re.fullmatch(r"log_assignment\.(\d+)\.(final_proj|matchability)\.(weight|bias)", key)
    if assignment_match is not None:
        layer_index, source_name, parameter_name = assignment_match.groups()
        if int(layer_index) == num_hidden_layers - 1:
            destination_name = "final_projection" if source_name == "final_proj" else source_name
            return f"match_assignment.{destination_name}.{parameter_name}"
    return None


def convert_matcher_state_dict(
    reference_state_dict: Mapping[str, torch.Tensor], num_hidden_layers: int
) -> dict[str, torch.Tensor]:
    """Extract and rename the matcher tensors from a LoMa reference state dictionary."""
    converted_state_dict = {}
    for source_key, tensor in reference_state_dict.items():
        destination_key = _rename_matcher_key(source_key, num_hidden_layers)
        if destination_key is not None:
            converted_state_dict[destination_key] = tensor
    return converted_state_dict


def convert_checkpoint(checkpoint_path: str | Path, variant: str, output_dir: str | Path) -> None:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unsupported variant: {variant}. Choose one of {sorted(VARIANT_CONFIGS)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a state dictionary checkpoint")

    config = LoMaConfig(**VARIANT_CONFIGS[variant])
    model = LoMaForKeypointMatching(config)
    matcher_state_dict = convert_matcher_state_dict(checkpoint, config.num_hidden_layers)
    missing_keys, unexpected_keys = model.load_state_dict(matcher_state_dict, strict=False)
    matcher_prefixes = ("input_projection", "positional_encoder", "transformer_layers", "match_assignment")
    missing_matcher_keys = [key for key in missing_keys if key.startswith(matcher_prefixes)]
    if missing_matcher_keys or unexpected_keys:
        raise ValueError(
            f"Matcher conversion failed. Missing keys: {missing_matcher_keys}. Unexpected keys: {unexpected_keys}."
        )

    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the official LoMa checkpoint.")
    parser.add_argument("--variant", choices=VARIANT_CONFIGS, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_path, args.variant, args.output_dir)
