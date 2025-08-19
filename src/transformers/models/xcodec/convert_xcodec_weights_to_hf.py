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
import re

import torch

from transformers import (
    EncodecFeatureExtractor,
    XcodecConfig,
    XcodecModel,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


torch.serialization.add_safe_globals([io.BytesIO])

MAPPING_ACOUSTIC_ENCODER = {
    r"^block\.0": ["conv1"],
    r"^block\.(\d+)\.block\.(\d+)\.block\.0": ["block", "res_unit", "snake1"],
    r"^block\.(\d+)\.block\.(\d+)\.block\.1": ["block", "res_unit", "conv1"],
    r"^block\.(\d+)\.block\.(\d+)\.block\.2": ["block", "res_unit", "snake2"],
    r"^block\.(\d+)\.block\.(\d+)\.block\.3": ["block", "res_unit", "conv2"],
    r"^block\.(\d+)\.block\.3": ["block", "snake1"],
    r"^block\.(\d+)\.block\.4": ["block", "conv1"],
    r"^block\.5": ["snake1"],
    r"^block\.6": ["conv2"],
}

MAPPING_ACOUSTIC_DECODER = {
    r"^model\.0": ["conv1"],
    r"^model\.(\d+)\.block\.0": ["block", "snake1"],
    r"^model\.(\d+)\.block\.1": ["block", "conv_t1"],
    r"^model\.(\d+)\.block\.(\d+)\.block\.0": ["block", "res_unit", "snake1"],
    r"^model\.(\d+)\.block\.(\d+)\.block\.1": ["block", "res_unit", "conv1"],
    r"^model\.(\d+)\.block\.(\d+)\.block\.2": ["block", "res_unit", "snake2"],
    r"^model\.(\d+)\.block\.(\d+)\.block\.3": ["block", "res_unit", "conv2"],
    r"^model\.5": ["snake1"],
    r"^model\.6": ["conv2"],
}

MAPPING_SEMANTIC_ENCODER = {
    "conv.conv.": "conv.",
    "conv1.conv.": "conv1.",
    "conv2.conv.": "conv2.",
}

MAPPING_SEMANTIC_DECODER = {
    "conv1.conv.": "conv1.",
    "conv2.conv.": "conv2.",
    "conv.conv.": "conv.",
}

MAPPING_QUANTIZER = {
    "quantizer.vq.layers": "quantizer.quantizers",
    "._codebook.": ".codebook.",
}


def safe_load(path: str) -> dict[str, torch.Tensor]:
    """
    Load only the tensor objects from a checkpoint, skipping any BytesIO
    """
    shard = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in shard.items() if not isinstance(v, io.BytesIO)}


def _rewrite_weight_norm(key: str) -> str:
    if key.endswith("weight_g"):
        return key[: -len("weight_g")] + "parametrizations.weight.original0"
    if key.endswith("weight_v"):
        return key[: -len("weight_v")] + "parametrizations.weight.original1"
    return key


def convert_old_keys_to_new_keys(original_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted_checkpoint: dict[str, torch.Tensor] = {}

    for old_key, value in original_state_dict.items():
        if old_key.startswith("encoder."):
            layer_key = old_key[len("encoder.") :]
            for pattern, path_parts in MAPPING_ACOUSTIC_ENCODER.items():
                pattern_match = re.match(pattern, layer_key)
                if pattern_match is None:
                    continue

                digit_strings = [g for g in pattern_match.groups() if g is not None]
                digit_indices = [int(ds) for ds in digit_strings]
                remainder = layer_key[pattern_match.end() :]

                if len(path_parts) == 1:
                    mapped_subkey = f"{path_parts[0]}{remainder}"
                elif len(path_parts) == 2:
                    encoder_layer = digit_indices[0] - 1
                    mapped_subkey = f"{path_parts[0]}.{encoder_layer}.{path_parts[1]}{remainder}"
                else:
                    encoder_layer, unit_idx = digit_indices
                    mapped_subkey = (
                        f"{path_parts[0]}.{encoder_layer - 1}.{path_parts[1]}{unit_idx + 1}.{path_parts[2]}{remainder}"
                    )

                new_key = f"acoustic_encoder.{_rewrite_weight_norm(mapped_subkey)}"
                converted_checkpoint[new_key] = value
                break

        elif old_key.startswith("decoder_2."):
            layer_key = old_key[len("decoder_2.") :]

            for pattern, path_parts in MAPPING_ACOUSTIC_DECODER.items():
                pattern_match = re.match(pattern, layer_key)
                if pattern_match is None:
                    continue
                digit_strings = [g for g in pattern_match.groups() if g is not None]
                digit_indices = [int(ds) for ds in digit_strings]
                remainder = layer_key[pattern_match.end() :]

                if len(path_parts) == 1:
                    mapped_subkey = f"{path_parts[0]}{remainder}"
                elif len(path_parts) == 2:
                    decoder_layer = digit_indices[0] - 1
                    mapped_subkey = f"{path_parts[0]}.{decoder_layer}.{path_parts[1]}{remainder}"
                else:
                    decoder_layer, unit_idx = digit_indices
                    mapped_subkey = (
                        f"{path_parts[0]}.{decoder_layer - 1}.{path_parts[1]}{unit_idx - 1}.{path_parts[2]}{remainder}"
                    )
                new_key = f"acoustic_decoder.{_rewrite_weight_norm(mapped_subkey)}"
                converted_checkpoint[new_key] = value
                break

        elif old_key.startswith("encoder_semantic."):
            semantic_key = old_key[len("encoder_semantic.") :]
            for old, new in MAPPING_SEMANTIC_ENCODER.items():
                semantic_key = semantic_key.replace(old, new)
            converted_checkpoint[f"encoder_semantic.{semantic_key}"] = value

        elif old_key.startswith("decoder_semantic."):
            semantic_key = old_key[len("decoder_semantic.") :]
            for old, new in MAPPING_SEMANTIC_DECODER.items():
                semantic_key = semantic_key.replace(old, new)
            converted_checkpoint[f"decoder_semantic.{semantic_key}"] = value

        elif old_key.startswith("semantic_model."):
            converted_checkpoint[old_key] = value

        elif old_key.startswith("fc_prior."):
            converted_checkpoint[f"fc.{old_key[len('fc_prior.') :]}"] = value

        elif old_key.startswith("fc_post1."):
            converted_checkpoint[f"fc1.{old_key[len('fc_post1.') :]}"] = value

        elif old_key.startswith("fc_post2."):
            converted_checkpoint[f"fc2.{old_key[len('fc_post2.') :]}"] = value

        elif old_key.startswith("quantizer.vq.layers"):
            new_key = old_key
            for old_sub, new_sub in MAPPING_QUANTIZER.items():
                new_key = new_key.replace(old_sub, new_sub)
            converted_checkpoint[new_key] = value

    return converted_checkpoint


@torch.no_grad()
def convert_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, push_to_hub=None):
    if config_path is not None:
        config = XcodecConfig.from_pretrained(config_path)
    else:
        config = XcodecConfig()

    with torch.device("meta"):
        model = XcodecModel(config)

    logger.info("Loading original checkpoint ...")

    state_dict = safe_load(checkpoint_path)

    # the original checkpoint has weight norm applied
    model.apply_weight_norm()

    logger.info("Converting model ...")

    new_state_dict = convert_old_keys_to_new_keys(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True, assign=True)  # strict=False)

    if len(unexpected_keys) != 0:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    if len(missing_keys) != 0:
        raise ValueError(f"missing keys found: {missing_keys}")

    model.remove_weight_norm()

    model.save_pretrained(pytorch_dump_folder_path)

    feature_extractor = EncodecFeatureExtractor(feature_size=config.audio_channels, sampling_rate=config.sample_rate)

    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(push_to_hub)
        model.push_to_hub(push_to_hub)


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
