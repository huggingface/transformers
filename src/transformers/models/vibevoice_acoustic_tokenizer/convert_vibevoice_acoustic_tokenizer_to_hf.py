# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import logging
import re
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceAcousticTokenizerModel,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    # Encoder
    r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":       r"encoder.stem.conv.conv.",
    r"^model\.acoustic_tokenizer\.encoder\.stages\.0\.":                           r"encoder.stem.stage.",
    r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.":   (r"encoder.conv_layers.\1.conv.conv.", -1),
    r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.":                       (r"encoder.conv_layers.\1.stage.", -1),
    r"^model\.acoustic_tokenizer\.encoder\.head\.conv\.":                          r"encoder.head.",

    # Decoder
    r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.": r"decoder.stem.conv.conv.",
    r"^model\.acoustic_tokenizer\.decoder\.stages\.0\.":                           r"decoder.stem.stage.",
    r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.": (r"decoder.conv_layers.\1.convtr.convtr.", -1),
    r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.":                       (r"decoder.conv_layers.\1.stage.", -1),
    r"^model\.acoustic_tokenizer\.decoder\.head\.conv\.":                          r"decoder.head.",

    # Common patterns (apply after specific patterns)
    r"mixer\.conv\.conv\.conv\.":                                                   r"mixer.conv.",
    r"\.conv\.conv\.conv\.":                                                        r".conv.conv.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key

    # Apply all regex patterns
    for pattern, replacement in STATE_DICT_MAPPING.items():
        # Check if replacement needs index shifting
        if isinstance(replacement, tuple):
            replacement_pattern, index_shift = replacement

            # Use callback to handle index shifting
            def shift_index(match):
                result = replacement_pattern
                for i, group in enumerate(match.groups(), 1):
                    if group and group.isdigit():
                        shifted_idx = int(group) + index_shift
                        result = result.replace(f"\\{i}", str(shifted_idx))
                    else:
                        result = result.replace(f"\\{i}", group)
                return result

            new_key, n = re.subn(pattern, shift_index, new_key)
        else:
            new_key, n = re.subn(pattern, replacement, new_key)

    return new_key


def convert_state_dict(original_state_dict: dict[str, Any]) -> dict[str, Any]:
    new_state_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")

    return new_state_dict


def convert_checkpoint(checkpoint, config_path, push_to_hub, bfloat16, processor_config=None):
    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) Load state dict from safetensors checkpoint
    logger.info(f"Loading checkpoint from {checkpoint}")
    original_state_dict = load_file(checkpoint)

    # 2) Prepare feature extractor
    audio_config = {}
    if processor_config is not None:
        with open(processor_config, "r") as f:
            processor_config = json.load(f)
        audio_config = processor_config.get("audio_processor", {})
    if "sampling_rate" not in audio_config:
        audio_config["sampling_rate"] = 24000
    if "normalize_audio" not in audio_config:
        audio_config["normalize_audio"] = True
    if "target_dB_FS" not in audio_config:
        audio_config["target_dB_FS"] = -25
    if "eps" not in audio_config:
        audio_config["eps"] = 1e-6
    feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor(**audio_config)

    # 3) Prepare model configuration
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Clean up acoustic tokenizer config
    acoustic_config_dict = model_config["acoustic_tokenizer_config"].copy()
    if "encoder_depths" in acoustic_config_dict and isinstance(acoustic_config_dict["encoder_depths"], str):
        acoustic_config_dict["encoder_depths"] = list(map(int, acoustic_config_dict["encoder_depths"].split("-")))
    if "layernorm_eps" in acoustic_config_dict:
        acoustic_config_dict["rms_norm_eps"] = acoustic_config_dict.pop("layernorm_eps")
    if "encoder_ratios" in acoustic_config_dict:
        acoustic_config_dict["downsampling_ratios"] = list(reversed(acoustic_config_dict.pop("encoder_ratios")))
    if "encoder_n_filters" in acoustic_config_dict:
        acoustic_config_dict["num_filters"] = acoustic_config_dict.pop("encoder_n_filters")
    if "encoder_depths" in acoustic_config_dict:
        acoustic_config_dict["depths"] = acoustic_config_dict.pop("encoder_depths")
    if "vae_dim" in acoustic_config_dict:
        acoustic_config_dict["hidden_size"] = acoustic_config_dict.pop("vae_dim")
    if "fix_std" in acoustic_config_dict:
        # Original hardcodes a scaling factor for vae_std
        acoustic_config_dict["vae_std"] = acoustic_config_dict.pop("fix_std") / 0.8

    # Remove unused/constant parameters
    for key in [
        "decoder_depths",
        "decoder_n_filters",
        "decoder_ratios",
        "std_dist_type",
        "pad_mode",
        "conv_bias",
        "causal",
        "mixer_layer",
        "layernorm",
        "disable_last_norm",
        "conv_norm",
        "corpus_normalize",
        "layernorm_elementwise_affine",
    ]:
        acoustic_config_dict.pop(key, None)

    # 4) Convert state dict to match HF model structure
    logger.info("Converting state dict")
    converted_state_dict = convert_state_dict(original_state_dict)

    # 5) Filter for acoustic tokenizer weights
    acoustic_state_dict = {
        k: v for k, v in converted_state_dict.items() if k.startswith("encoder.") or k.startswith("decoder.")
    }

    # 6) Create and save acoustic tokenizer
    logger.info("Creating acoustic tokenizer model")
    acoustic_config = VibeVoiceAcousticTokenizerConfig(**acoustic_config_dict)
    acoustic_model = VibeVoiceAcousticTokenizerModel(acoustic_config).to(dtype)

    # Load weights into HF model
    logger.info("Loading weights into model")
    missing, unexpected = acoustic_model.load_state_dict(acoustic_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"Missing keys: {missing}")

    if push_to_hub:
        logger.info(f"Pushing to hub as {push_to_hub}")
        feature_extractor.push_to_hub(push_to_hub)
        acoustic_model.push_to_hub(push_to_hub)

        gc.collect()
        logger.info("Verifying conversion by reloading model")
        AutoFeatureExtractor.from_pretrained(push_to_hub)
        AutoModel.from_pretrained(push_to_hub, dtype=torch.bfloat16, device_map="auto")
        logger.info("Model reloaded successfully!")
        logger.info("Conversion complete!")


"""
Conversion script to extract the acoustic tokenizer from the original VibeVoice model checkpoint and push a checkpoint
for an `VibeVoiceAcousticTokenizerModel` object.


1) download 1.5B model.
```bash
# -- download checkpoint and configs
# -- download script here: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-download_vibevoice_checkpoint-py
python src/transformers/models/vibevoice/download_vibevoice_checkpoint.py
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json -P /raid/eric/vibevoice
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/preprocessor_config.json -P /raid/eric/vibevoice
```

2) run conversion with:
```
python src/transformers/models/vibevoice_acoustic_tokenizer/convert_vibevoice_acoustic_tokenizer_to_hf.py \
    --checkpoint /raid/eric/vibevoice/VibeVoice-1.5B-combined.safetensors \
    --config_path /raid/eric/vibevoice/config.json \
    --processor_config /raid/eric/vibevoice/preprocessor_config.json \
    --push_to_hub bezzam/VibeVoice-AcousticTokenizer
```

A checkpoint will be pushed to `bezzam/VibeVoice-AcousticTokenizer` on the HF Hub.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, default=None, type=str, help="Original VibeVoice model checkpoint."
    )
    parser.add_argument("--config_path", default=None, type=str, help="Path to config.json of model to convert")
    parser.add_argument(
        "--processor_config", default=None, type=str, help="Path to preprocessor_config.json of model to convert"
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--float32", action="store_true", help="Whether to use float32 precision. Default is bfloat16."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint,
        args.config_path,
        args.push_to_hub,
        bfloat16=not args.float32,
        processor_config=args.processor_config,
    )
