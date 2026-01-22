# coding=utf-8
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
import os
import re

import torch
from safetensors.torch import load_file

from transformers import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    AutoFeatureExtractor,
    AutoModel,
)


def update_state_dict_for_hf_model(state_dict):
    """
    Update the state_dict to match the HuggingFace model structure.
    """
    updated_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle acoustic tokenizer transformations
        if "acoustic_tokenizer.encoder" in key:
            if "downsample_layers.0.0.conv." in key:
                new_key = new_key.replace("downsample_layers.0.0.conv.", "stem.conv.conv.")
            elif "stages.0." in key:
                new_key = new_key.replace("stages.0.", "stem.stage.")
            elif "downsample_layers." in key and not "downsample_layers.0" in key:
                match = re.search(r'downsample_layers\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since downsample_layers[0] became stem
                    new_key = re.sub(r'downsample_layers\.\d+\.0\.conv\.', f'conv_layers.{new_idx}.conv.conv.', new_key)
            elif "stages." in key and not "stages.0." in key:
                match = re.search(r'stages\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since stages[0] became stem
                    new_key = re.sub(r'stages\.\d+\.', f'conv_layers.{new_idx}.stage.', new_key)
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.conv.")
            if ".conv.conv.conv." in new_key:
                new_key = new_key.replace(".conv.conv.conv.", ".conv.conv.")
            elif ".conv.conv." in key and "stem.conv.conv" not in new_key and "conv_layers." not in new_key:
                new_key = new_key.replace(".conv.conv.", ".conv.")
        if "acoustic_tokenizer.decoder" in key:
            if "upsample_layers.0.0.conv.conv." in key:
                new_key = new_key.replace("acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.", "acoustic_tokenizer.decoder.stem.conv.conv.")
            elif "stages.0." in key:
                new_key = new_key.replace("stages.0.", "stem.stage.")
            elif "upsample_layers." in key and not "upsample_layers.0" in key:
                match = re.search(r'upsample_layers\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since upsample_layers[0] became conv0
                    new_key = re.sub(r'upsample_layers\.\d+\.0\.convtr\.convtr\.', f'conv_layers.{new_idx}.convtr.convtr.', new_key)
            elif "stages." in key and not "stages.0." in key:
                match = re.search(r'stages\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since stages[0] became stage0
                    new_key = re.sub(r'stages\.\d+\.', f'conv_layers.{new_idx}.stage.', new_key)
            if "head.conv." in key:
                new_key = new_key.replace("head.conv.", "head.")
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.conv.")

        updated_state_dict[new_key] = value

    return updated_state_dict


def convert_checkpoint(
    checkpoint, config_path, push_to_hub, bfloat16, processor_config=None
):
    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) Load state dict from safetensors checkpoint
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
    # -- Load
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # clean up acoustic tokenizer config
    model_config["acoustic_tokenizer_config"]["encoder_depths"] = list(
        map(int, model_config["acoustic_tokenizer_config"]["encoder_depths"].split("-"))
    )
    model_config["acoustic_tokenizer_config"]["rms_norm_eps"] = model_config["acoustic_tokenizer_config"].pop("layernorm_eps")
    # -- reverse order of ratios here instead of in modeling (as done in original)
    model_config["acoustic_tokenizer_config"]["downsampling_ratios"] = list(
        reversed(model_config["acoustic_tokenizer_config"].pop("encoder_ratios"))
    )
    model_config["acoustic_tokenizer_config"]["n_filters"] = model_config["acoustic_tokenizer_config"].pop(
        "encoder_n_filters"
    )
    model_config["acoustic_tokenizer_config"]["depths"] = model_config["acoustic_tokenizer_config"].pop(
        "encoder_depths"
    )
    model_config["acoustic_tokenizer_config"]["hidden_size"] = model_config["acoustic_tokenizer_config"].pop("vae_dim")
    model_config["acoustic_tokenizer_config"]["bias"] = model_config["acoustic_tokenizer_config"].pop("conv_bias")
    # -- original hardcodes a scaling factor for vae_std: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L963
    model_config["acoustic_tokenizer_config"]["vae_std"] = (
        model_config["acoustic_tokenizer_config"].pop("fix_std") / 0.8
    )
    # -- remove decoder parameters as they can be derived from encoder ones
    model_config["acoustic_tokenizer_config"].pop("decoder_depths")
    model_config["acoustic_tokenizer_config"].pop("decoder_n_filters")
    model_config["acoustic_tokenizer_config"].pop("decoder_ratios")
    # -- remove unused / constant parameters that lead to unused code paths removed in HF model
    model_config["acoustic_tokenizer_config"].pop("std_dist_type")  # always Gaussian
    model_config["acoustic_tokenizer_config"].pop("pad_mode")   # always constant
    model_config["acoustic_tokenizer_config"].pop("causal")     # always True
    model_config["acoustic_tokenizer_config"].pop("mixer_layer")
    model_config["acoustic_tokenizer_config"].pop("layernorm")
    model_config["acoustic_tokenizer_config"].pop("disable_last_norm")
    model_config["acoustic_tokenizer_config"].pop("conv_norm")
    model_config["acoustic_tokenizer_config"].pop("corpus_normalize")
    model_config["acoustic_tokenizer_config"].pop("layernorm_elementwise_affine")

    # 4) Update state dict to match HF model structure
    updated_state_dict = update_state_dict_for_hf_model(original_state_dict)

    # 5) Create and save acoustic tokenizer
    print("\n=== Creating acoustic tokenizer ===")
    acoustic_config = VibeVoiceAcousticTokenizerConfig(**model_config["acoustic_tokenizer_config"])
    acoustic_model = VibeVoiceAcousticTokenizerModel(acoustic_config).to(dtype)
    # -- filter for acoustic tokenizer weights
    prefix = "model.acoustic_tokenizer"
    acoustic_state_dict = {
        k[len(prefix) + 1 :]: v  # +1 to remove the dot after the prefix
        for k, v in updated_state_dict.items()
        if k.startswith(prefix)
    }
    # -- load into HF model
    missing, unexpected = acoustic_model.load_state_dict(acoustic_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    if push_to_hub:
        hub_repo = push_to_hub.split("/")[0] + "/VibeVoice-AcousticTokenizer"
        print(f"------ Pushing to hub as {hub_repo} ------")
        feature_extractor.push_to_hub(hub_repo)
        acoustic_model.push_to_hub(hub_repo)

    # 6) Check model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    AutoFeatureExtractor.from_pretrained(push_to_hub)
    AutoModel.from_pretrained(push_to_hub, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


"""
Conversion script to convert extract acoustic tokenizer from the original VibeVoice model checkpoint
and push a checkpoint for an `VibeVoiceAcousticTokenizerModel` object.


First download 1.5B model.
```bash
# -- download checkpoint and configs
# -- download script here: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-download_vibevoice_checkpoint-py
python src/transformers/models/vibevoice/download_vibevoice_checkpoint.py
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json -P /raid/eric/vibevoice
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/preprocessor_config.json -P /raid/eric/vibevoice
```

Then we can run conversion with:
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
