#!/usr/bin/env python3
# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Convert Fun-ASR-Nano checkpoint to HuggingFace Transformers format."""

import argparse
import json
import os
import re

import torch
import yaml

from transformers import (
    AutoTokenizer,
    Qwen3Config,
)

from .configuration_fun_asr_nano import (
    FunAsrNanoAdaptorConfig,
    FunAsrNanoConfig,
    FunAsrNanoCtcConfig,
    FunAsrNanoEncoderConfig,
)
from .modeling_fun_asr_nano import FunAsrNanoForConditionalGeneration


def load_original_checkpoint(checkpoint_path: str) -> dict:
    """Load the original FunASR model.pt checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def convert_encoder_key(key: str) -> str | None:
    """Convert original encoder key to HF format.

    Original: audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight
    Target:   audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight
    """
    if key.startswith("audio_encoder."):
        return key
    return None


def convert_adaptor_key(key: str) -> str | None:
    """Convert original adaptor key to HF format.

    Original: audio_adaptor.linear1.weight
    Target:   audio_adaptor.linear1.weight
    """
    if key.startswith("audio_adaptor."):
        return key
    return None


def convert_ctc_key(key: str) -> str | None:
    """Convert original CTC decoder key to HF format.

    Original: ctc_decoder.linear1.weight -> ctc_decoder.linear1.weight
    Original: ctc.ctc_lo.weight -> ctc_decoder.ctc_lo.weight
    """
    if key.startswith("ctc_decoder."):
        return key
    if key.startswith("ctc."):
        return "ctc_decoder." + key[4:]
    return None


def convert_llm_key(key: str) -> str | None:
    """Convert original LLM key to HF format.

    Original: llm.model.layers.0.self_attn.q_proj.weight
    Target:   language_model.model.layers.0.self_attn.q_proj.weight
    """
    if key.startswith("llm."):
        return "language_model." + key[4:]
    return None


def build_config_from_yaml(config_yaml_path: str, qwen3_config_path: str) -> FunAsrNanoConfig:
    """Build HF config from original config.yaml."""
    with open(config_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Audio encoder config
    enc_conf = cfg.get("audio_encoder_conf", {})
    audio_encoder_config = FunAsrNanoEncoderConfig(
        input_size=enc_conf.get("input_layer_size", 560),  # 80 * 7 (lfr_m)
        output_size=enc_conf.get("output_size", 512),
        attention_heads=enc_conf.get("attention_heads", 4),
        linear_units=enc_conf.get("linear_units", 2048),
        num_blocks=enc_conf.get("num_blocks", 50),
        tp_blocks=enc_conf.get("tp_blocks", 20),
        dropout_rate=enc_conf.get("dropout_rate", 0.1),
        positional_dropout_rate=enc_conf.get("positional_dropout_rate", 0.1),
        attention_dropout_rate=enc_conf.get("attention_dropout_rate", 0.0),
        kernel_size=enc_conf.get("kernel_size", 11),
        sanm_shift=enc_conf.get("sanm_shfit", 0),
    )

    # Adaptor config
    adp_conf = cfg.get("audio_adaptor_conf", {})
    adaptor_config = FunAsrNanoAdaptorConfig(
        downsample_rate=adp_conf.get("downsample_rate", 1),
        encoder_dim=adp_conf.get("encoder_dim", 512),
        llm_dim=adp_conf.get("llm_dim", 1024),
        ffn_dim=adp_conf.get("ffn_dim", 2048),
        num_layers=adp_conf.get("n_layer", 2),
        attention_heads=8,
        dropout_rate=0.0,
        use_low_frame_rate=adp_conf.get("use_low_frame_rate", True),
    )

    # CTC config
    ctc_conf = cfg.get("ctc_decoder_conf", {})
    ctc_config = FunAsrNanoCtcConfig(
        vocab_size=cfg.get("ctc_vocab_size", 60515),
        encoder_dim=ctc_conf.get("encoder_dim", 512),
        decoder_dim=ctc_conf.get("llm_dim", 512),
        ffn_dim=ctc_conf.get("ffn_dim", 2048),
        num_layers=ctc_conf.get("n_layer", 5),
        downsample_rate=ctc_conf.get("downsample_rate", 1),
        blank_id=cfg.get("ctc_conf", {}).get("blank_id", 60514),
    )

    # Text (LLM) config
    with open(os.path.join(qwen3_config_path, "config.json"), "r") as f:
        qwen3_cfg = json.load(f)
    text_config = Qwen3Config(**qwen3_cfg)

    config = FunAsrNanoConfig(
        audio_encoder_config=audio_encoder_config,
        adaptor_config=adaptor_config,
        text_config=text_config,
        ctc_config=ctc_config,
    )

    return config


def convert_checkpoint(
    model_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """Convert Fun-ASR-Nano checkpoint to HuggingFace format.

    Args:
        model_path: Path to the original Fun-ASR-Nano model directory (containing config.yaml, model.pt, Qwen3-0.6B/).
        output_path: Output directory for the converted model.
        push_to_hub: Whether to push the converted model to HuggingFace Hub.
        hub_model_id: Hub model ID for pushing.
    """
    config_yaml_path = os.path.join(model_path, "config.yaml")
    checkpoint_path = os.path.join(model_path, "model.pt")
    qwen3_path = os.path.join(model_path, "Qwen3-0.6B")

    print(f"Building config from {config_yaml_path}...")
    config = build_config_from_yaml(config_yaml_path, qwen3_path)

    print(f"Loading original checkpoint from {checkpoint_path}...")
    original_state_dict = load_original_checkpoint(checkpoint_path)

    print("Converting state dict keys...")
    converted_state_dict = {}
    unconverted_keys = []

    for key, value in original_state_dict.items():
        new_key = None

        # Try each converter
        new_key = convert_encoder_key(key)
        if new_key is None:
            new_key = convert_adaptor_key(key)
        if new_key is None:
            new_key = convert_ctc_key(key)
        if new_key is None:
            new_key = convert_llm_key(key)

        if new_key is not None:
            converted_state_dict[new_key] = value
        else:
            unconverted_keys.append(key)

    if unconverted_keys:
        print(f"Warning: {len(unconverted_keys)} keys were not converted:")
        for k in unconverted_keys[:20]:
            print(f"  - {k}")
        if len(unconverted_keys) > 20:
            print(f"  ... and {len(unconverted_keys) - 20} more")

    print("Initializing HF model...")
    model = FunAsrNanoForConditionalGeneration(config)

    print("Loading converted weights...")
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)

    if missing:
        print(f"Missing keys ({len(missing)}):")
        for k in missing[:20]:
            print(f"  - {k}")

    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:20]:
            print(f"  - {k}")

    print(f"Saving model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    # Copy tokenizer
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(qwen3_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")

    if push_to_hub and hub_model_id:
        print(f"Pushing to hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Fun-ASR-Nano to HuggingFace format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the original Fun-ASR-Nano model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID",
    )
    args = parser.parse_args()

    convert_checkpoint(
        model_path=args.model_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
