# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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
Convert VibeVoice checkpoints from the original repository to HuggingFace Transformers format.

Usage:
    python convert_vibevoice_weights_to_hf.py \
        --model_id microsoft/VibeVoice-1.5B \
        --output_dir ./vibevoice-1.5b-hf

    python convert_vibevoice_weights_to_hf.py \
        --model_id microsoft/VibeVoice-Realtime-0.5B \
        --output_dir ./vibevoice-realtime-0.5b-hf
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Weight name mappings from original to HF format
# Format: (original_pattern, replacement_pattern)
WEIGHT_MAPPING = [
    # Language model (Qwen2) - mostly unchanged
    (r"^model\.language_model\.", "model.language_model."),
    # Acoustic tokenizer encoder
    (r"^model\.acoustic_tokenizer\.encoder\.conv_in\.", "model.acoustic_tokenizer.encoder.conv_in.conv."),
    (
        r"^model\.acoustic_tokenizer\.encoder\.downsample\.(\d+)\.",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.downsample.conv.",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.mixer\.",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.blocks.\2.mixer.conv.",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.ffn\.",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.blocks.\2.ffn.",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.norm1\.",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.blocks.\2.norm1.",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.norm2\.",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.blocks.\2.norm2.",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.gamma1",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.blocks.\2.layer_scale1",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.gamma2",
        r"model.acoustic_tokenizer.encoder.down_blocks.\1.blocks.\2.layer_scale2",
    ),
    (
        r"^model\.acoustic_tokenizer\.encoder\.head\.(\d+)\.blocks\.(\d+)\.",
        r"model.acoustic_tokenizer.encoder.final_blocks.\2.",
    ),
    (r"^model\.acoustic_tokenizer\.encoder\.head\.proj\.", "model.acoustic_tokenizer.encoder.proj_out."),
    (r"^model\.acoustic_tokenizer\.encoder\.head\.norm\.", "model.acoustic_tokenizer.encoder.final_norm."),
    # Acoustic tokenizer decoder
    (r"^model\.acoustic_tokenizer\.decoder\.conv_out\.", "model.acoustic_tokenizer.decoder.conv_out.conv."),
    (
        r"^model\.acoustic_tokenizer\.decoder\.upsample\.(\d+)\.",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.upsample.conv.",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.blocks\.(\d+)\.mixer\.",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.blocks.\2.mixer.conv.",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.blocks\.(\d+)\.ffn\.",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.blocks.\2.ffn.",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.blocks\.(\d+)\.norm1\.",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.blocks.\2.norm1.",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.blocks\.(\d+)\.norm2\.",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.blocks.\2.norm2.",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.blocks\.(\d+)\.gamma1",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.blocks.\2.layer_scale1",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.blocks\.(\d+)\.gamma2",
        r"model.acoustic_tokenizer.decoder.up_blocks.\1.blocks.\2.layer_scale2",
    ),
    (
        r"^model\.acoustic_tokenizer\.decoder\.head\.(\d+)\.blocks\.(\d+)\.",
        r"model.acoustic_tokenizer.decoder.initial_blocks.\2.",
    ),
    (r"^model\.acoustic_tokenizer\.decoder\.head\.proj\.", "model.acoustic_tokenizer.decoder.proj_in."),
    # Semantic tokenizer (1.5B only)
    (r"^model\.semantic_tokenizer\.encoder\.", "model.semantic_tokenizer.encoder."),
    # Connectors
    (r"^model\.acoustic_connector\.", "model.speech_connector.proj."),
    (r"^model\.semantic_connector\.", "model.semantic_connector.proj."),
    # Prediction head -> Diffusion head
    (r"^model\.prediction_head\.t_embedder\.", "model.diffusion_head.t_embedder."),
    (r"^model\.prediction_head\.x_proj\.", "model.diffusion_head.x_proj."),
    (r"^model\.prediction_head\.layers\.(\d+)\.", r"model.diffusion_head.layers.\1."),
    (r"^model\.prediction_head\.final_layer\.", "model.diffusion_head.final_layer."),
    # LM head
    (r"^lm_head\.", "lm_head."),
]


def convert_weight_name(original_name: str) -> str:
    """Convert original weight name to HF format."""
    new_name = original_name

    for pattern, replacement in WEIGHT_MAPPING:
        new_name = re.sub(pattern, replacement, new_name)

    return new_name


def load_original_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Load weights from safetensors files."""
    weights = {}

    # Find all safetensors files
    safetensors_files = sorted(Path(model_path).glob("*.safetensors"))

    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    logger.info(f"Found {len(safetensors_files)} safetensors files")

    for sf_file in safetensors_files:
        logger.info(f"Loading {sf_file.name}...")
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    return weights


def convert_weights(original_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert weight names from original to HF format."""
    converted_weights = {}
    unmapped_keys = []

    for original_name, tensor in original_weights.items():
        new_name = convert_weight_name(original_name)

        if new_name == original_name:
            # Check if it's expected to stay the same or if it's unmapped
            if not any(original_name.startswith(prefix) for prefix in ["model.language_model.", "lm_head."]):
                unmapped_keys.append(original_name)

        converted_weights[new_name] = tensor

    if unmapped_keys:
        logger.warning(f"The following keys were not mapped (kept as-is): {unmapped_keys[:10]}...")
        if len(unmapped_keys) > 10:
            logger.warning(f"... and {len(unmapped_keys) - 10} more")

    return converted_weights


def convert_config(original_config: dict[str, Any], is_streaming: bool = False) -> dict[str, Any]:
    """Convert original config to HF format."""

    # Base config structure
    hf_config = {
        "model_type": "vibevoice_streaming" if is_streaming else "vibevoice",
        "architectures": [
            "VibeVoiceStreamingForConditionalGeneration" if is_streaming else "VibeVoiceForConditionalGeneration"
        ],
    }

    # Extract decoder config (Qwen2)
    decoder_config = {
        "model_type": original_config.get("model_type", "qwen2"),
        "hidden_size": original_config.get("hidden_size", 1536),
        "intermediate_size": original_config.get("intermediate_size", 8960),
        "num_hidden_layers": original_config.get("num_hidden_layers", 28),
        "num_attention_heads": original_config.get("num_attention_heads", 12),
        "num_key_value_heads": original_config.get("num_key_value_heads", 2),
        "hidden_act": original_config.get("hidden_act", "silu"),
        "max_position_embeddings": original_config.get("max_position_embeddings", 32768),
        "initializer_range": original_config.get("initializer_range", 0.02),
        "rms_norm_eps": original_config.get("rms_norm_eps", 1e-6),
        "use_cache": original_config.get("use_cache", True),
        "vocab_size": original_config.get("vocab_size", 151936),
        "tie_word_embeddings": original_config.get("tie_word_embeddings", False),
        "rope_theta": original_config.get("rope_theta", 1000000.0),
        "attention_dropout": original_config.get("attention_dropout", 0.0),
    }
    hf_config["decoder_config"] = decoder_config

    # Extract acoustic tokenizer config
    acoustic_config = original_config.get("acoustic_tokenizer_config", {})
    hf_config["acoustic_tokenizer_config"] = {
        "model_type": "vibevoice_acoustic_codec",
        "channels": acoustic_config.get("channels", 1),
        "vae_dim": acoustic_config.get("vae_dim", 64),
        "encoder_n_filters": acoustic_config.get("encoder_n_filters", 32),
        "decoder_n_filters": acoustic_config.get("decoder_n_filters", 32),
        "encoder_ratios": acoustic_config.get("encoder_ratios", [8, 5, 5, 4, 2, 2]),
        "decoder_ratios": acoustic_config.get("decoder_ratios", [8, 5, 5, 4, 2, 2]),
        "encoder_depths": acoustic_config.get("encoder_depths", "3-3-3-3-3-3-8"),
        "causal": acoustic_config.get("causal", True),
        "layernorm": acoustic_config.get("layernorm", "RMSNorm"),
        "layernorm_eps": acoustic_config.get("layernorm_eps", 1e-5),
        "mixer_layer": acoustic_config.get("mixer_layer", "depthwise_conv"),
        "fix_std": acoustic_config.get("fix_std", 0.5),
        "std_dist_type": acoustic_config.get("std_dist_type", "gaussian"),
    }

    # Extract semantic tokenizer config (only for 1.5B)
    if not is_streaming:
        semantic_config = original_config.get("semantic_tokenizer_config", {})
        hf_config["semantic_tokenizer_config"] = {
            "model_type": "vibevoice_semantic_encoder",
            "channels": semantic_config.get("channels", 1),
            "vae_dim": semantic_config.get("vae_dim", 128),
            "encoder_n_filters": semantic_config.get("encoder_n_filters", 32),
            "encoder_ratios": semantic_config.get("encoder_ratios", [8, 5, 5, 4, 2, 2]),
            "encoder_depths": semantic_config.get("encoder_depths", "3-3-3-3-3-3-8"),
            "causal": semantic_config.get("causal", True),
            "layernorm": semantic_config.get("layernorm", "RMSNorm"),
            "fix_std": semantic_config.get("fix_std", 0),
            "std_dist_type": semantic_config.get("std_dist_type", "none"),
        }
    else:
        hf_config["semantic_tokenizer_config"] = None

    # Extract diffusion head config
    diffusion_config = original_config.get("prediction_head_config", original_config.get("diffusion_head_config", {}))
    hf_config["diffusion_head_config"] = {
        "model_type": "vibevoice_diffusion_head",
        "hidden_size": diffusion_config.get("hidden_size", decoder_config["hidden_size"]),
        "latent_size": diffusion_config.get("latent_size", 64),
        "speech_vae_dim": diffusion_config.get("speech_vae_dim", 64),
        "head_layers": diffusion_config.get("head_layers", 4),
        "head_ffn_ratio": diffusion_config.get("head_ffn_ratio", 3.0),
        "rms_norm_eps": diffusion_config.get("rms_norm_eps", 1e-5),
        "diffusion_type": diffusion_config.get("diffusion_type", "ddpm"),
        "ddpm_num_steps": diffusion_config.get("ddpm_num_steps", 1000),
        "ddpm_num_inference_steps": diffusion_config.get("ddpm_num_inference_steps", 20),
        "ddpm_beta_schedule": diffusion_config.get("ddpm_beta_schedule", "cosine"),
        "prediction_type": diffusion_config.get("prediction_type", "v_prediction"),
    }

    # Other config values
    hf_config["acoustic_vae_dim"] = original_config.get("acoustic_vae_dim", 64)
    hf_config["semantic_vae_dim"] = original_config.get("semantic_vae_dim", 128 if not is_streaming else 0)
    hf_config["sampling_rate"] = original_config.get("sampling_rate", 24000)
    hf_config["num_speakers"] = original_config.get("num_speakers", 4 if not is_streaming else 1)

    if is_streaming:
        hf_config["tts_backbone_num_hidden_layers"] = original_config.get("tts_backbone_num_hidden_layers", 20)

    return hf_config


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice weights to HuggingFace format")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/VibeVoice-1.5B",
        help="HuggingFace model ID or local path to original checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted model",
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
        help="Model ID for pushing to Hub (defaults to output_dir name)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if this is the streaming model
    is_streaming = "realtime" in args.model_id.lower() or "streaming" in args.model_id.lower()

    logger.info(f"Converting {'streaming' if is_streaming else 'standard'} VibeVoice model")

    # Download or use local path
    if os.path.isdir(args.model_id):
        model_path = args.model_id
    else:
        logger.info(f"Downloading model from {args.model_id}...")
        model_path = snapshot_download(
            args.model_id,
            allow_patterns=["*.safetensors", "*.json"],
        )

    # Load original config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        original_config = json.load(f)

    logger.info("Converting config...")
    hf_config = convert_config(original_config, is_streaming=is_streaming)

    # Save converted config
    config_output_path = output_dir / "config.json"
    with open(config_output_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    logger.info(f"Saved config to {config_output_path}")

    # Load and convert weights
    logger.info("Loading original weights...")
    original_weights = load_original_weights(model_path)

    logger.info(f"Loaded {len(original_weights)} weight tensors")

    logger.info("Converting weight names...")
    converted_weights = convert_weights(original_weights)

    # Free memory
    del original_weights
    gc.collect()

    # Save converted weights
    logger.info("Saving converted weights...")
    weights_output_path = output_dir / "model.safetensors"
    save_file(converted_weights, weights_output_path)
    logger.info(f"Saved weights to {weights_output_path}")

    # Copy tokenizer files
    logger.info("Copying tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer saved")
    except Exception as e:
        logger.warning(f"Could not copy tokenizer: {e}")
        logger.info("You may need to copy tokenizer files manually")

    # Create a simple preprocessor config for the processor
    preprocessor_config = {
        "processor_class": "VibeVoiceProcessor",
        "sampling_rate": hf_config.get("sampling_rate", 24000),
    }
    with open(output_dir / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f, indent=2)

    logger.info(f"Conversion complete! Model saved to {output_dir}")

    # Push to hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi

        hub_model_id = args.hub_model_id or output_dir.name
        logger.info(f"Pushing to Hub as {hub_model_id}...")

        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=hub_model_id,
            repo_type="model",
        )
        logger.info(f"Pushed to https://huggingface.co/{hub_model_id}")


if __name__ == "__main__":
    main()
