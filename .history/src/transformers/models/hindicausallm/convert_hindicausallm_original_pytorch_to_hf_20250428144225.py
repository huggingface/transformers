# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Inc. team. All rights reserved.
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
"""Convert HindiCausalLM checkpoints from the original repository to Hugging Face format."""

import argparse
import json
import os
import re
from pathlib import Path

import torch

from transformers import (
    HindiCausalLMConfig,
    HindiCausalLMForCausalLM,
    HindiCausalLMTokenizer,
)


MAPPING = {
    # Embedding layers
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    # Layer norms
    r"model.layers.(\d+).input_layernorm.weight": r"model.layers.\1.input_layernorm.weight",
    r"model.layers.(\d+).input_layernorm.bias": r"model.layers.\1.input_layernorm.bias",
    r"model.layers.(\d+).post_attention_layernorm.weight": r"model.layers.\1.post_attention_layernorm.weight",
    r"model.layers.(\d+).post_attention_layernorm.bias": r"model.layers.\1.post_attention_layernorm.bias",
    "model.norm.weight": "model.norm.weight",
    "model.norm.bias": "model.norm.bias",
    # Attention layers
    r"model.layers.(\d+).self_attn.q_proj.weight": r"model.layers.\1.self_attn.q_proj.weight",
    r"model.layers.(\d+).self_attn.k_proj.weight": r"model.layers.\1.self_attn.k_proj.weight",
    r"model.layers.(\d+).self_attn.v_proj.weight": r"model.layers.\1.self_attn.v_proj.weight",
    r"model.layers.(\d+).self_attn.o_proj.weight": r"model.layers.\1.self_attn.o_proj.weight",
    # MLP
    r"model.layers.(\d+).mlp.gate_proj.weight": r"model.layers.\1.mlp.gate_proj.weight",
    r"model.layers.(\d+).mlp.up_proj.weight": r"model.layers.\1.mlp.up_proj.weight",
    r"model.layers.(\d+).mlp.down_proj.weight": r"model.layers.\1.mlp.down_proj.weight",
    # LM head
    "lm_head.weight": "lm_head.weight",
}


def convert_hindicausallm_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Convert a HindiCausalLM checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the original checkpoint (model.safetensors or pytorch_model.bin)
        pytorch_dump_folder_path: Path to save the converted model
        config_path: Path to the model config.json file
    """
    # Load model config
    if config_path is None:
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "config.json")

    if not os.path.isfile(config_path):
        raise ValueError(f"Config file not found at {config_path}")

    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Convert to HindiCausalLM config format
    config = HindiCausalLMConfig(
        vocab_size=config_dict.get("vocab_size", 16000),
        hidden_size=config_dict.get("hidden_size", 768),
        num_hidden_layers=config_dict.get("num_hidden_layers", 12),
        num_attention_heads=config_dict.get("num_attention_heads", 16),
        num_key_value_heads=config_dict.get("num_key_value_heads", 4),
        intermediate_size=config_dict.get("intermediate_size", 3072),
        hidden_act=config_dict.get("hidden_act", "silu"),
        max_position_embeddings=config_dict.get("max_position_embeddings", 512),
        initializer_range=config_dict.get("initializer_range", 0.02),
        layer_norm_eps=config_dict.get("layer_norm_eps", 1e-5),
        use_cache=True,
        pad_token_id=config_dict.get("pad_token_id", 0),
        bos_token_id=config_dict.get("bos_token_id", 1),
        eos_token_id=config_dict.get("eos_token_id", 2),
    )

    # Create model with the config
    model = HindiCausalLMForCausalLM(config)

    # Load state dict from checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    # Check file extension to determine loading method
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                original_state_dict = {key: f.get_tensor(key) for key in f.keys()}
        except ImportError:
            raise ImportError("Please install safetensors: `pip install safetensors`")
    else:
        original_state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Check if model state dict is nested
        if "model_state_dict" in original_state_dict:
            original_state_dict = original_state_dict["model_state_dict"]

    # Map original weights to HF weights
    converted_state_dict = {}

    for name, value in original_state_dict.items():
        converted = False

        # Handle regex mapping patterns
        for pattern, mapped in MAPPING.items():
            if re.match(f"^{pattern}$", name):
                hf_name = re.sub(f"^{pattern}$", mapped, name)
                converted_state_dict[hf_name] = value
                converted = True
                break

        if not converted:
            print(f"Warning: Could not convert parameter {name}")

    # Load state dict into model
    model.load_state_dict(converted_state_dict, strict=False)

    # Check for missing keys
    model_state_dict = model.state_dict()
    missing_keys = set(model_state_dict.keys()) - set(converted_state_dict.keys())
    if missing_keys:
        print(f"Warning: Missing keys in converted model: {missing_keys}")

    # Save model and config
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(pytorch_dump_folder_path)

    # Copy tokenizer if available
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.model")
    if os.path.isfile(tokenizer_path):
        print(f"Copying tokenizer from {tokenizer_path}")
        tokenizer = HindiCausalLMTokenizer(tokenizer_path)
        tokenizer.save_pretrained(pytorch_dump_folder_path)
    else:
        print("Tokenizer file not found. Skipping tokenizer conversion.")

    print(f"Model saved to {pytorch_dump_folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to the model config.json file")
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True, help="Path to save the converted model")
    args = parser.parse_args()

    convert_hindicausallm_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path
    )
