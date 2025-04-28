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
from shutil import copyfile

import torch

# Ensure these imports work relative to the script's location if run outside the library
try:
    from transformers import (
        HindiCausalLMConfig,
        HindiCausalLMForCausalLM,
        HindiCausalLMTokenizer,
    )
except ImportError:
    # Allow running the script standalone assuming transformers is installed
    print("transformers library not found directly, attempting import assuming it's installed.")
    from transformers import (
        HindiCausalLMConfig,
        HindiCausalLMForCausalLM,
        HindiCausalLMTokenizer,
    )


# Define the mapping from original keys to HF keys
# This is a common pattern for decoder-only models
# Adjust if the original model has different naming conventions
MAPPING = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    r"model.layers.(\d+).input_layernorm.weight": r"model.layers.\1.input_layernorm.weight",
    # Add bias if the original Layernorm used it and HF one doesn't (or vice-versa)
    # r"model.layers.(\d+).input_layernorm.bias": r"model.layers.\1.input_layernorm.bias",
    r"model.layers.(\d+).self_attn.q_proj.weight": r"model.layers.\1.self_attn.q_proj.weight",
    r"model.layers.(\d+).self_attn.k_proj.weight": r"model.layers.\1.self_attn.k_proj.weight",
    r"model.layers.(\d+).self_attn.v_proj.weight": r"model.layers.\1.self_attn.v_proj.weight",
    r"model.layers.(\d+).self_attn.o_proj.weight": r"model.layers.\1.self_attn.o_proj.weight",
    r"model.layers.(\d+).post_attention_layernorm.weight": r"model.layers.\1.post_attention_layernorm.weight",
    # r"model.layers.(\d+).post_attention_layernorm.bias": r"model.layers.\1.post_attention_layernorm.bias",
    r"model.layers.(\d+).mlp.gate_proj.weight": r"model.layers.\1.mlp.gate_proj.weight",
    r"model.layers.(\d+).mlp.up_proj.weight": r"model.layers.\1.mlp.up_proj.weight",
    r"model.layers.(\d+).mlp.down_proj.weight": r"model.layers.\1.mlp.down_proj.weight",
    "model.norm.weight": "model.norm.weight",
    # "model.norm.bias": "model.norm.bias",
    "lm_head.weight": "lm_head.weight",  # Assumes lm_head is separate, might be tied in original
}


def convert_hindicausallm_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Convert a HindiCausalLM checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the original checkpoint (e.g., model.pth, pytorch_model.bin)
        pytorch_dump_folder_path: Path to save the converted model
        config_path: Path to the model config.json file (optional, attempts to find in checkpoint dir)
    """
    # Load model config
    if config_path is None:
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "config.json")

    if not os.path.isfile(config_path):
        # Try finding params.json as often used in Llama-based models
        params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")
        if os.path.isfile(params_path):
            print(f"Config file not found at {config_path}, using params.json from {params_path}")
            with open(params_path) as f:
                params = json.load(f)
            # Manual mapping from params.json keys to HindiCausalLMConfig args
            # This needs adjustment based on the actual params.json content
            config_dict = {
                "hidden_size": params.get("dim"),
                "num_hidden_layers": params.get("n_layers"),
                "num_attention_heads": params.get("n_heads"),
                "num_key_value_heads": params.get(
                    "n_kv_heads", params.get("n_heads")
                ),  # Handle GQA
                "intermediate_size": params.get("multiple_of", 256)
                * params.get("dim")
                * 4
                // 3
                // params.get("multiple_of", 256),  # Approximate SwiGLU size
                "layer_norm_eps": params.get("norm_eps"),
                "max_position_embeddings": params.get("max_seq_len", 512),  # Default if not present
                # Defaults for missing keys
                "vocab_size": params.get("vocab_size", 16000),
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "use_cache": True,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "rope_theta": params.get("rope_theta", 10000.0),
            }
            print(f"Inferred config from params.json: {config_dict}")
        else:
            raise ValueError(f"Config file (config.json or params.json) not found near {checkpoint_path}")
    else:
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config_dict = json.load(f)

    # Convert to HindiCausalLM config format
    # Use get() with defaults for robustness
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
        use_cache=config_dict.get("use_cache", True),
        pad_token_id=config_dict.get("pad_token_id", 0),
        bos_token_id=config_dict.get("bos_token_id", 1),
        eos_token_id=config_dict.get("eos_token_id", 2),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
        rope_theta=config_dict.get("rope_theta", 10000.0),
        attention_dropout=config_dict.get("attention_dropout", 0.0),  # Added attention dropout
    )

    # Create model with the config (low_cpu_mem_usage can be helpful for large models)
    print("Instantiating HF model...")
    # model = HindiCausalLMForCausalLM(config, low_cpu_mem_usage=True) # Requires accelerate
    model = HindiCausalLMForCausalLM(config)
    model.eval()  # Set to evaluation mode

    # Load state dict from checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    # Check file extension to determine loading method
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open

            original_state_dict = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)
        except ImportError:
            raise ImportError("Please install safetensors: `pip install safetensors`")
    else:
        # Assume it's a PyTorch .bin or .pth file
        original_state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Check common patterns for nested state dicts
        if "state_dict" in original_state_dict:
            original_state_dict = original_state_dict["state_dict"]
        elif "model_state_dict" in original_state_dict:
            original_state_dict = original_state_dict["model_state_dict"]
        elif "model" in original_state_dict and isinstance(original_state_dict["model"], dict):
            original_state_dict = original_state_dict["model"]

    # Map original weights to HF weights
    print("Converting state dict...")
    converted_state_dict = {}
    unmatched_keys = set(original_state_dict.keys())

    for name, value in original_state_dict.items():
        converted = False
        # Try direct match first
        if name in MAPPING.values():
            converted_state_dict[name] = value
            unmatched_keys.discard(name)
            converted = True
        else:
            # Handle regex mapping patterns
            for pattern, mapped_pattern in MAPPING.items():
                if re.match(f"^{pattern}$", name):
                    hf_name = re.sub(f"^{pattern}$", mapped_pattern, name)
                    converted_state_dict[hf_name] = value
                    unmatched_keys.discard(name)
                    converted = True
                    break

        if not converted:
            # Keep track of keys that didn't match any pattern if needed later
            pass  # print(f"Warning: Could not convert parameter {name}")

    if unmatched_keys:
        print(f"Warning: The following keys from the original checkpoint were not converted: {unmatched_keys}")

    # Load state dict into model
    print("Loading converted state dict into HF model...")
    load_result = model.load_state_dict(converted_state_dict, strict=False)

    # Check for missing/unexpected keys
    if load_result.missing_keys:
        print(f"Warning: Missing keys in HF model: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: Unexpected keys in converted state dict: {load_result.unexpected_keys}")

    # Save model and config
    print(f"Saving converted model to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(pytorch_dump_folder_path)
    config.save_pretrained(pytorch_dump_folder_path)  # Explicitly save config too

    # Copy tokenizer if available
    # Check for both tokenizer.model and tokenizer.json
    tokenizer_model_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.model")
    tokenizer_json_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.json")
    tokenizer_config_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer_config.json")

    if os.path.isfile(tokenizer_model_path):
        try:
            print(f"Saving tokenizer from {tokenizer_model_path}")
            # Use the slow tokenizer to save consistently if only .model is present
            tokenizer = HindiCausalLMTokenizer(tokenizer_model_path)
            # Copy additional config files if they exist
            if os.path.isfile(tokenizer_config_path):
                copyfile(tokenizer_config_path, os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"))
            tokenizer.save_pretrained(pytorch_dump_folder_path)
        except Exception as e:
            print(f"Could not load or save tokenizer: {e}")
    elif os.path.isfile(tokenizer_json_path):
        # If only tokenizer.json exists, copy it directly
        print(f"Copying tokenizer files from {os.path.dirname(checkpoint_path)}")
        copyfile(tokenizer_json_path, os.path.join(pytorch_dump_folder_path, "tokenizer.json"))
        if os.path.isfile(tokenizer_config_path):
            copyfile(tokenizer_config_path, os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"))
        # Copy other potential tokenizer files like special_tokens_map.json
        special_tokens_map_path = os.path.join(os.path.dirname(checkpoint_path), "special_tokens_map.json")
        if os.path.isfile(special_tokens_map_path):
            copyfile(
                special_tokens_map_path, os.path.join(pytorch_dump_folder_path, "special_tokens_map.json")
            )

    else:
        print("Tokenizer file (tokenizer.model or tokenizer.json) not found. Skipping tokenizer conversion.")

    print(f"Model successfully converted and saved to {pytorch_dump_folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HindiCausalLM checkpoints to Hugging Face format.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original PyTorch checkpoint file (.bin, .pth, .safetensors)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the model config.json or params.json file (optional, defaults to checkpoint directory)",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        required=True,
        help="Path to the output directory for the converted Hugging Face model",
    )
    args = parser.parse_args()

    convert_hindicausallm_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)