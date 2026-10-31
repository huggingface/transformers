# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
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
Rewrite MiniMaxAI/MiniMax-VL-01's original `config.json` into the native
`MiniMaxVL01Config` layout. This script writes configuration only; it does not
load, copy, or rewrite weights, tokenizers, or processor files. The output directory
is therefore not a complete model checkpoint.

Weight renaming and MoE expert conversion are handled by the existing entries in
`src/transformers/conversion_mapping.py` when loading the original weights with
the converted configuration.

Sample usage:
    python src/transformers/models/minimax_vl_01/convert_minimax_vl_01_weights_to_hf.py \
        /path/to/MiniMax-VL-01 /path/to/MiniMax-VL-01-native-config
"""

import argparse
import copy
import json
import os

from transformers import MiniMaxVL01Config


def _convert_text_config(text_config: dict) -> dict:
    """Convert the original MiniMax-VL-01 checkpoint's nested text config to native MiniMax field names."""
    text_config = dict(text_config)
    model_type = text_config.get("model_type")
    legacy_keys = {
        "attn_type_list",
        "layernorm_full_attention_alpha",
        "layernorm_full_attention_beta",
        "layernorm_linear_attention_alpha",
        "layernorm_linear_attention_beta",
        "layernorm_mlp_alpha",
        "layernorm_mlp_beta",
        "postnorm",
        "rotary_dim",
        "shared_intermediate_size",
        "shared_moe_mode",
    }
    is_legacy = model_type in {"minimax_text_01", "MiniMaxText01"} or bool(legacy_keys & text_config.keys())
    if not is_legacy:
        return text_config

    if text_config.pop("postnorm", True) is not True:
        raise ValueError("MiniMax-VL-01 only supports the released MiniMax-Text-01 `postnorm=True` layout.")

    shared_intermediate_size = text_config.pop("shared_intermediate_size", [0])
    if isinstance(shared_intermediate_size, (list, tuple)):
        if len(shared_intermediate_size) != 1:
            raise ValueError("`shared_intermediate_size` must contain exactly one value.")
        shared_intermediate_size = shared_intermediate_size[0]
    if shared_intermediate_size != 0:
        raise ValueError("MiniMax-VL-01 does not support the experimental shared-MoE branch.")
    text_config.pop("shared_moe_mode", None)

    attention_types = text_config.pop("attn_type_list", None)
    if attention_types is not None:
        if any(attention_type not in (0, 1) for attention_type in attention_types):
            raise ValueError("`attn_type_list` entries must be 0 (linear) or 1 (full attention).")
        num_hidden_layers = text_config.get("num_hidden_layers", len(attention_types))
        if len(attention_types) != num_hidden_layers:
            raise ValueError(
                f"`attn_type_list` has {len(attention_types)} entries but `num_hidden_layers` is {num_hidden_layers}."
            )
        text_config["layer_types"] = [
            "linear_attention" if attention_type == 0 else "full_attention" for attention_type in attention_types
        ]

    legacy_factor_mapping = {
        "layernorm_full_attention_alpha": "full_attn_alpha_factor",
        "layernorm_full_attention_beta": "full_attn_beta_factor",
        "layernorm_linear_attention_alpha": "linear_attn_alpha_factor",
        "layernorm_linear_attention_beta": "linear_attn_beta_factor",
        "layernorm_mlp_alpha": "mlp_alpha_factor",
        "layernorm_mlp_beta": "mlp_beta_factor",
    }
    for legacy_name, native_name in legacy_factor_mapping.items():
        if legacy_name in text_config:
            legacy_value = text_config.pop(legacy_name)
            if native_name in text_config and text_config[native_name] != legacy_value:
                raise ValueError(f"Conflicting `{legacy_name}` and `{native_name}` values.")
            text_config[native_name] = legacy_value

    rotary_dim = text_config.pop("rotary_dim", None)
    rope_theta = text_config.pop("rope_theta", 1_000_000.0)
    if rotary_dim is not None:
        head_dim = text_config.get("head_dim")
        if head_dim is None:
            head_dim = text_config["hidden_size"] // text_config["num_attention_heads"]
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError(f"`rotary_dim` must be a positive even value no larger than `head_dim` ({head_dim}).")
        native_rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": rotary_dim / head_dim,
        }
        if "rope_parameters" in text_config and text_config["rope_parameters"] != native_rope_parameters:
            raise ValueError("Conflicting legacy RoPE fields and `rope_parameters` values.")
        text_config["rope_parameters"] = native_rope_parameters
    else:
        text_config.setdefault("rope_parameters", {"rope_type": "default", "rope_theta": rope_theta})

    text_config.setdefault("bos_token_id", None)
    text_config.setdefault("eos_token_id", None)
    text_config["model_type"] = "minimax_vl_01_text"
    if text_config.get("architectures") == ["MiniMaxText01ForCausalLM"]:
        text_config["architectures"] = ["MiniMaxVL01TextModel"]
    return text_config


def convert_config(original_config: dict) -> MiniMaxVL01Config:
    if original_config.get("model_type") != "minimax_vl_01":
        raise ValueError(
            "Expected a MiniMax-VL-01 configuration with `model_type='minimax_vl_01'`, "
            f"got {original_config.get('model_type')!r}."
        )

    config_kwargs = copy.deepcopy(original_config)
    config_kwargs.pop("auto_map", None)
    config_kwargs.pop("image_seq_length", None)
    config_kwargs["model_type"] = "minimax_vl_01"
    config_kwargs["architectures"] = ["MiniMaxVL01ForConditionalGeneration"]

    if isinstance(config_kwargs.get("text_config"), dict):
        text_config = config_kwargs["text_config"]
        text_model_type = text_config.get("model_type", "minimax_vl_01_text")
        if text_model_type not in {"minimax", "minimax_text_01", "MiniMaxText01", "minimax_vl_01_text"}:
            raise ValueError(f"Expected a MiniMax text configuration, got `model_type={text_model_type!r}`.")
        text_config = _convert_text_config(text_config)
        text_config.pop("auto_map", None)
        text_config.pop("router_aux_loss_coef", None)
        text_config["model_type"] = "minimax_vl_01_text"
        text_config["architectures"] = ["MiniMaxVL01TextModel"]
        config_kwargs["text_config"] = text_config

    if isinstance(config_kwargs.get("vision_config"), dict):
        config_kwargs["vision_config"].pop("auto_map", None)
        config_kwargs["vision_config"].pop("vocab_size", None)

    return MiniMaxVL01Config(**config_kwargs)


def convert_minimax_vl_01_model(input_dir, output_dir):
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as f:
        original_config = json.load(f)
    config = convert_config(original_config)
    config.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the MiniMax-VL-01 configuration to native Transformers format."
    )
    parser.add_argument("input_dir", type=str, help="Local directory containing the original config.json.")
    parser.add_argument("output_dir", type=str, help="Directory in which to write only the converted config.json.")
    args = parser.parse_args()
    convert_minimax_vl_01_model(args.input_dir, args.output_dir)
