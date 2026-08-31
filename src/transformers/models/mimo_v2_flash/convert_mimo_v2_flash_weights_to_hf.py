# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
Rewrite the Hub-format `config.json` of XiaomiMiMo/MiMo-V2-Flash into the native
`MiMoV2FlashConfig` layout. Safetensors and the tokenizer are reused as-is.
Per-expert MoE weights and the `attention_sink_bias` → `sinks` rename are handled by the
`"mimo_v2_flash"` entry in `src/transformers/conversion_mapping.py` at load time, so no
weight modification is needed here.

Sample usage:
    python src/transformers/models/mimo_v2_flash/convert_mimo_v2_flash_weights_to_hf.py \
        /path/to/XiaomiMiMo--MiMo-V2-Flash /path/to/MiMo-V2-Flash-hf
"""

import argparse
import json
import os

from transformers import MiMoV2FlashConfig


def convert_config(original_config: dict):
    # Hub-only keys with no equivalent in `MiMoV2FlashConfig` (hardcoded or redundant in the native port).
    keys_to_drop = {
        "attention_chunk_size",
        "sliding_window_size",
        "n_shared_experts",
        "scoring_func",
        "topk_method",
        "swa_num_attention_heads",
        "swa_num_key_value_heads",
        "swa_qk_head_dim",
        "swa_head_dim",
        "swa_v_head_dim",
        "add_swa_attention_sink_bias",
        "add_full_attention_sink_bias",
        "auto_map",
    }
    new_config_kwargs = {k: v for k, v in original_config.items() if k not in keys_to_drop}

    if "layernorm_epsilon" in new_config_kwargs:
        new_config_kwargs["rms_norm_eps"] = new_config_kwargs.pop("layernorm_epsilon")

    # Binary per-layer lists -> native `layer_types` / `mlp_layer_types`.
    pattern = new_config_kwargs.pop("hybrid_layer_pattern", None)
    if pattern is not None:
        new_config_kwargs["layer_types"] = ["sliding_attention" if p == 1 else "full_attention" for p in pattern]
    freq = new_config_kwargs.pop("moe_layer_freq", None)
    if freq is not None:
        new_config_kwargs["mlp_layer_types"] = ["sparse" if f == 1 else "dense" for f in freq]

    # Hub uses `null` to mean "no rescale" while native expects 1.0.
    if new_config_kwargs.get("routed_scaling_factor") is None:
        new_config_kwargs["routed_scaling_factor"] = 1.0

    # Legacy rope fields -> per-layer-type `rope_parameters` dict.
    rope_theta = new_config_kwargs.pop("rope_theta", 5_000_000.0)
    swa_rope_theta = new_config_kwargs.pop("swa_rope_theta", 10_000.0)
    partial_rotary_factor = new_config_kwargs.pop("partial_rotary_factor", 0.334)
    rope_scaling = new_config_kwargs.pop("rope_scaling", None)
    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": swa_rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        },
    }
    if rope_scaling is not None:
        rope_parameters["full_attention"].update(rope_scaling)
    new_config_kwargs["rope_parameters"] = rope_parameters

    new_config = MiMoV2FlashConfig(**new_config_kwargs)
    return new_config


def convert_mimo_v2_flash_model(input_dir, output_dir):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json")) as f:
        original_config = json.load(f)
    config = convert_config(original_config)
    config.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help=("Location to write the converted `config.json`."),
    )
    args = parser.parse_args()
    convert_mimo_v2_flash_model(args.input_dir, args.output_dir)
