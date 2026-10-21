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
    # Configuration normalization is shared with AutoConfig and retains explicit Pro sliding-attention settings.
    return MiMoV2FlashConfig(
        **{
            key: value
            for key, value in original_config.items()
            if key not in ("model_type", "auto_map", "architectures")
        }
    )


def convert_mimo_v2_flash_model(input_dir, output_dir):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as f:
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
