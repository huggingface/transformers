# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert LLaVA-OneVision-1.5 checkpoints from the original repository (trust_remote_code format) to the
Hugging Face Transformers format.

The original checkpoint stores the following top-level parameter groups:
    - `visual.*`               -> RICE vision tower
    - `model.*` (except above) -> Qwen3-based text backbone
    - `lm_head.weight`         -> untied language modeling head (NOT tied to `model.embed_tokens.weight`)

Usage:

```bash
python convert_llava_onevision1_5_weights_to_hf.py \
    --input_dir /path/to/original/checkpoint \
    --output_dir /path/to/output/checkpoint
```
"""

import argparse
import glob
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import (
    LlavaOnevision1_5Config,
    LlavaOnevision1_5ForConditionalGeneration,
    LlavaOnevision1_5TextConfig,
    LlavaOnevision1_5VisionConfig,
)


def build_config(orig_cfg: dict) -> LlavaOnevision1_5Config:
    text_cfg = orig_cfg["text_config"]
    vision_cfg = orig_cfg["vision_config"]

    new_text_config = LlavaOnevision1_5TextConfig(
        vocab_size=text_cfg["vocab_size"],
        hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg["intermediate_size"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"],
        hidden_act=text_cfg["hidden_act"],
        max_position_embeddings=text_cfg["max_position_embeddings"],
        initializer_range=text_cfg["initializer_range"],
        rms_norm_eps=text_cfg["rms_norm_eps"],
        use_cache=text_cfg["use_cache"],
        rope_parameters={"rope_type": "default", "rope_theta": text_cfg["rope_theta"]},
        attention_bias=text_cfg["attention_bias"],
        attention_dropout=text_cfg["attention_dropout"],
        use_sliding_window=text_cfg["use_sliding_window"],
        sliding_window=text_cfg["sliding_window"],
        max_window_layers=text_cfg["max_window_layers"],
    )
    new_vision_config = LlavaOnevision1_5VisionConfig(
        depth=vision_cfg["depth"],
        hidden_size=vision_cfg["hidden_size"],
        hidden_act=vision_cfg["hidden_act"],
        intermediate_size=vision_cfg["intermediate_size"],
        num_heads=vision_cfg["num_heads"],
        in_channels=vision_cfg["in_channels"],
        patch_size=vision_cfg["patch_size"],
        spatial_merge_size=vision_cfg["spatial_merge_size"],
        temporal_patch_size=vision_cfg["temporal_patch_size"],
        out_hidden_size=vision_cfg["text_hidden_size"],
        layer_norm_eps=vision_cfg["layer_norm_eps"],
        initializer_range=vision_cfg["initializer_range"],
    )
    return LlavaOnevision1_5Config(
        text_config=new_text_config,
        vision_config=new_vision_config,
        image_token_id=orig_cfg["image_token_id"],
        video_token_id=orig_cfg["video_token_id"],
        # The original checkpoint stores an independent (untied) `lm_head.weight`.
        tie_word_embeddings=False,
    )


def remap_state_dict_key(key: str) -> str:
    if re.match(r"^visual", key):
        return "model.visual." + key[len("visual.") :]
    if re.match(r"^model(?!\.(language_model|visual))", key):
        return "model.language_model." + key[len("model.") :]
    return key


def load_original_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict = {}
    for file in sorted(glob.glob(os.path.join(input_dir, "*.safetensors"))):
        state_dict.update(load_file(file))
    return {remap_state_dict_key(k): v for k, v in state_dict.items()}


def convert_llava_onevision1_5_to_hf(input_dir: str, output_dir: str) -> None:
    with open(os.path.join(input_dir, "config.json")) as f:
        orig_cfg = json.load(f)

    config = build_config(orig_cfg)
    model = LlavaOnevision1_5ForConditionalGeneration(config)

    state_dict = load_original_state_dict(input_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if "rotary_emb.inv_freq" not in key]
    if missing:
        raise ValueError(f"Missing keys when loading converted checkpoint: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected keys when loading converted checkpoint: {unexpected}")

    model = model.to(torch.bfloat16)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Saved converted model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the original (trust_remote_code) LLaVA-OneVision-1.5 checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where the converted Transformers checkpoint will be saved.",
    )
    args = parser.parse_args()

    convert_llava_onevision1_5_to_hf(args.input_dir, args.output_dir)
