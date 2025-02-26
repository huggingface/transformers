# Copyright 2023 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import (
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalVisionConfig,
)


# fmt: off
STATE_DICT_MAPPING = {
    # CausalLM keys
    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).feed_forward_in.net.0.linear"  : r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.feed_forward_in.gate_up_proj",
    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).feed_forward_out.net.2": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.feed_forward_in.down_proj",

    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).self_attn.linear_(q|k|v)": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.self_attn.\2_proj",
    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).self_attn.linear_out": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.self_attn.o_proj"
}
# fmt: on


def map_old_key_to_new(old_key):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    # not part of the key mapping
    return old_key


def convert_state_dict(original_state_dict: dict):
    """Convert a state dict file."""
    new_dict = {}
    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_dict[new_key] = tensor
    return new_dict


def convert_config(original_config: dict):
    # Remove unused args
    original_config.pop("_name_or_path", None)
    original_config.pop("architectures", None)
    original_config.pop("auto_map", None)
    original_config.pop("vision_lora", None)
    original_config.pop("speech_lora", None)
    original_config.pop("transformers_version", None)
    original_config.pop("_attn_implementation", None)

    embd_layer = original_config.pop("embd_layer")
    audio_embd_layer = embd_layer["audio_embd_layer"]
    vision_embd_layer = embd_layer["image_embd_layer"]

    audio_config = original_config.pop("audio_processor")["config"]
    # remove
    audio_config.pop("activation_checkpointing", None)
    audio_config.pop("cnn_layer_norm", None)
    audio_config.pop("input_layer", None)
    audio_config.pop("relative_attention_bias_args", None)
    # rename
    audio_config["hidden_size"] = audio_config.pop("attention_dim")
    audio_config["num_attention_heads"] = audio_config.pop("attention_heads")
    audio_config["intermediate_size"] = audio_config.pop("linear_units")
    # add
    audio_config["audio_embd_layer"] = audio_embd_layer

    # Create transformers config objects
    audio_config = Phi4MultimodalAudioConfig(**audio_config)
    vision_config = Phi4MultimodalVisionConfig(image_embd_layer=vision_embd_layer)

    new_config = Phi4MultimodalConfig(**original_config, vision_config=vision_config, audio_config=audio_config)
    return new_config


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def convert_and_write_model(input_dir: str, output_dir: str):
    """Convert the model and save it (this implicitly save the config as well)."""
    original_config = read_json(os.path.join(input_dir, "config.json"))
    config = convert_config(original_config)

    full_state_dict = {}
    shards = [file for file in os.listdir(input_dir) if file.endswith(".safetensors")]
    for shard_file in shards:
        original_state_dict = load_file(os.path.join(input_dir, shard_file))
        new_dict = convert_state_dict(original_state_dict)
        full_state_dict.update(new_dict)

    # Load weights into model and resave them
    with torch.device("meta"):
        model = Phi4MultimodalForCausalLM(config)
    model.load_state_dict(full_state_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()

    # Convert
    convert_and_write_model(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
