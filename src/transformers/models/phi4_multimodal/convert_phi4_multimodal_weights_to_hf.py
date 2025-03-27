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
import argparse
import json
import os
import re

import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from transformers import (
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalProcessor,
    Phi4MultimodalVisionConfig,
)


# fmt: off
STATE_DICT_MAPPING = {
    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).feed_forward_(in|out).net.0.linear": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.feed_forward_\2.gate_up_proj",
    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).feed_forward_(in|out).net.2": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.feed_forward_\2.down_proj",

    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).self_attn.linear_(q|k|v)": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.self_attn.\2_proj",
    r"^model.embed_tokens_extend.audio_embed.encoder.encoders.(\d+).self_attn.linear_out": r"model.embed_tokens_extend.audio_embed.encoder.encoders.\1.self_attn.o_proj",

    r"^model.embed_tokens_extend.image_embed.img_projection.0": r"model.embed_tokens_extend.image_embed.img_projection_up",
    r"^model.embed_tokens_extend.image_embed.img_projection.2": r"model.embed_tokens_extend.image_embed.img_projection_down",

    r"^model.embed_tokens_extend.image_embed.glb_GN": r"model.embed_tokens_extend.image_embed.global_img_feature_extensor",
    r"^model.embed_tokens_extend.image_embed.sub_GN": r"model.embed_tokens_extend.image_embed.sub_img_feature_extensor",

    r"^model.embed_tokens_extend.audio_embed.audio_projection.speech.0": r"model.embed_tokens_extend.audio_embed.up_proj_for_speech",
    r"^model.embed_tokens_extend.audio_embed.audio_projection.speech.2": r"model.embed_tokens_extend.audio_embed.down_proj_for_speech",
    r"^model.embed_tokens_extend.audio_embed.audio_projection.vision.0": r"model.embed_tokens_extend.audio_embed.up_proj_for_vision_speech",
    r"^model.embed_tokens_extend.audio_embed.audio_projection.vision.2": r"model.embed_tokens_extend.audio_embed.down_proj_for_vision_speech",
}
# fmt: on


def map_old_key_to_new(old_key):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    # The state dict contains lora keys....
    if "lora" in old_key:
        return None
    # This extracts the original weight before adding the lora adapter
    if "base_layer." in old_key:
        return old_key.replace("base_layer.", "")

    # not part of the key mapping, we keep the original name
    return old_key


def convert_state_dict(original_state_dict: dict):
    """Convert a state dict file."""
    new_dict = {}
    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        if new_key is not None:
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

    # Keep only some of the subdict
    keep_audio_embd_layer = ["downsample_rate"]
    keep_vision_embd_layer = ["crop_size"]
    audio_embd_layer = {k: v for k, v in audio_embd_layer.items() if k in keep_audio_embd_layer}
    vision_embd_layer = {k: v for k, v in vision_embd_layer.items() if k in keep_vision_embd_layer}

    audio_config = original_config.pop("audio_processor")["config"]
    # remove
    audio_config.pop("activation_checkpointing", None)
    audio_config.pop("cnn_layer_norm", None)
    audio_config.pop("input_layer", None)
    audio_config.pop("batch_norm", None)
    audio_config.pop("encoder_embedding_config", None)
    audio_config.pop("ext_pw_kernel_size", None)
    audio_config.pop("bias_in_glu", None)
    audio_config.pop("causal", None)
    # rename
    audio_config["hidden_size"] = audio_config.pop("attention_dim")
    audio_config["num_attention_heads"] = audio_config.pop("attention_heads")
    audio_config["intermediate_size"] = audio_config.pop("linear_units")
    audio_config["nemo_conv_channels"] = audio_config.pop("nemo_conv_settings")["conv_channels"]
    audio_config["bias_max_distance"] = audio_config.pop("relative_attention_bias_args")["t5_bias_max_distance"]
    # add
    audio_config = {**audio_config, **audio_embd_layer}

    # Create transformers config objects
    audio_config = Phi4MultimodalAudioConfig(**audio_config)
    vision_config = Phi4MultimodalVisionConfig(**vision_embd_layer)

    # Add 2nd eos to config
    original_config["eos_token_id"] = [199999, 200020]

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
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False, assign=True)
    # The lm_head is missing because it's tied
    if missing != ["lm_head.weight"]:
        raise ValueError("Missing keys:\n{missing}")
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected keys:\n{unexpected}")

    model.tie_weights()
    model.save_pretrained(output_dir)


def convert_and_save_processor(input_dir: str, output_dir: str):
    """Convert the processor."""
    processor = Phi4MultimodalProcessor.from_pretrained(input_dir)
    del processor.image_processor.auto_map
    del processor.audio_processor.auto_map
    processor.chat_template = processor.tokenizer.chat_template
    processor.tokenizer.extra_special_tokens = {"image_token": "<|endoftext10|>", "audio_token": "<|endoftext11|>"}
    processor.save_pretrained(output_dir)


def extract_adapters_data(input_dir: str, output_dir: str):
    """Extract adapters data from the state dict and save weights and configs."""
    speech_lora = {}
    vision_lora = {}
    shards = [file for file in os.listdir(input_dir) if file.endswith(".safetensors")]
    for shard_file in shards:
        original_state_dict = load_file(os.path.join(input_dir, shard_file))
        for k, v in original_state_dict.items():
            if "lora" in k:
                if "speech" in k:
                    speech_lora[k.replace("speech.", "")] = v
                elif "vision" in k:
                    vision_lora[k.replace("vision.", "")] = v

    # Create and save the lora configs
    speech_lora_config = LoraConfig(
        r=320,
        lora_alpha=640,
        target_modules=r"model.layers.\d+.((self_attn.(qkv|o)_proj)|(mlp.(gate_up|down)_proj))",
        lora_dropout=0.01,
        task_type="CAUSAL_LM",
    )
    speech_lora_config.save_pretrained(os.path.join(output_dir, "speech-lora"))
    vision_lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=r"model.layers.\d+.((self_attn.(qkv|o)_proj)|(mlp.(gate_up|down)_proj))",
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    vision_lora_config.save_pretrained(os.path.join(output_dir, "vision-lora"))

    save_file(speech_lora, os.path.join(output_dir, "speech-lora", "adapter_model.safetensors"))
    save_file(vision_lora, os.path.join(output_dir, "vision-lora", "adapter_model.safetensors"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Location of the model folder containing the weights and configs.",
    )
    parser.add_argument(
        "output_dir",
        help="Location to write HF model.",
    )
    args = parser.parse_args()

    # Convert
    convert_and_write_model(args.input_dir, args.output_dir)
    convert_and_save_processor(args.input_dir, args.output_dir)
    extract_adapters_data(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
