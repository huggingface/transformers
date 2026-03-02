# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    Qwen2Config,
    Qwen2TokenizerFast,
    VibeVoiceAcousticTokenizerEncoderConfig,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceAsrConfig,
    VibeVoiceAsrForConditionalGeneration,
    VibeVoiceAsrProcessor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    # Language model
    r"^model\.language_model\.embed_tokens\.weight":                                r"language_model.model.embed_tokens.weight",
    r"^model\.language_model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.":         r"language_model.model.layers.\1.self_attn.\2_proj.",
    r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.":          r"language_model.model.layers.\1.mlp.\2_proj.",
    r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.":                   r"language_model.model.layers.\1.input_layernorm.",
    r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.":          r"language_model.model.layers.\1.post_attention_layernorm.",
    r"^model\.language_model\.norm\.":                                              r"language_model.model.norm.",
    r"^lm_head\.":                                                                  r"language_model.lm_head.",

    # Acoustic and semantic tokenizer (encoder-only)
    r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":       r"acoustic_tokenizer_encoder.stem.conv.conv.",
    r"^model\.acoustic_tokenizer\.encoder\.stages\.0\.":                           r"acoustic_tokenizer_encoder.stem.stage.",
    r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.":   r"acoustic_tokenizer_encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.":                       r"acoustic_tokenizer_encoder.conv_layers.PLACEHOLDER.stage.",
    r"^model\.acoustic_tokenizer\.encoder\.head\.conv\.":                          r"acoustic_tokenizer_encoder.head.",
    r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":       r"semantic_tokenizer_encoder.stem.conv.conv.",
    r"^model\.semantic_tokenizer\.encoder\.stages\.0\.":                           r"semantic_tokenizer_encoder.stem.stage.",
    r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.":   r"semantic_tokenizer_encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"^model\.semantic_tokenizer\.encoder\.stages\.(\d+)\.":                       r"semantic_tokenizer_encoder.conv_layers.PLACEHOLDER.stage.",
    r"^model\.semantic_tokenizer\.encoder\.head\.conv\.":                          r"semantic_tokenizer_encoder.head.",
    # -- important! should be after above mapping
    r"mixer\.conv\.conv\.conv\.":                                                   r"mixer.conv.",
    r"\.conv\.conv\.conv\.":                                                        r".conv.conv.",

    # Merged acoustic and semantic projector
    r"^model\.acoustic_connector\.fc1\.":                                           r"multi_modal_projector.acoustic_linear_1.",
    r"^model\.acoustic_connector\.fc2\.":                                           r"multi_modal_projector.acoustic_linear_2.",
    r"^model\.acoustic_connector\.norm\.":                                          r"multi_modal_projector.acoustic_norm.",
    r"^model\.semantic_connector\.fc1\.":                                           r"multi_modal_projector.semantic_linear_1.",
    r"^model\.semantic_connector\.fc2\.":                                           r"multi_modal_projector.semantic_linear_2.",
    r"^model\.semantic_connector\.norm\.":                                          r"multi_modal_projector.semantic_norm.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key

    for pattern, replacement in STATE_DICT_MAPPING.items():
        match = re.search(pattern, new_key)
        if match:
            # Handle index shifts for conv_layers (downsample_layers/upsample_layers indexed from 1)
            if "PLACEHOLDER" in replacement and match.groups():
                layer_idx = int(match.group(1))
                # Shift down by 1 since layer 0 becomes stem
                new_idx = layer_idx - 1
                replacement = replacement.replace("PLACEHOLDER", str(new_idx))

            new_key = re.sub(pattern, replacement, new_key)

    return new_key


def convert_state_dict(original_state_dict: dict[str, Any]) -> dict[str, Any]:
    new_state_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")

    return new_state_dict


def load_original_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_dir():
        raise ValueError(
            f"checkpoint_path must be a directory containing sharded safetensors files, got: {checkpoint_path}"
        )

    # Load sharded safetensors checkpoint
    index_path = checkpoint_path / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Could not find 'model.safetensors.index.json' in {checkpoint_path}. "
            "Expected sharded safetensors checkpoint format."
        )

    logger.info(f"Loading sharded checkpoint from {checkpoint_path}")
    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    # Get unique shard files
    shard_files = sorted(set(index["weight_map"].values()))

    for shard_file in shard_files:
        shard_path = checkpoint_path / shard_file
        logger.info(f"Loading shard: {shard_file}")
        shard_dict = load_file(str(shard_path))
        state_dict.update(shard_dict)

    return state_dict


def create_config_from_checkpoint(checkpoint_path: str | Path) -> VibeVoiceAsrConfig:
    checkpoint_path = Path(checkpoint_path)
    config_path = (
        checkpoint_path / "config.json" if checkpoint_path.is_dir() else checkpoint_path.parent / "config.json"
    )

    if config_path.exists():
        with open(config_path, "r") as f:
            original_config = json.load(f)

        config_keys_to_remove = [
            "decoder_depths",
            "decoder_n_filters",
            "decoder_ratios",
            "std_dist_type",
            "fix_std",
            "pad_mode",
            "conv_bias",
            "causal",
            "mixer_layer",
            "layernorm",
            "disable_last_norm",
            "conv_norm",
            "corpus_normalize",
            "layernorm_elementwise_affine",
        ]

        # Prepare acoustic tokenizer config
        acoustic_config_dict = original_config.get("acoustic_tokenizer_config", {}).copy()
        if "encoder_depths" in acoustic_config_dict and isinstance(acoustic_config_dict["encoder_depths"], str):
            acoustic_config_dict["encoder_depths"] = list(map(int, acoustic_config_dict["encoder_depths"].split("-")))
        if "layernorm_eps" in acoustic_config_dict:
            acoustic_config_dict["rms_norm_eps"] = acoustic_config_dict.pop("layernorm_eps")
        if "encoder_ratios" in acoustic_config_dict:
            acoustic_config_dict["downsampling_ratios"] = list(reversed(acoustic_config_dict.pop("encoder_ratios")))
        if "encoder_n_filters" in acoustic_config_dict:
            acoustic_config_dict["num_filters"] = acoustic_config_dict.pop("encoder_n_filters")
        if "encoder_depths" in acoustic_config_dict:
            acoustic_config_dict["depths"] = acoustic_config_dict.pop("encoder_depths")
        if "vae_dim" in acoustic_config_dict:
            acoustic_config_dict["hidden_size"] = acoustic_config_dict.pop("vae_dim")
        if "fix_std" in acoustic_config_dict:
            acoustic_config_dict["vae_std"] = acoustic_config_dict.pop("fix_std") / 0.8
        for key in config_keys_to_remove:
            acoustic_config_dict.pop(key, None)
        acoustic_tokenizer_encoder_config = VibeVoiceAcousticTokenizerEncoderConfig(**acoustic_config_dict)

        # Prepare semantic tokenizer config
        semantic_config_dict = original_config.get("semantic_tokenizer_config", {}).copy()
        if "encoder_depths" in semantic_config_dict and isinstance(semantic_config_dict["encoder_depths"], str):
            semantic_config_dict["encoder_depths"] = list(map(int, semantic_config_dict["encoder_depths"].split("-")))
        if "layernorm_eps" in semantic_config_dict:
            semantic_config_dict["rms_norm_eps"] = semantic_config_dict.pop("layernorm_eps")
        if "encoder_ratios" in semantic_config_dict:
            semantic_config_dict["downsampling_ratios"] = list(reversed(semantic_config_dict.pop("encoder_ratios")))
        if "encoder_n_filters" in semantic_config_dict:
            semantic_config_dict["num_filters"] = semantic_config_dict.pop("encoder_n_filters")
        if "encoder_depths" in semantic_config_dict:
            semantic_config_dict["depths"] = semantic_config_dict.pop("encoder_depths")
        if "vae_dim" in semantic_config_dict:
            semantic_config_dict["hidden_size"] = semantic_config_dict.pop("vae_dim")
        for key in config_keys_to_remove:
            semantic_config_dict.pop(key, None)
        semantic_tokenizer_encoder_config = VibeVoiceAcousticTokenizerEncoderConfig(**semantic_config_dict)

        # Create main config
        config = VibeVoiceAsrConfig(
            acoustic_tokenizer_encoder_config=acoustic_tokenizer_encoder_config,
            semantic_tokenizer_encoder_config=semantic_tokenizer_encoder_config,
            text_config=Qwen2Config(**original_config.get("decoder_config", {})),
        )
    else:
        logger.warning("No config.json found, using default configuration")
        config = VibeVoiceAsrConfig()

    return config


def create_processor(checkpoint_path: str | Path, output_dir: str | Path) -> VibeVoiceAsrProcessor:
    checkpoint_path = Path(checkpoint_path)

    # Original building of token sequence: https://github.com/microsoft/VibeVoice/blob/b2aee8015c3c2d97c388346ebcfffdaf2f427f7d/vibevoice/processor/vibevoice_asr_processor.py#L347
    chat_template = """{%- set system_prompt = system_prompt | default("You are a helpful assistant that transcribes audio input into text output in JSON format.") -%}
<|im_start|>system
{{ system_prompt }}<|im_end|>
{%- set audio_token = audio_token | default("<|box_start|>") -%}
{%- set audio_start_token = "<|object_ref_start|>" -%}
{%- set audio_end_token = "<|object_ref_end|>" -%}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
{{ '\n' }}<|im_start|>user{{ '\n' }}{%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list -%}
        {%- set context_text = text_items[0]['text'] if text_items else none -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'audio' -%}
{{ audio_start_token }}{{ audio_token }}{{ audio_end_token }}{{ "\n" }}{%- if context_text -%}
This is a <|AUDIO_DURATION|> seconds audio, with extra info: {{ context_text }}

Please transcribe it with these keys: Start time, End time, Speaker ID, Content{%- else -%}
This is a <|AUDIO_DURATION|> seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content{%- endif -%}
            {%- endif -%}
        {%- endfor -%}
<|im_end|>{{ '\n' }}
    {%- endif -%}
{%- endfor -%}"""

    processor = VibeVoiceAsrProcessor(
        feature_extractor=VibeVoiceAcousticTokenizerFeatureExtractor(),
        # Original: https://github.com/microsoft/VibeVoice/blob/b2aee8015c3c2d97c388346ebcfffdaf2f427f7d/demo/vibevoice_asr_inference_from_file.py#L49
        tokenizer=Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-7B"),
        chat_template=chat_template,
    )
    processor.save_pretrained(str(output_dir))
    logger.info(f"Saved processor to {output_dir}")
    return processor


def convert_checkpoint(checkpoint_path, output_dir, push_to_hub, bfloat16, max_shard_size):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating processor")
    processor = create_processor(checkpoint_path, output_path)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    original_state_dict = load_original_checkpoint(checkpoint_path)

    logger.info("Number of parameters in original state dict: " + str(len(original_state_dict)))
    num_acoustic_decoder_params = sum(
        1 for k in original_state_dict.keys() if k.startswith("model.acoustic_tokenizer.decoder.")
    )

    # remove acoustic tokenizer decoder parameters
    logger.info(f"Number of (unused) acoustic tokenizer decoder parameters: {num_acoustic_decoder_params}")
    original_state_dict = {
        k: v for k, v in original_state_dict.items() if not k.startswith("model.acoustic_tokenizer.decoder.")
    }

    logger.info("Converting state dict")
    converted_state_dict = convert_state_dict(original_state_dict)

    logger.info("Creating config")
    config = create_config_from_checkpoint(checkpoint_path)
    config.save_pretrained(str(output_path))

    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    logger.info(f"Creating model with dtype {dtype}")
    model = VibeVoiceAsrForConditionalGeneration(config).to(dtype)
    logger.info("Number of parameters in model state dict: " + str(len(model.state_dict())))

    logger.info("Loading weights into model")
    load_result = model.load_state_dict(converted_state_dict, strict=False)
    if load_result.missing_keys:
        raise ValueError(f"{len(load_result.missing_keys)} missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        raise ValueError(f"{len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys}")

    model.generation_config.pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.bos_token_id = processor.tokenizer.bos_token_id
    model.generation_config.do_sample = False
    model.generation_config.max_new_tokens = 32768
    model.generation_config.max_length = 32768

    logger.info(f"Saving model to {output_path}")
    model.save_pretrained(str(output_path), max_shard_size=max_shard_size)

    if push_to_hub:
        logger.info(f"Pushing to Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)
        processor.push_to_hub(push_to_hub)

    logger.info("Verifying conversion by reloading model")
    gc.collect()
    VibeVoiceAsrProcessor.from_pretrained(str(output_path))
    VibeVoiceAsrForConditionalGeneration.from_pretrained(str(output_path))
    logger.info("Model reloaded successfully!")
    logger.info("Conversion complete!")


"""
Conversion script to convert the original VibeVoice ASR model checkpoint to Hugging Face format.

Usage:

1) Download the original VibeVoice ASR model checkpoint:
```bash
huggingface-cli download microsoft/VibeVoice-ASR --local-dir /path/to/vibevoice-asr
```

2) Run conversion script (with optional `push_to_hub` argument):
```bash
python src/transformers/models/vibevoice_asr/convert_vibevoice_asr_to_hf.py \
    --checkpoint_path /path/to/vibevoice-asr \
    --output_dir ./vibevoice_asr_hf \
    --push_to_hub your-username/VibeVoice-ASR-7B
```
"""


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice ASR checkpoint to Hugging Face format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original VibeVoice ASR checkpoint (directory or file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted checkpoint",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (e.g., 'username/vibevoice-asr-hf'). If not provided, only saves locally.",
    )
    parser.add_argument(
        "--float32", action="store_true", help="Whether to use float32 precision. Default is bfloat16."
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2.5GB",
        help="Maximum shard size for safetensors files in GB.",
    )

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        bfloat16=not args.float32,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
