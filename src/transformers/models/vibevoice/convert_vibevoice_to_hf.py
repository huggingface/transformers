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
import os
import re
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    Qwen2TokenizerFast,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceConfig,
    VibeVoiceForConditionalGeneration,
    VibeVoiceProcessor,
    VibeVoiceSemanticTokenizerConfig,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    # Tokenizer encoder: downsample_layers.0 -> stem, downsample_layers.N -> conv_layers.N-1
    r"(?:semantic|acoustic)_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":    r"TOKENIZER_TYPE_tokenizer.encoder.stem.conv.conv.",
    r"(?:semantic|acoustic)_tokenizer\.encoder\.stages\.0\.":                        r"TOKENIZER_TYPE_tokenizer.encoder.stem.stage.",
    r"(?:semantic|acoustic)_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.": r"TOKENIZER_TYPE_tokenizer.encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"(?:semantic|acoustic)_tokenizer\.encoder\.stages\.(\d+)\.":                    r"TOKENIZER_TYPE_tokenizer.encoder.conv_layers.PLACEHOLDER.stage.",
    r"(?:semantic|acoustic)_tokenizer\.encoder\.head\.conv\.":                       r"TOKENIZER_TYPE_tokenizer.encoder.head.",

    # Acoustic tokenizer decoder: upsample_layers.0 -> stem, upsample_layers.N -> conv_layers.N-1
    r"acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.":           r"acoustic_tokenizer.decoder.stem.conv.conv.",
    r"acoustic_tokenizer\.decoder\.stages\.0\.":                                   r"acoustic_tokenizer.decoder.stem.stage.",
    r"acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.":  r"acoustic_tokenizer.decoder.conv_layers.PLACEHOLDER.convtr.convtr.",
    r"acoustic_tokenizer\.decoder\.stages\.(\d+)\.":                               r"acoustic_tokenizer.decoder.conv_layers.PLACEHOLDER.stage.",
    r"acoustic_tokenizer\.decoder\.head\.conv\.":                                  r"acoustic_tokenizer.decoder.head.",

    # Diffusion head renaming
    r"prediction_head\.t_embedder\.mlp\.0\.":                                      r"diffusion_head.timestep_embedder.layer_1.",
    r"prediction_head\.t_embedder\.mlp\.2\.":                                      r"diffusion_head.timestep_embedder.layer_2.",
    r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\.":                     r"diffusion_head.layers.\1.linear.",
    r"prediction_head\.final_layer\.adaLN_modulation\.1\.":                       r"diffusion_head.final_layer.linear_1.",
    r"prediction_head\.final_layer\.linear\.":                                     r"diffusion_head.final_layer.linear_2.",
    r"prediction_head\.":                                                          r"diffusion_head.",

    # Latent factors
    r"^model\.speech_scaling_factor":                                              r"latent_scaling_factor",
    r"^model\.speech_bias_factor":                                                 r"latent_bias_factor",

    # Clean up nested conv layers (must be after above mappings)
    r"mixer\.conv\.conv\.conv\.":                                                  r"mixer.conv.",
    r"\.conv\.conv\.conv\.":                                                       r".conv.conv.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key

    for pattern, replacement in STATE_DICT_MAPPING.items():
        match = re.search(pattern, new_key)
        if match:
            # Handle tokenizer type replacement for non-capturing group patterns
            if "TOKENIZER_TYPE" in replacement:
                tokenizer_type = "semantic" if "semantic_tokenizer" in new_key else "acoustic"
                replacement = replacement.replace("TOKENIZER_TYPE", tokenizer_type)

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


def process_tokenizer_config(config_dict: dict[str, Any], keys_to_remove: list[str]) -> dict[str, Any]:
    if "encoder_depths" in config_dict and isinstance(config_dict["encoder_depths"], str):
        config_dict["encoder_depths"] = list(map(int, config_dict["encoder_depths"].split("-")))

    # Rename keys
    if "layernorm_eps" in config_dict:
        config_dict["rms_norm_eps"] = config_dict.pop("layernorm_eps")
    if "encoder_ratios" in config_dict:
        config_dict["downsampling_ratios"] = list(reversed(config_dict.pop("encoder_ratios")))
    if "encoder_n_filters" in config_dict:
        config_dict["num_filters"] = config_dict.pop("encoder_n_filters")
    if "encoder_depths" in config_dict:
        config_dict["depths"] = config_dict.pop("encoder_depths")
    if "vae_dim" in config_dict:
        config_dict["hidden_size"] = config_dict.pop("vae_dim")

    # Remove unwanted keys
    for key in keys_to_remove:
        config_dict.pop(key, None)

    return config_dict


def convert_checkpoint(
    checkpoint, output_dir, config_path, push_to_hub, bfloat16, processor_config=None, max_shard_size="2GB"
):
    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) Load state dict from safetensors checkpoint
    logger.info(f"Loading checkpoint from {checkpoint}")
    original_state_dict = load_file(checkpoint)
    logger.info(f"Number of parameters in original checkpoint: {len(original_state_dict)}")

    # 2) Prepare feature extractor (same for all models)
    logger.info("Creating feature extractor")
    audio_config = {}
    if processor_config is not None:
        with open(processor_config, "r") as f:
            processor_config = json.load(f)
        audio_config = processor_config.get("audio_processor", {})
        language_model_pretrained_name = processor_config.get("language_model_pretrained_name", None)
    if "sampling_rate" not in audio_config:
        audio_config["sampling_rate"] = 24000
    if "normalize_audio" not in audio_config:
        audio_config["normalize_audio"] = True
    if "target_dB_FS" not in audio_config:
        audio_config["target_dB_FS"] = -25
    if "eps" not in audio_config:
        audio_config["eps"] = 1e-6
    if language_model_pretrained_name is None:
        if "1.5B" in checkpoint:
            language_model_pretrained_name = "Qwen/Qwen2.5-1.5B"
        else:
            language_model_pretrained_name = "Qwen/Qwen2.5-7B"
    feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor(**audio_config)

    # 3) Prepare model configuration
    logger.info(f"Loading model config from {config_path}")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # fmt: off
    config_keys_to_remove = [
        "decoder_depths", "decoder_n_filters", "decoder_ratios", "std_dist_type", "fix_std", "pad_mode", "conv_bias",
        "causal", "mixer_layer", "layernorm", "disable_last_norm", "conv_norm", "corpus_normalize",
        "layernorm_elementwise_affine",
    ]
    # fmt: on

    # Process tokenizer configs
    semantic_config_dict = process_tokenizer_config(
        model_config.get("semantic_tokenizer_config", {}).copy(), config_keys_to_remove
    )
    acoustic_config_dict = process_tokenizer_config(
        model_config.get("acoustic_tokenizer_config", {}).copy(), config_keys_to_remove
    )
    # Acoustic tokenizer has additional vae_std parameter
    if "fix_std" in acoustic_config_dict:
        acoustic_config_dict["vae_std"] = acoustic_config_dict.pop("fix_std") / 0.8

    # Process diffusion head config
    diffusion_config = model_config["diffusion_head_config"]
    model_config["intermediate_size"] = int(diffusion_config.pop("head_ffn_ratio") * diffusion_config["hidden_size"])
    diffusion_config["num_head_layers"] = diffusion_config.pop("head_layers")
    if diffusion_config["ddpm_beta_schedule"] == "cosine":
        diffusion_config["ddpm_beta_schedule"] = "squaredcos_cap_v2"
    for key in ["speech_vae_dim", "diffusion_type", "ddpm_batch_mul", "latent_size", "hidden_size"]:
        diffusion_config.pop(key, None)
    # Flatten diffusion config into main config
    model_config.update(diffusion_config)
    del model_config["diffusion_head_config"]

    # Process language model config
    model_config["text_config"] = model_config.pop("decoder_config")
    model_config["text_config"]["dtype"] = model_config["text_config"].pop("torch_dtype")

    # Clean up main model config
    model_config["dtype"] = model_config.pop("torch_dtype", None)
    for key in ["acoustic_vae_dim", "semantic_vae_dim", "tie_word_embeddings"]:
        model_config.pop(key, None)

    # 4) Update state dict to match HF model structure
    logger.info("Converting state dict")
    updated_state_dict = convert_state_dict(original_state_dict)

    # 5) Create semantic tokenizer config
    logger.info("Creating semantic tokenizer config")
    semantic_config = VibeVoiceSemanticTokenizerConfig(**semantic_config_dict)

    # 6) Create acoustic tokenizer config
    logger.info("Creating acoustic tokenizer config")
    acoustic_config = VibeVoiceAcousticTokenizerConfig(**acoustic_config_dict)

    # 7) Create VibeVoice processor
    logger.info("Creating VibeVoice processor")

    # Define a chat template adapted for VibeVoice's speech use case
    chat_template = """{%- set system_prompt = system_prompt | default(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n") -%}
{{ system_prompt -}}
{%- set audio_bos_token = audio_bos_token | default("<|vision_start|>") %}
{%- set audio_eos_token = audio_eos_token | default("<|vision_end|>") %}
{%- set audio_diffusion_token = audio_diffusion_token | default("<|vision_pad|>") %}
{%- set ns = namespace(speakers_with_audio="") %}
{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set content = message['content'] %}
    {%- set has_audio = content | selectattr('type', 'equalto', 'audio') | list | length > 0 %}
    {%- if has_audio and role not in ns.speakers_with_audio %}
        {%- set ns.speakers_with_audio = ns.speakers_with_audio + role + "," %}
    {%- endif %}
{%- endfor %}

{%- if ns.speakers_with_audio %}
{{ " Voice input:\n" }}
{%- for speaker in ns.speakers_with_audio.rstrip(',').split(',') %}
{%- if speaker %}
 Speaker {{ speaker }}:{{ audio_bos_token }}{{ audio_diffusion_token }}{{ audio_eos_token }}{{ "\n" }}
{%- endif %}
{%- endfor %}
{%- endif %}
 Text input:{{ "\n" }}

{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list %}
    {%- for item in text_items %}
 Speaker {{ role }}: {{ item['text'] }}{{ "\n" }}
    {%- endfor %}
{%- endfor %}
 Speech output:{{ "\n" }}{{ audio_bos_token }}"""

    # Explicitly use Qwen2TokenizerFast to ensure proper class name in config
    tokenizer = Qwen2TokenizerFast.from_pretrained(language_model_pretrained_name)
    processor = VibeVoiceProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.save_pretrained(output_dir)
    logger.info(f"Saved processor to {output_dir}")

    # Ensure tokenizer_config.json has the correct tokenizer_class
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["tokenizer_class"] = "Qwen2TokenizerFast"

        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

    if push_to_hub is not None:
        logger.info(f"Pushing processor to Hub: {push_to_hub}")
        processor.push_to_hub(push_to_hub)

    # 8) Create and save full VibeVoice model
    logger.info("Creating full model")
    model_config["acoustic_tokenizer_config"] = acoustic_config.to_dict()
    model_config["semantic_tokenizer_config"] = semantic_config.to_dict()
    vibevoice_config = VibeVoiceConfig(**model_config)
    vibevoice_model = VibeVoiceForConditionalGeneration(vibevoice_config).to(dtype)
    logger.info(f"Number of parameters in model: {len(vibevoice_model.state_dict())}")
    # -- print dtypes of key components for verification
    logger.info(f"Acoustic connector dtype: {next(vibevoice_model.model.acoustic_connector.parameters()).dtype}")
    logger.info(f"Semantic connector dtype: {next(vibevoice_model.model.semantic_connector.parameters()).dtype}")
    logger.info(f"Language model dtype: {next(vibevoice_model.model.language_model.parameters()).dtype}")
    logger.info(f"Acoustic tokenizer dtype: {next(vibevoice_model.model.acoustic_tokenizer.parameters()).dtype}")
    logger.info(f"Semantic tokenizer dtype: {next(vibevoice_model.model.semantic_tokenizer.parameters()).dtype}")
    logger.info(f"Diffusion head dtype: {next(vibevoice_model.model.diffusion_head.parameters()).dtype}")

    # -- load into HF model
    if model_config["text_config"].get("tie_word_embeddings", False):
        # 1.5B ties weights: https://huggingface.co/microsoft/VibeVoice-1.5B/blob/main/config.json#L61
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modeling_vibevoice_inference.py#L123
        updated_state_dict["lm_head.weight"] = updated_state_dict["model.language_model.embed_tokens.weight"]
    # 7B does not tie weights: https://huggingface.co/vibevoice/VibeVoice-7B/blob/main/config.json#L113

    logger.info("Loading weights into model")
    load_result = vibevoice_model.load_state_dict(updated_state_dict, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"{len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        raise ValueError(f"{len(load_result.missing_keys)} missing keys: {load_result.missing_keys}")
    logger.info("Full model checkpoint loaded successfully")

    # Set default generation config
    vibevoice_model.generation_config._from_model_config = False
    if "7B" in checkpoint:
        vibevoice_model.generation_config.max_new_tokens = 20250
        vibevoice_model.generation_config.max_length = 20250
    else:
        vibevoice_model.generation_config.max_new_tokens = 40500
        vibevoice_model.generation_config.max_length = 40500
    vibevoice_model.generation_config.do_sample = False
    vibevoice_model.generation_config.guidance_scale = 1.3
    vibevoice_model.generation_config.noise_scheduler_class = "DPMSolverMultistepScheduler"
    vibevoice_model.generation_config.noise_scheduler_config = {
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "v_prediction",
    }

    logger.info(f"Saving model to {output_dir}")
    vibevoice_model.save_pretrained(output_dir, max_shard_size=max_shard_size)

    if push_to_hub is not None:
        logger.info(f"Pushing model to Hub: {push_to_hub}")
        vibevoice_model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)
        processor.push_to_hub(push_to_hub)

    # 9) Check model
    logger.info("Verifying conversion by reloading model")
    gc.collect()
    VibeVoiceProcessor.from_pretrained(output_dir)
    VibeVoiceForConditionalGeneration.from_pretrained(output_dir, dtype=dtype, device_map="auto")
    logger.info("Model reloaded successfully!")
    logger.info("Conversion complete!")


"""
Conversion script to convert original VibeVoice model into a checkpoint for `VibeVoiceForConditionalGeneration`


# 1.5B Model: https://huggingface.co/microsoft/VibeVoice-1.5B

```bash
# -- download checkpoint and configs
# -- download script here: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-download_vibevoice_checkpoint-py
python src/transformers/models/vibevoice/download_vibevoice_checkpoint.py
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json -P /raid/eric/vibevoice
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/preprocessor_config.json -P /raid/eric/vibevoice

# -- run conversion
python src/transformers/models/vibevoice/convert_vibevoice_to_hf.py \
    --checkpoint /raid/eric/vibevoice/VibeVoice-1.5B-combined.safetensors \
    --output_dir /raid/eric/vibevoice/hf_vibevoice \
    --config_path /raid/eric/vibevoice/config.json \
    --processor_config /raid/eric/vibevoice/preprocessor_config.json \
    --push_to_hub bezzam/VibeVoice-1.5B
```

# 7B Model: https://huggingface.co/aoi-ot/VibeVoice-Large

```bash
# -- download checkpoint and configs
# -- download script here: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-download_vibevoice_7b_checkpoint-py
python src/transformers/models/vibevoice/download_vibevoice_7b_checkpoint.py
wget https://huggingface.co/aoi-ot/VibeVoice-Large/resolve/main/config.json -P /raid/eric/vibevoice_7b
wget https://huggingface.co/aoi-ot/VibeVoice-Large/resolve/main/preprocessor_config.json -P /raid/eric/vibevoice_7b

# -- run conversion
python src/transformers/models/vibevoice/convert_vibevoice_to_hf.py \
    --checkpoint /raid/eric/vibevoice_7b/VibeVoice-7B-combined.safetensors \
    --output_dir /raid/eric/vibevoice/hf_vibevoice_7b \
    --config_path /raid/eric/vibevoice_7b/config.json \
    --processor_config /raid/eric/vibevoice_7b/preprocessor_config.json \
    --push_to_hub bezzam/VibeVoice-7B
```

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, default=None, type=str, help="Original VibeVoice model checkpoint."
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to config.json of model to convert")
    parser.add_argument(
        "--processor_config", default=None, type=str, help="Path to preprocessor_config.json of model to convert"
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--float32", action="store_true", help="Whether to use float32 precision. Default is bfloat16."
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Maximum shard size for safetensors files in GB.",
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        args.config_path,
        args.push_to_hub,
        bfloat16=not args.float32,
        processor_config=args.processor_config,
        max_shard_size=args.max_shard_size,
    )
