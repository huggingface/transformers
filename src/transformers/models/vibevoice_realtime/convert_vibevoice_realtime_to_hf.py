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
    VibeVoiceRealTimeConfig,
    VibeVoiceRealTimeForConditionalGeneration,
    VibeVoiceRealTimeProcessor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    # Acoustic decoder: upsample_layers.0 -> stem, upsample_layers.N -> conv_layers.N-1
    r"acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.":          r"audio_tower.decoder.stem.conv.conv.",
    r"acoustic_tokenizer\.decoder\.stages\.0\.":                                    r"audio_tower.decoder.stem.stage.",
    r"acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.": r"audio_tower.decoder.conv_layers.PLACEHOLDER.convtr.convtr.",
    r"acoustic_tokenizer\.decoder\.stages\.(\d+)\.":                               r"audio_tower.decoder.conv_layers.PLACEHOLDER.stage.",
    r"acoustic_tokenizer\.decoder\.head\.conv\.":                                   r"audio_tower.decoder.head.",

    # Rename any remaining acoustic tokenizer keys (the module is `audio_tower` in the HF model)
    r"acoustic_tokenizer\.":                                                          r"audio_tower.",

    # Diffusion head renaming
    r"prediction_head\.t_embedder\.mlp\.0\.":                                       r"diffusion_head.timestep_proj.fc1.",
    r"prediction_head\.t_embedder\.mlp\.2\.":                                       r"diffusion_head.timestep_proj.fc2.",
    r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\.":                      r"diffusion_head.layers.\1.linear.",
    r"prediction_head\.final_layer\.adaLN_modulation\.1\.":                         r"diffusion_head.final_layer.linear_1.",
    r"prediction_head\.final_layer\.linear\.":                                       r"diffusion_head.final_layer.linear_2.",
    r"prediction_head\.":                                                             r"diffusion_head.",

    # Multimodal connector (the acoustic connector is the `multi_modal_projector` in the HF model)
    r"acoustic_connector\.fc1\.":  r"multi_modal_projector.linear_1.",
    r"acoustic_connector\.norm\.": r"multi_modal_projector.act.",
    r"acoustic_connector\.fc2\.":  r"multi_modal_projector.linear_2.",

    # Latent factors
    r"^model\.speech_scaling_factor": r"model.latent_scaling_factor",
    r"^model\.speech_bias_factor":    r"model.latent_bias_factor",

    # Clean up nested conv layers (must be after above mappings)
    r"mixer\.conv\.conv\.conv\.": r"mixer.conv.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key

    for pattern, replacement in STATE_DICT_MAPPING.items():
        match = re.search(pattern, new_key)
        if match:
            # Handle index shifts for conv_layers (upsample_layers/stages indexed from 1)
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


def convert_checkpoint(checkpoint, output_dir, config_path, push_to_hub, bfloat16, processor_config=None):
    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) Load state dict from safetensors checkpoint
    logger.info(f"Loading checkpoint from {checkpoint}")
    original_state_dict = load_file(checkpoint)
    logger.info(f"Number of parameters in original checkpoint: {len(original_state_dict)}")

    # 2) Prepare feature extractor (same for all models)
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
        language_model_pretrained_name = "Qwen/Qwen2.5-0.5B"

    # 3) Prepare model configuration
    # -- Load
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # clean up acoustic decoder config (the real-time model only decodes audio)
    audio_config = model_config.pop("acoustic_tokenizer_config")
    audio_config["model_type"] = "vibevoice_realtime_acoustic_decoder"
    audio_config["hidden_size"] = audio_config.pop("vae_dim")
    audio_config["num_filters"] = audio_config.pop("decoder_n_filters")
    audio_config["upsampling_ratios"] = audio_config.pop("decoder_ratios")
    audio_config["initializer_range"] = audio_config.pop("weight_init_value")
    if "layernorm_eps" in audio_config:
        audio_config["rms_norm_eps"] = audio_config.pop("layernorm_eps")
    # -- depths are stored in encoder order in the original config
    encoder_depths = audio_config.pop("encoder_depths")
    if isinstance(encoder_depths, str):
        encoder_depths = list(map(int, encoder_depths.split("-")))
    audio_config["depths"] = encoder_depths[::-1]
    # -- remove encoder and sampling/vae parameters (decoder only), as well as constant parameters that lead to
    #    unused code paths removed in the HF model
    # fmt: off
    for key in [
        "encoder_n_filters", "encoder_ratios", "decoder_depths", "std_dist_type", "fix_std", "conv_bias", "causal",
        "mixer_layer", "layernorm", "layernorm_elementwise_affine", "disable_last_norm", "conv_norm",
        "corpus_normalize", "pad_mode",
    ]:
        audio_config.pop(key, None)
    # fmt: on
    model_config["audio_config"] = audio_config

    # build the diffusion head config (scheduler parameters are set in the generation config below)
    diffusion_config = model_config.pop("diffusion_head_config")
    model_config["diffusion_head_config"] = {
        "hidden_size": diffusion_config["hidden_size"],
        "latent_size": audio_config["hidden_size"],
        "num_hidden_layers": diffusion_config["head_layers"],
        "intermediate_size": int(diffusion_config["head_ffn_ratio"] * diffusion_config["hidden_size"]),
        "rms_norm_eps": diffusion_config["rms_norm_eps"],
    }

    # clean up and configuration language model config -> 2 language models (one for text, one for tts)
    model_config["text_config"] = model_config.pop("decoder_config")
    model_config["text_config"]["dtype"] = model_config["text_config"].pop("torch_dtype")
    model_config["text_config"]["num_hidden_layers"] = (
        model_config["text_config"]["num_hidden_layers"] - model_config["tts_backbone_num_hidden_layers"]
    )
    model_config["tts_text_config"] = model_config["text_config"].copy()
    model_config["tts_text_config"]["num_hidden_layers"] = model_config.pop("tts_backbone_num_hidden_layers")

    # clean up main model config
    for key in ["acoustic_vae_dim", "model_type"]:
        model_config.pop(key, None)
    model_config["dtype"] = model_config.pop("torch_dtype")

    # 4) Update state dict to match HF model structure
    logger.info("Converting state dict")
    updated_state_dict = convert_state_dict(original_state_dict)

    # 5) Create VibeVoiceRealTime processor
    logger.info("Creating VibeVoiceRealTime processor")

    # Explicitly use Qwen2TokenizerFast to ensure proper class name in config
    tokenizer = Qwen2TokenizerFast.from_pretrained(language_model_pretrained_name)
    processor = VibeVoiceRealTimeProcessor(tokenizer=tokenizer)
    processor.save_pretrained(output_dir)

    # Custom pad token for VibeVoice: https://github.com/microsoft/VibeVoice/blob/d295d1e1d0fff1ad42bc0450d5b593f8e59356b9/vibevoice/modular/modular_vibevoice_text_tokenizer.py#L181
    pad_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model_config["pad_token_id"] = pad_token_id

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

    # 6) Create and save full VibeVoice model
    logger.info("Creating full model")
    vibevoice_config = VibeVoiceRealTimeConfig(**model_config)
    vibevoice_model = VibeVoiceRealTimeForConditionalGeneration(vibevoice_config).to(dtype)
    logger.info(f"Number of parameters in model: {len(vibevoice_model.state_dict())}")

    # -- print dtypes of key components for verification
    logger.info(f"Acoustic connector dtype: {next(vibevoice_model.model.multi_modal_projector.parameters()).dtype}")
    logger.info(f"Language model dtype: {next(vibevoice_model.model.language_model.parameters()).dtype}")
    logger.info(f"TTS language model dtype: {next(vibevoice_model.model.tts_language_model.parameters()).dtype}")
    logger.info(f"Acoustic decoder dtype: {next(vibevoice_model.model.audio_tower.parameters()).dtype}")
    logger.info(f"Diffusion head dtype: {next(vibevoice_model.model.diffusion_head.parameters()).dtype}")

    # -- load into HF model
    logger.info("Loading weights into model")
    load_result = vibevoice_model.load_state_dict(updated_state_dict, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"{len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        raise ValueError(f"{len(load_result.missing_keys)} missing keys: {load_result.missing_keys}")
    logger.info("Full model checkpoint loaded successfully")

    # Set default generation config
    vibevoice_model.generation_config._from_model_config = False
    # https://github.com/microsoft/VibeVoice/blob/79470ff5768e17cbef6a3e1a93d1fd82ecc9a00d/demo/realtime_model_inference_from_file.py#L129
    vibevoice_model.generation_config.cfg_scale = 1.5
    vibevoice_model.generation_config.do_sample = False
    vibevoice_model.generation_config.sampling_rate = 24000
    vibevoice_model.generation_config.noise_scheduler_class = "DPMSolverMultistepScheduler"
    vibevoice_model.generation_config.noise_scheduler_config = {
        "num_train_timesteps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "v_prediction",
    }
    # https://github.com/microsoft/VibeVoice/blob/79470ff5768e17cbef6a3e1a93d1fd82ecc9a00d/demo/realtime_model_inference_from_file.py#L225C11-L225C35
    vibevoice_model.generation_config.n_diffusion_steps = 5
    # https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/blob/main/config.json#L51
    vibevoice_model.generation_config.max_new_tokens = 8192
    vibevoice_model.generation_config.max_length = 8192
    # https://github.com/microsoft/VibeVoice/blob/6c7369bb311f42e33b5c51715ca047c9e0757bc6/vibevoice/modular/modeling_vibevoice_streaming_inference.py#L29
    vibevoice_model.generation_config.text_window_size = 5
    vibevoice_model.generation_config.speech_window_size = 6

    logger.info(f"Saving model to {output_dir}")
    vibevoice_model.save_pretrained(output_dir)

    # -- push to hub
    if push_to_hub is not None:
        logger.info(f"Pushing model to Hub: {push_to_hub}")
        vibevoice_model.push_to_hub(push_to_hub)

    # 7) Check model
    logger.info("Verifying conversion by reloading model")
    gc.collect()
    VibeVoiceRealTimeProcessor.from_pretrained(output_dir)
    VibeVoiceRealTimeForConditionalGeneration.from_pretrained(output_dir, dtype=dtype, device_map="auto")
    logger.info("Model reloaded successfully!")
    logger.info("Conversion complete!")


"""
Conversion script to convert original VibeVoice model into three HF checkpoints for:
- VibeVoiceRealTimeForConditionalGeneration

# -- download checkpoint and configs
wget -P /raid/eric/vibevoice_0.5b \
  https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/main/preprocessor_config.json \
  https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/main/config.json \
  https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/main/model.safetensors

# -- run conversion
python src/transformers/models/vibevoice_realtime/convert_vibevoice_realtime_to_hf.py \
    --checkpoint  /raid/eric/vibevoice_0.5b/model.safetensors \
    --output_dir /raid/eric/vibevoice/hf_vibevoice_0.5b \
    --config_path /raid/eric/vibevoice_0.5b/config.json \
    --processor_config /raid/eric/vibevoice_0.5b/preprocessor_config.json \
    --push_to_hub bezzam/VibeVoice-Realtime-0.5B-hf

# -- converted voice embeddings should be added to the hub (not done automatically by this script)
by running this script: https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-convert_realtime_presets-py
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

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        args.config_path,
        args.push_to_hub,
        bfloat16=not args.float32,
        processor_config=args.processor_config,
    )
