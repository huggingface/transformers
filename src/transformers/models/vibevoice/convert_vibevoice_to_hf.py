# coding=utf-8
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
import os
import re

import torch
from safetensors.torch import load_file

from transformers import (
    Qwen2TokenizerFast,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceConfig,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceForConditionalGeneration,
    VibeVoiceProcessor,
    VibeVoiceSemanticTokenizerConfig,
)


def update_state_dict_for_hf_model(state_dict):
    """
    Update the state_dict to match the HuggingFace model structure.
    """
    updated_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle semantic tokenizer transformations
        if "semantic_tokenizer" in key:
            if "downsample_layers.0.0.conv." in key:
                new_key = new_key.replace("downsample_layers.0.0.conv.", "stem.conv.conv.")
            elif "stages.0." in key:
                new_key = new_key.replace("stages.0.", "stem.stage.")
            elif "downsample_layers." in key and not "downsample_layers.0" in key:
                match = re.search(r'downsample_layers\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since downsample_layers[0] became stem
                    new_key = re.sub(r'downsample_layers\.\d+\.0\.conv\.', f'conv_layers.{new_idx}.conv.conv.', new_key)
            elif "stages." in key and not "stages.0." in key:
                match = re.search(r'stages\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since stages[0] became stem
                    new_key = re.sub(r'stages\.\d+\.', f'conv_layers.{new_idx}.stage.', new_key)
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.conv.")
            if ".conv.conv.conv." in new_key:
                new_key = new_key.replace(".conv.conv.conv.", ".conv.conv.")
            elif ".conv.conv." in key and "stem.conv.conv" not in new_key and "conv_layers." not in new_key:
                new_key = new_key.replace(".conv.conv.", ".conv.")

        # Handle acoustic tokenizer transformations
        if "acoustic_tokenizer.encoder" in key:
            if "downsample_layers.0.0.conv." in key:
                new_key = new_key.replace("downsample_layers.0.0.conv.", "stem.conv.conv.")
            elif "stages.0." in key:
                new_key = new_key.replace("stages.0.", "stem.stage.")
            elif "downsample_layers." in key and not "downsample_layers.0" in key:
                match = re.search(r'downsample_layers\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since downsample_layers[0] became stem
                    new_key = re.sub(r'downsample_layers\.\d+\.0\.conv\.', f'conv_layers.{new_idx}.conv.conv.', new_key)
            elif "stages." in key and not "stages.0." in key:
                match = re.search(r'stages\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since stages[0] became stem
                    new_key = re.sub(r'stages\.\d+\.', f'conv_layers.{new_idx}.stage.', new_key)
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.conv.")
            if ".conv.conv.conv." in new_key:
                new_key = new_key.replace(".conv.conv.conv.", ".conv.conv.")
            elif ".conv.conv." in key and "stem.conv.conv" not in new_key and "conv_layers." not in new_key:
                new_key = new_key.replace(".conv.conv.", ".conv.")
        if "acoustic_tokenizer.decoder" in key:
            if "upsample_layers.0.0.conv.conv." in key:
                new_key = new_key.replace("acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.", "acoustic_tokenizer.decoder.stem.conv.conv.")
            elif "stages.0." in key:
                new_key = new_key.replace("stages.0.", "stem.stage.")
            elif "upsample_layers." in key and not "upsample_layers.0" in key:
                match = re.search(r'upsample_layers\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since upsample_layers[0] became conv0
                    new_key = re.sub(r'upsample_layers\.\d+\.0\.convtr\.convtr\.', f'conv_layers.{new_idx}.convtr.convtr.', new_key)
            elif "stages." in key and not "stages.0." in key:
                match = re.search(r'stages\.(\d+)', key)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = old_idx - 1  # Shift down by 1 since stages[0] became stage0
                    new_key = re.sub(r'stages\.\d+\.', f'conv_layers.{new_idx}.stage.', new_key)
            if "head.conv." in key:
                new_key = new_key.replace("head.conv.", "head.")
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.conv.")

        # Handle main model
        if "prediction_head." in key:
            key = key.replace("prediction_head.", "diffusion_head.")
            new_key = new_key.replace("prediction_head.", "diffusion_head.")
        if "diffusion_head.t_embedder.mlp." in key:
            if "diffusion_head.t_embedder.mlp.0." in key:
                new_key = new_key.replace(
                    "diffusion_head.t_embedder.mlp.0.", "diffusion_head.timestep_embedder.layer_1."
                )
            elif "diffusion_head.t_embedder.mlp.2." in key:
                new_key = new_key.replace(
                    "diffusion_head.t_embedder.mlp.2.", "diffusion_head.timestep_embedder.layer_2."
                )
        if "diffusion_head.final_layer.linear." in key and "adaLN_modulation" not in key:
            new_key = new_key.replace("diffusion_head.final_layer.linear.", "diffusion_head.final_layer.linear_2.")
        if "diffusion_head.final_layer.adaLN_modulation." in key:
            if ".adaLN_modulation.1." in key:
                new_key = new_key.replace(".adaLN_modulation.1.", ".linear_1.")
        if "diffusion_head.layers." in key and ".adaLN_modulation." in key:
            if ".adaLN_modulation.1." in key:
                new_key = new_key.replace(".adaLN_modulation.1.", ".linear.")
        if "model.speech_scaling_factor" in key:
            new_key = new_key.replace("model.speech_scaling_factor", "latent_scaling_factor")
        if "model.speech_bias_factor" in key:
            new_key = new_key.replace("model.speech_bias_factor", "latent_bias_factor")

        updated_state_dict[new_key] = value

    return updated_state_dict


def convert_checkpoint(
    checkpoint, output_dir, config_path, push_to_hub, bfloat16, processor_config=None
):
    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) Load state dict from safetensors checkpoint
    original_state_dict = load_file(checkpoint)

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
        if "1.5B" in checkpoint:
            language_model_pretrained_name = "Qwen/Qwen2.5-1.5B"
        else:
            language_model_pretrained_name = "Qwen/Qwen2.5-7B"
    feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor(**audio_config)

    # 3) Prepare model configuration
    # -- Load
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # clean up semantic tokenizer config
    model_config["semantic_tokenizer_config"]["encoder_depths"] = list(
        map(int, model_config["semantic_tokenizer_config"]["encoder_depths"].split("-"))
    )
    # -- reverse order of ratios here instead of in modeling
    model_config["semantic_tokenizer_config"]["downsampling_ratios"] = list(
        reversed(model_config["semantic_tokenizer_config"]["encoder_ratios"])
    )
    del model_config["semantic_tokenizer_config"]["encoder_ratios"]
    model_config["semantic_tokenizer_config"]["n_filters"] = model_config["semantic_tokenizer_config"].pop(
        "encoder_n_filters"
    )
    model_config["semantic_tokenizer_config"]["depths"] = model_config["semantic_tokenizer_config"].pop(
        "encoder_depths"
    )
    model_config["semantic_tokenizer_config"]["hidden_size"] = model_config["semantic_tokenizer_config"].pop("vae_dim")
    model_config["semantic_tokenizer_config"]["bias"] = model_config["semantic_tokenizer_config"].pop("conv_bias")
    # -- remove unused / constant parameters that lead to unused code paths removed in HF model
    if "mixer_layer" in model_config["semantic_tokenizer_config"]:
        del model_config["semantic_tokenizer_config"]["mixer_layer"]
    if "layernorm" in model_config["semantic_tokenizer_config"]:
        del model_config["semantic_tokenizer_config"]["layernorm"]
    if "disable_last_norm" in model_config["semantic_tokenizer_config"]:
        del model_config["semantic_tokenizer_config"]["disable_last_norm"]
    if "conv_norm" in model_config["semantic_tokenizer_config"]:
        del model_config["semantic_tokenizer_config"]["conv_norm"]
    if "corpus_normalize" in model_config["semantic_tokenizer_config"]:
        del model_config["semantic_tokenizer_config"]["corpus_normalize"]
    if "std_dist_type" in model_config["semantic_tokenizer_config"]:
        # No vae component, so no sampling
        del model_config["semantic_tokenizer_config"]["std_dist_type"]
    if "layernorm_elementwise_affine" in model_config["semantic_tokenizer_config"]:
        del model_config["semantic_tokenizer_config"]["layernorm_elementwise_affine"]
    if "layernorm_eps" in model_config["semantic_tokenizer_config"]:
        model_config["semantic_tokenizer_config"]["rms_norm_eps"] = model_config["semantic_tokenizer_config"][
            "layernorm_eps"
        ]
        del model_config["semantic_tokenizer_config"]["layernorm_eps"]
    if "pad_mode" in model_config["semantic_tokenizer_config"]:
        # always "constant"
        del model_config["semantic_tokenizer_config"]["pad_mode"]
    if "fix_std" in model_config["semantic_tokenizer_config"]:
        # Only delete for semantic model!
        del model_config["semantic_tokenizer_config"]["fix_std"]
    if "causal" in model_config["semantic_tokenizer_config"]:
        # always True
        del model_config["semantic_tokenizer_config"]["causal"]

    # clean up acoustic tokenizer config
    model_config["acoustic_tokenizer_config"]["encoder_depths"] = list(
        map(int, model_config["acoustic_tokenizer_config"]["encoder_depths"].split("-"))
    )
    if "std_dist_type" in model_config["acoustic_tokenizer_config"]:
        # always gaussian
        del model_config["acoustic_tokenizer_config"]["std_dist_type"]
    # -- reverse order of ratios here instead of in modeling
    model_config["acoustic_tokenizer_config"]["downsampling_ratios"] = list(
        reversed(model_config["acoustic_tokenizer_config"]["encoder_ratios"])
    )
    del model_config["acoustic_tokenizer_config"]["encoder_ratios"]
    model_config["acoustic_tokenizer_config"]["n_filters"] = model_config["acoustic_tokenizer_config"].pop(
        "encoder_n_filters"
    )
    model_config["acoustic_tokenizer_config"]["depths"] = model_config["acoustic_tokenizer_config"].pop(
        "encoder_depths"
    )
    model_config["acoustic_tokenizer_config"]["hidden_size"] = model_config["acoustic_tokenizer_config"].pop("vae_dim")
    model_config["acoustic_tokenizer_config"]["bias"] = model_config["acoustic_tokenizer_config"].pop("conv_bias")
    # -- original hardcodes a scaling factor for vae_std: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L963
    model_config["acoustic_tokenizer_config"]["vae_std"] = (
        model_config["acoustic_tokenizer_config"].pop("fix_std") / 0.8
    )
    # -- remove decoder parameters as they can be derived from encoder ones
    if "decoder_depths" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["decoder_depths"]
    if "decoder_n_filters" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["decoder_n_filters"]
    if "decoder_ratios" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["decoder_ratios"]
    # -- remove unused / constant parameters that lead to unused code paths removed in HF model
    if "mixer_layer" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["mixer_layer"]
    if "layernorm" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["layernorm"]
    if "disable_last_norm" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["disable_last_norm"]
    if "conv_norm" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["conv_norm"]
    if "corpus_normalize" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["corpus_normalize"]
    if "layernorm_elementwise_affine" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["layernorm_elementwise_affine"]
    if "layernorm_eps" in model_config["acoustic_tokenizer_config"]:
        model_config["acoustic_tokenizer_config"]["rms_norm_eps"] = model_config["acoustic_tokenizer_config"][
            "layernorm_eps"
        ]
        del model_config["acoustic_tokenizer_config"]["layernorm_eps"]
    if "pad_mode" in model_config["acoustic_tokenizer_config"]:
        # always "constant"
        del model_config["acoustic_tokenizer_config"]["pad_mode"]
    if "causal" in model_config["acoustic_tokenizer_config"]:
        # always True
        del model_config["acoustic_tokenizer_config"]["causal"]

    # clean up diffusion head config
    model_config["intermediate_size"] = int(
        model_config["diffusion_head_config"].pop("head_ffn_ratio") * model_config["diffusion_head_config"]["hidden_size"]
    )
    model_config["diffusion_head_config"]["num_head_layers"] = model_config["diffusion_head_config"].pop("head_layers")
    if model_config["diffusion_head_config"]["ddpm_beta_schedule"] == "cosine":
        model_config["diffusion_head_config"]["ddpm_beta_schedule"] = "squaredcos_cap_v2"
    model_config["diffusion_head_config"].pop("speech_vae_dim")
    model_config["diffusion_head_config"].pop("diffusion_type")
    model_config["diffusion_head_config"].pop("ddpm_batch_mul")
    model_config["diffusion_head_config"].pop("latent_size")    # same as acoustic tokenizer hidden size
    model_config["diffusion_head_config"].pop("hidden_size")    # same as text model hidden size
    # -- flatten diffusion head config
    for k, v in model_config["diffusion_head_config"].items():
        model_config[k] = v
    del model_config["diffusion_head_config"]

    # clean up language model config
    model_config["text_config"] = model_config.pop("decoder_config")
    model_config["text_config"]["dtype"] = model_config["text_config"].pop("torch_dtype")

    # clean up main model config
    if "acoustic_vae_dim" in model_config:
        del model_config["acoustic_vae_dim"]
    if "semantic_vae_dim" in model_config:
        del model_config["semantic_vae_dim"]
    if "tie_word_embeddings" in model_config:
        del model_config["tie_word_embeddings"]
    model_config["dtype"] = model_config.pop("torch_dtype")

    # 4) Update state dict to match HF model structure
    updated_state_dict = update_state_dict_for_hf_model(original_state_dict)

    # 5) Create semantic tokenizer config
    print("\n=== Creating semantic tokenizer ===")
    semantic_config = VibeVoiceSemanticTokenizerConfig(**model_config["semantic_tokenizer_config"])

    # 6) Create acoustic tokenizer config
    print("\n=== Creating acoustic tokenizer ===")
    acoustic_config = VibeVoiceAcousticTokenizerConfig(**model_config["acoustic_tokenizer_config"])

    # 7) Create VibeVoice processor
    # -- load processor config
    print("\n=== Creating VibeVoice processor ===")

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

    # Ensure tokenizer_config.json has the correct tokenizer_class
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["tokenizer_class"] = "Qwen2TokenizerFast"

        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

    if push_to_hub is not None:
        print(f"------ Pushing processor to hub as {push_to_hub} ------")
        processor.push_to_hub(push_to_hub)

    # 8) Create and save full VibeVoice model
    print("\n=== Creating full model ===")
    model_config["acoustic_tokenizer_config"] = acoustic_config.to_dict()
    model_config["semantic_tokenizer_config"] = semantic_config.to_dict()
    vibevoice_config = VibeVoiceConfig(**model_config)
    vibevoice_model = VibeVoiceForConditionalGeneration(vibevoice_config).to(dtype)
    # -- print dtypes of key components for verification
    print("Acoustic connector dtype : ", next(vibevoice_model.model.acoustic_connector.parameters()).dtype)
    print("Semantic connector dtype : ", next(vibevoice_model.model.semantic_connector.parameters()).dtype)
    print("Language model dtype : ", next(vibevoice_model.model.language_model.parameters()).dtype)
    print("Acoustic tokenizer dtype : ", next(vibevoice_model.model.acoustic_tokenizer.parameters()).dtype)
    print("Semantic tokenizer dtype : ", next(vibevoice_model.model.semantic_tokenizer.parameters()).dtype)
    print("Diffusion head dtype : ", next(vibevoice_model.model.diffusion_head.parameters()).dtype)

    # -- load into HF model
    if model_config["text_config"].get("tie_word_embeddings", False):
        # 1.5B ties weights: https://huggingface.co/microsoft/VibeVoice-1.5B/blob/main/config.json#L61
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modeling_vibevoice_inference.py#L123
        updated_state_dict["lm_head.weight"] = updated_state_dict["model.language_model.embed_tokens.weight"]
    # 7B does not tie weights: https://huggingface.co/vibevoice/VibeVoice-7B/blob/main/config.json#L113

    missing, unexpected = vibevoice_model.load_state_dict(updated_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    print("Full model checkpoint loaded successfully.")

    # Calculate speech token IDs from tokenizer for generation config
    audio_bos_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    audio_diffusion_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")

    # Set default generation config
    vibevoice_model.generation_config._from_model_config = False
    vibevoice_model.generation_config.audio_bos_token_id = audio_bos_token_id
    vibevoice_model.generation_config.audio_eos_token_id = audio_eos_token_id
    vibevoice_model.generation_config.audio_diffusion_id = audio_diffusion_id
    vibevoice_model.generation_config.bos_token_id = tokenizer.bos_token_id
    vibevoice_model.generation_config.eos_token_id = tokenizer.eos_token_id
    vibevoice_model.generation_config.pad_token_id = tokenizer.pad_token_id
    vibevoice_model.generation_config.cfg_scale = 1.3
    vibevoice_model.n_diffusion_steps = 10
    vibevoice_model.generation_config.do_sample = False
    vibevoice_model.generation_config.noise_scheduler_class = "DPMSolverMultistepScheduler"
    vibevoice_model.generation_config.noise_scheduler_config = {
        "num_train_timesteps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "v_prediction",
    }
    vibevoice_model.generation_config.n_diffusion_steps = 10
    if "7B" in checkpoint:
        vibevoice_model.generation_config.max_new_tokens = 20250
        vibevoice_model.generation_config.max_length = 20250
    else:
        vibevoice_model.generation_config.max_new_tokens = 40500
        vibevoice_model.generation_config.max_length = 40500

    vibevoice_model.save_pretrained(output_dir)
    # -- push to hub
    if push_to_hub is not None:
        print(f"------ Pushing full VibeVoice model to hub as {push_to_hub} ------")
        vibevoice_model.push_to_hub(push_to_hub)

    # 9) Check model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    VibeVoiceProcessor.from_pretrained(output_dir)
    VibeVoiceForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


"""
Conversion script to convert original VibeVoice model into a checkpoint for `VibeVoiceForConditionalGeneration`


# 1.5 Model: https://huggingface.co/microsoft/VibeVoice-1.5B

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
Models will be pushed to: bezzam/VibeVoice-1.5B


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
Models will be pushed to: bezzam/VibeVoice-7B

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
