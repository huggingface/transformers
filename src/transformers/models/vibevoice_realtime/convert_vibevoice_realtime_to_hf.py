# coding=utf-8
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
import os

import torch
from safetensors.torch import load_file

from transformers import (
    Qwen2TokenizerFast,
    VibeVoiceRealTimeConfig,
    VibeVoiceRealTimeForConditionalGeneration,
    VibeVoiceRealTimeProcessor,
)


def update_state_dict_for_hf_model(state_dict):
    """
    Update the state_dict to match the HuggingFace model structure.
    """
    updated_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle acoustic tokenizer transformations
        if "acoustic_tokenizer.decoder" in key:
            if "upsample_layers.0.0.conv.conv." in key:
                new_key = new_key.replace("upsample_layers.0.0.conv.conv.", "upsample_layers.0.conv.")
            elif "0.convtr.convtr." in key:
                new_key = new_key.replace("0.convtr.convtr.", "convtr.")
            elif "head.conv." in key:
                new_key = new_key.replace("head.conv.", "head.")
            elif "stages." in key and "mixer.conv.conv.conv." in key:
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
        language_model_pretrained_name = "Qwen/Qwen2.5-0.5B"

    # 3) Prepare model configuration
    # -- Load
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # clean up acoustic tokenizer config
    model_config["acoustic_tokenizer_config"]["hidden_size"] = model_config["acoustic_tokenizer_config"].pop("vae_dim")
    model_config["acoustic_tokenizer_config"]["bias"] = model_config["acoustic_tokenizer_config"].pop("conv_bias")
    # -- since decoder only, remove encoder/decoder naming
    model_config["acoustic_tokenizer_config"]["encoder_depths"] = list(
        map(int, model_config["acoustic_tokenizer_config"]["encoder_depths"].split("-"))
    )
    model_config["acoustic_tokenizer_config"]["decoder_depths"] = model_config["acoustic_tokenizer_config"].pop("encoder_depths")[::-1]
    model_config["acoustic_tokenizer_config"]["n_filters"] = model_config["acoustic_tokenizer_config"].pop( "decoder_n_filters")
    del model_config["acoustic_tokenizer_config"]["encoder_n_filters"]
    model_config["acoustic_tokenizer_config"]["upsampling_ratios"] = model_config["acoustic_tokenizer_config"].pop("decoder_ratios")
    del model_config["acoustic_tokenizer_config"]["encoder_ratios"]
    # -- remove all sampling/vae related parameters since decoder only
    if "std_dist_type" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["std_dist_type"]
    if "fix_std" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["fix_std"]
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
        del model_config["acoustic_tokenizer_config"]["pad_mode"]
    if "causal" in model_config["acoustic_tokenizer_config"]:
        del model_config["acoustic_tokenizer_config"]["causal"]
    model_config["acoustic_tokenizer_config"]["model_type"] = "vibevoice_realtime_acoustic_decoder"

    # clean up diffusion head config
    model_config["diffusion_head_config"]["head_ffn_ratio"] = int(
        model_config["diffusion_head_config"]["head_ffn_ratio"]
    )
    model_config["diffusion_head_config"]["num_head_layers"] = model_config["diffusion_head_config"].pop("head_layers")
    if model_config["diffusion_head_config"]["ddpm_beta_schedule"] == "cosine":
        model_config["diffusion_head_config"]["ddpm_beta_schedule"] = "squaredcos_cap_v2"
    if "speech_vae_dim" in model_config["diffusion_head_config"]:
        del model_config["diffusion_head_config"]["speech_vae_dim"]
    if "diffusion_type" in model_config["diffusion_head_config"]:
        del model_config["diffusion_head_config"]["diffusion_type"]
    if "ddpm_batch_mul" in model_config["diffusion_head_config"]:
        del model_config["diffusion_head_config"]["ddpm_batch_mul"]
    # -- flatten diffusion head config
    for k, v in model_config["diffusion_head_config"].items():
        model_config[k] = v
    del model_config["diffusion_head_config"]

    # clean up and configuration language model config -> 2 language models (one for text, one for tts)
    model_config["text_config"] = model_config.pop("decoder_config")
    model_config["text_config"]["dtype"] = model_config["text_config"].pop("torch_dtype")
    model_config["text_config"]["num_hidden_layers"] = model_config["text_config"]["num_hidden_layers"] - model_config["tts_backbone_num_hidden_layers"] 
    model_config["tts_text_config"] = model_config["text_config"].copy()
    model_config["tts_text_config"]["num_hidden_layers"] = model_config.pop("tts_backbone_num_hidden_layers")

    # clean up main model config
    if "acoustic_vae_dim" in model_config:
        del model_config["acoustic_vae_dim"]
    if "hidden_size" in model_config:
        del model_config["hidden_size"]
    model_config["dtype"] = model_config.pop("torch_dtype")

    # 4) Update state dict to match HF model structure
    updated_state_dict = update_state_dict_for_hf_model(original_state_dict)

    # 7) Create VibeVoiceRealTime processor
    print("\n=== Creating VibeVoiceRealTime processor ===")

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
        print(f"------ Pushing processor to hub as {push_to_hub} ------")
        processor.push_to_hub(push_to_hub)

    # 8) Create and save full VibeVoice model
    print("\n=== Creating full model ===")
    vibevoice_config = VibeVoiceRealTimeConfig(**model_config)
    vibevoice_model = VibeVoiceRealTimeForConditionalGeneration(vibevoice_config).to(dtype)

    # -- print dtypes of key components for verification
    print("Acoustic connector dtype : ", vibevoice_model.acoustic_connector.fc1.weight.dtype)
    print("Language model dtype : ", vibevoice_model.language_model.embed_tokens.weight.dtype)
    print(
        "Acoustic tokenizer dtype : ",
        vibevoice_model.acoustic_tokenizer.decoder.upsample_layers[0].conv.weight.dtype,
    )
    print("Diffusion head dtype : ", vibevoice_model.diffusion_head.noisy_images_proj.weight.dtype)

    # -- load into HF model
    missing, unexpected = vibevoice_model.load_state_dict(updated_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    print("Full model checkpoint loaded successfully.")

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

    vibevoice_model.save_pretrained(output_dir)
    # -- push to hub
    if push_to_hub is not None:
        print(f"------ Pushing full VibeVoice model to hub as {push_to_hub} ------")
        vibevoice_model.push_to_hub(push_to_hub)

    # 9) Check model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    VibeVoiceRealTimeProcessor.from_pretrained(output_dir)
    VibeVoiceRealTimeForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


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
    --push_to_hub bezzam/VibeVoice-0.5B

# -- converted voice embeddings should be added to the hub (not done automatically this script)
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
