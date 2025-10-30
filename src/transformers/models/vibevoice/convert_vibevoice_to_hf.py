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

import torch
from safetensors.torch import load_file

from transformers import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceConfig,
    VibeVoiceFeatureExtractor,
    VibeVoiceForConditionalGeneration,
    VibeVoiceProcessor,
    VibeVoiceSemanticTokenizerConfig,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceTokenizer,
)


def update_state_dict_for_hf_model(state_dict):
    """
    Update the state_dict to match the HuggingFace model structure.
    """
    updated_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle conv.conv -> conv mapping for semantic tokenizer SConv1d layers
        # This removes one level of .conv nesting
        if "semantic_tokenizer" in key:
            # Handle downsample_layers Sequential removal: .X.0.conv -> .X.conv
            if "downsample_layers." in key and ".0.conv." in key:
                new_key = new_key.replace(".0.conv.", ".conv.")
            # Handle ConvNext1DLayer mixer simplification: mixer.conv.conv.conv.* -> mixer.*
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.")
            # Handle general conv.conv -> conv mapping (after mixer handling to avoid conflicts)
            elif ".conv.conv." in key:
                new_key = new_key.replace(".conv.conv.", ".conv.")

        # Handle conv.conv -> conv mapping for acoustic tokenizer encoder layers
        # This removes one level of .conv nesting for the updated TokenizerEncoder
        if "acoustic_tokenizer.encoder" in key:
            # Handle downsample_layers Sequential removal: .X.0.conv -> .X.conv
            if "downsample_layers." in key and ".0.conv." in key:
                new_key = new_key.replace(".0.conv.", ".conv.")
            # Handle ConvNext1DLayer mixer simplification: mixer.conv.conv.conv.* -> mixer.*
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.")
            # Handle general conv.conv -> conv mapping (after mixer handling to avoid conflicts)
            elif ".conv.conv." in key:
                new_key = new_key.replace(".conv.conv.", ".conv.")

        # Handle conv.conv -> conv mapping for acoustic tokenizer decoder layers
        # This removes one level of .conv nesting for the updated TokenizerDecoder
        if "acoustic_tokenizer.decoder" in key:
            # Handle stem layer (upsample_layers.0) conv.conv -> conv mapping
            if "upsample_layers.0.0.conv.conv." in key:
                new_key = new_key.replace("upsample_layers.0.0.conv.conv.", "upsample_layers.0.conv.")
            # Handle transpose conv layers: convtr.convtr.* -> convtr.*
            elif "0.convtr.convtr." in key:
                new_key = new_key.replace("0.convtr.convtr.", "convtr.")
            # Handle head layer conv.conv -> conv mapping
            elif "head.conv." in key:
                new_key = new_key.replace("head.conv.", "head.")
            # Handle stages (changed from Block1D to VibeVoiceAcousticTokenizerConvNext1dLayer)
            # Original Block1D had: mixer.conv.conv.conv.* -> VibeVoiceAcousticTokenizerConvNext1dLayer has: mixer.*
            elif "stages." in key and "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.")

        # Handle prediction_head -> diffusion_head mapping
        if "prediction_head." in key:
            key = key.replace("prediction_head.", "diffusion_head.")
            new_key = new_key.replace("prediction_head.", "diffusion_head.")

        # Handle TimestepEmbedder MLP Sequential -> individual layers mapping
        if "diffusion_head.t_embedder.mlp." in key:
            if "diffusion_head.t_embedder.mlp.0." in key:
                new_key = new_key.replace("diffusion_head.t_embedder.mlp.0.", "diffusion_head.timestep_embedder.layer_1.")
            elif "diffusion_head.t_embedder.mlp.2." in key:
                new_key = new_key.replace("diffusion_head.t_embedder.mlp.2.", "diffusion_head.timestep_embedder.layer_2.")

        # Handle FinalLayer linear -> linear_2 mapping
        if "diffusion_head.final_layer.linear." in key and "adaLN_modulation" not in key:
            new_key = new_key.replace("diffusion_head.final_layer.linear.", "diffusion_head.final_layer.linear_2.")

        # Handle FinalLayer adaLN_modulation Sequential -> individual layers mapping
        if "diffusion_head.final_layer.adaLN_modulation." in key:
            if ".adaLN_modulation.1." in key:
                new_key = new_key.replace(".adaLN_modulation.1.", ".linear_1.")

        # Handle HeadLayer adaLN_modulation Sequential -> individual layers mapping
        if "diffusion_head.layers." in key and ".adaLN_modulation." in key:
            if ".adaLN_modulation.1." in key:
                new_key = new_key.replace(".adaLN_modulation.1.", ".linear.")

        updated_state_dict[new_key] = value

    return updated_state_dict


def convert_checkpoint(checkpoint, output_dir, config_path, push_to_hub, bfloat16, processor_config=None):

    if bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) Load state dict from safetensors checkpoint
    original_state_dict = load_file(checkpoint)
    # -- remove "model." prefix
    if list(original_state_dict.keys())[0].startswith("model."):
        original_state_dict = {k[len("model."):]: v for k, v in original_state_dict.items()}

    # 2) Prepare model configuration
    # -- Load
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # clean up semantic tokenizer config
    model_config["semantic_tokenizer_config"]["encoder_depths"] = list(map(int, model_config["semantic_tokenizer_config"]["encoder_depths"].split("-")))
    # -- reverse order of ratios here instead of in modeling
    model_config["semantic_tokenizer_config"]["downsampling_ratios"] = list(reversed(model_config["semantic_tokenizer_config"]["encoder_ratios"]))
    del model_config["semantic_tokenizer_config"]["encoder_ratios"]
    model_config["semantic_tokenizer_config"]["n_filters"] = model_config["semantic_tokenizer_config"].pop("encoder_n_filters")
    model_config["semantic_tokenizer_config"]["depths"] = model_config["semantic_tokenizer_config"].pop("encoder_depths")
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
        model_config["semantic_tokenizer_config"]["rms_norm_eps"] = model_config["semantic_tokenizer_config"]["layernorm_eps"]
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
    model_config["acoustic_tokenizer_config"]["encoder_depths"] = list(map(int, model_config["acoustic_tokenizer_config"]["encoder_depths"].split("-")))
    if "std_dist_type" in model_config["acoustic_tokenizer_config"]:
        # always gaussian
        del model_config["acoustic_tokenizer_config"]["std_dist_type"]
    # -- reverse order of ratios here instead of in modeling
    model_config["acoustic_tokenizer_config"]["downsampling_ratios"] = list(reversed(model_config["acoustic_tokenizer_config"]["encoder_ratios"]))
    del model_config["acoustic_tokenizer_config"]["encoder_ratios"]
    model_config["acoustic_tokenizer_config"]["n_filters"] = model_config["acoustic_tokenizer_config"].pop("encoder_n_filters")
    model_config["acoustic_tokenizer_config"]["depths"] = model_config["acoustic_tokenizer_config"].pop("encoder_depths")
    model_config["acoustic_tokenizer_config"]["hidden_size"] = model_config["acoustic_tokenizer_config"].pop("vae_dim")
    model_config["acoustic_tokenizer_config"]["bias"] = model_config["acoustic_tokenizer_config"].pop("conv_bias")
    model_config["acoustic_tokenizer_config"]["vae_std"] = model_config["acoustic_tokenizer_config"].pop("fix_std")
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
        model_config["acoustic_tokenizer_config"]["rms_norm_eps"] = model_config["acoustic_tokenizer_config"]["layernorm_eps"]
        del model_config["acoustic_tokenizer_config"]["layernorm_eps"]
    if "pad_mode" in model_config["acoustic_tokenizer_config"]:
        # always "constant"
        del model_config["acoustic_tokenizer_config"]["pad_mode"]
    if "causal" in model_config["acoustic_tokenizer_config"]:
        # always True
        del model_config["acoustic_tokenizer_config"]["causal"]

    # clean up diffusion head config
    model_config["diffusion_head_config"]["head_ffn_ratio"] = int(model_config["diffusion_head_config"]["head_ffn_ratio"])
    model_config["diffusion_head_config"]["num_head_layers"] = model_config["diffusion_head_config"].pop("head_layers")
    if model_config["diffusion_head_config"]["ddpm_beta_schedule"] == "cosine":
        model_config["diffusion_head_config"]["ddpm_beta_schedule"] = "squaredcos_cap_v2"
    if "speech_vae_dim" in model_config["diffusion_head_config"]:
        del model_config["diffusion_head_config"]["speech_vae_dim"]
    if "diffusion_type" in model_config["diffusion_head_config"]:
        del model_config["diffusion_head_config"]["diffusion_type"]
    if "ddpm_batch_mul" in model_config["diffusion_head_config"]:
        del model_config["diffusion_head_config"]["ddpm_batch_mul"]

    # clean up language model config
    model_config["text_config"] = model_config.pop("decoder_config")
    model_config["text_config"]["dtype"] = model_config["text_config"].pop("torch_dtype")

    # clean up main model config
    if "acoustic_vae_dim" in model_config:
        del model_config["acoustic_vae_dim"]
    if "semantic_vae_dim" in model_config:
        del model_config["semantic_vae_dim"]
    if "num_hidden_layers" in model_config:
        del model_config["num_hidden_layers"]
    model_config["dtype"] = model_config.pop("torch_dtype")

    # 3) Update state dict to match HF model structure
    updated_state_dict = update_state_dict_for_hf_model(original_state_dict)

    # 4) Create and save semantic tokenizer
    print("\n=== Creating semantic tokenizer ===")
    semantic_config = VibeVoiceSemanticTokenizerConfig(**model_config["semantic_tokenizer_config"])
    semantic_model = VibeVoiceSemanticTokenizerModel(semantic_config).to(dtype)
    # -- filter for semantic tokenizer weights
    prefix = "semantic_tokenizer"
    semantic_state_dict = {
        k[len(prefix)+1:]: v  # +1 to remove the dot after the prefix
        for k, v in updated_state_dict.items()
        if k.startswith(prefix)
    }
    # -- load into HF model
    missing, unexpected = semantic_model.load_state_dict(semantic_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    # -- push to hub
    if push_to_hub is not None:
        print(f"------ Pushing to hub as {push_to_hub + '-SemanticTokenizer'} ------")
        semantic_model.push_to_hub(push_to_hub + "-SemanticTokenizer")

    # 5) Create and save acoustic tokenizer
    print("\n=== Creating acoustic tokenizer ===")
    acoustic_config = VibeVoiceAcousticTokenizerConfig(**model_config["acoustic_tokenizer_config"])
    acoustic_model = VibeVoiceAcousticTokenizerModel(acoustic_config).to(dtype)
    # -- filter for acoustic tokenizer weights
    prefix = "acoustic_tokenizer"
    acoustic_state_dict = {
        k[len(prefix)+1:]: v  # +1 to remove the dot after the prefix
        for k, v in updated_state_dict.items()
        if k.startswith(prefix)
    }
    # -- load into HF model
    missing, unexpected = acoustic_model.load_state_dict(acoustic_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    # -- push to hub
    if push_to_hub is not None:
        print(f"------ Pushing to hub as {push_to_hub + '-AcousticTokenizer'} ------")
        acoustic_model.push_to_hub(push_to_hub + "-AcousticTokenizer")

    # 6) Create VibeVoice processor
    # -- load processor config
    print("\n=== Creating VibeVoice processor ===")
    if processor_config is not None:
        with open(processor_config, "r") as f:
            processor_config = json.load(f)
        audio_config = processor_config.get("audio_processor", {})
        language_model_pretrained_name = processor_config.get("language_model_pretrained_name", None)

    # Default to 1.5B model: https://huggingface.co/microsoft/VibeVoice-1.5B/blob/main/preprocessor_config.json
    if "sampling_rate" not in audio_config:
        audio_config["sampling_rate"] = 24000
    if "normalize_audio" not in audio_config:
        audio_config["normalize_audio"] = True
    if "target_dB_FS" not in audio_config:
        audio_config["target_dB_FS"] = -25
    if "eps" not in audio_config:
        audio_config["eps"] = 1e-6
    if language_model_pretrained_name is None:
        language_model_pretrained_name = "Qwen/Qwen2.5-1.5B"

    processor = VibeVoiceProcessor(
        feature_extractor=VibeVoiceFeatureExtractor(**audio_config),
        tokenizer=VibeVoiceTokenizer.from_pretrained(language_model_pretrained_name),
        # audio_tokenizer=VibeVoiceAcousticTokenizerModel.from_pretrained(push_to_hub + "-AcousticTokenizer"),
    )
    processor.save_pretrained(output_dir)

    if push_to_hub is not None:
        print(f"------ Pushing processor to hub as {push_to_hub} ------")
        processor.push_to_hub(push_to_hub)

    # 7) Create and save full VibeVoice model
    print("\n=== Creating full model ===")
    model_config["acoustic_tokenizer_config"] = acoustic_config.to_dict()
    model_config["semantic_tokenizer_config"] = semantic_config.to_dict()
    vibevoice_config = VibeVoiceConfig(**model_config)
    vibevoice_model = VibeVoiceForConditionalGeneration(vibevoice_config).to(dtype)

    # -- print dtypes of key components for verification
    print("Acoustic connector dtype : ", vibevoice_model.acoustic_connector.fc1.weight.dtype)
    print("Semantic connector dtype : ", vibevoice_model.semantic_connector.fc1.weight.dtype)
    print("Language model dtype : ", vibevoice_model.language_model.embed_tokens.weight.dtype)
    print("Acoustic tokenizer dtype : ", vibevoice_model.acoustic_tokenizer.encoder.downsample_layers[0].conv.weight.dtype)
    print("Semantic tokenizer dtype : ", vibevoice_model.semantic_tokenizer.encoder.downsample_layers[0].conv.weight.dtype)
    print("Diffusion head dtype : ", vibevoice_model.diffusion_head.noisy_images_proj.weight.dtype)

    # -- load into HF model
    # add "model." prefix
    updated_state_dict = {f"model.{k}": v for k, v in updated_state_dict.items()}
    # add lm_head weights
    # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modeling_vibevoice_inference.py#L123
    updated_state_dict["lm_head.weight"] = updated_state_dict["model.language_model.embed_tokens.weight"]

    missing, unexpected = vibevoice_model.load_state_dict(updated_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    vibevoice_model.save_pretrained(output_dir)

    # -- push to hub
    if push_to_hub is not None:
        print(f"------ Pushing full VibeVoice model to hub as {push_to_hub} ------")
        vibevoice_model.push_to_hub(push_to_hub)

    # 8) Check model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    VibeVoiceProcessor.from_pretrained(output_dir)
    VibeVoiceForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16)
    # TODO (ebezzam) "auto" not working for: model.speech_scaling_factor, model.speech_bias_factor
    # VibeVoiceForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


"""
Conversion script to convert original VibeVoice model into three HF checkpoints for:
- VibeVoiceForConditionalGeneration
- VibeVoiceAcousticTokenizerModel
- VibeVoiceSemanticTokenizerModel

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
Models will be pushed to:
- bezzam/VibeVoice-1.5B
- bezzam/VibeVoice-1.5B-AcousticTokenizer
- bezzam/VibeVoice-1.5B-SemanticTokenizer

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, default=None, type=str, help="Original VibeVoice model checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Output directory for HuggingFace model")
    parser.add_argument(
        "--config_path", default=None, type=str, help="Path to config.json of model to convert"
    )
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
        processor_config=args.processor_config
    )
