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
import torch
from safetensors.torch import load_file
import json
from transformers import VibeVoiceConfig, VibeVoiceModel, VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel, VibeVoiceAcousticTokenizerConfig, VibeVoiceSemanticTokenizerConfig


def update_state_dict_for_hf_model(state_dict):
    """
    Update the state_dict to match the HuggingFace model structure.
    """
    updated_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Handle conv.conv -> conv mapping for semantic tokenizer SConv1d layers only
        # This removes one level of .conv nesting
        if "semantic_tokenizer" in key: # TODO remove when acoustic tokeniezer also updated!
            # Handle downsample_layers Sequential removal: .X.0.conv -> .X.conv
            if "downsample_layers." in key and ".0.conv." in key:
                new_key = new_key.replace(".0.conv.", ".conv.")
            # Handle ConvNext1DLayer mixer simplification: mixer.conv.conv.conv.* -> mixer.*
            if "mixer.conv.conv.conv." in key:
                new_key = new_key.replace("mixer.conv.conv.conv.", "mixer.")
            # Handle general conv.conv -> conv mapping (after mixer handling to avoid conflicts)
            elif ".conv.conv." in key:
                new_key = new_key.replace(".conv.conv.", ".conv.")

        updated_state_dict[new_key] = value
    
    return updated_state_dict


def convert_checkpoint(checkpoint, config_path, push_to_hub, bfloat16):

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

    # -- cleanup
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

    # TODO same for acoustic tokenizer
    model_config["acoustic_tokenizer_config"]["encoder_depths"] = list(map(int, model_config["acoustic_tokenizer_config"]["encoder_depths"].split("-")))
    # if "std_dist_type" in model_config["acoustic_tokenizer_config"]:
    #     model_config["acoustic_tokenizer_config"]["sample_latent"] = False if model_config["semantic_tokenizer_config"]["std_dist_type"] == "none" else True
    #     del model_config["acoustic_tokenizer_config"]["std_dist_type"]


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

    # 6) Create and save full VibeVoice model
    print("\n=== Creating full model ===")
    model_config["acoustic_tokenizer_config"] = acoustic_config.to_dict()
    model_config["semantic_tokenizer_config"] = semantic_config.to_dict()
    vibevoice_config = VibeVoiceConfig(**model_config)
    vibevoice_model = VibeVoiceModel(vibevoice_config).to(dtype)
    # -- load into HF model
    missing, unexpected = vibevoice_model.load_state_dict(updated_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")
    # -- push to hub
    if push_to_hub is not None:
        print(f"------ Pushing full VibeVoice model to hub as {push_to_hub} ------")
        vibevoice_model.push_to_hub(push_to_hub)


    # # TODO create audio feature extractor / processor config


"""
Conversion script to convert original VibeVoice model into three HF checkpoints:
- ViveVoiceModel
- VibeVoiceAcousticTokenizerModel
- VibeVoiceSemanticTokenizerModel

```bash
# -- download checkpoint and config
python src/transformers/models/vibevoice/download_vibevoice_checkpoint.py
wget https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json -P /raid/eric/vibevoice

# -- run conversion
python src/transformers/models/vibevoice/convert_vibevoice_to_hf.py \
    --checkpoint /raid/eric/vibevoice/VibeVoice-1.5B-combined.safetensors \
    --config_path /raid/eric/vibevoice/config.json \
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
    parser.add_argument(
        "--config_path", default=None, type=str, help="Path to hf config.json of model to convert"
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
        args.config_path,
        args.push_to_hub,
        bfloat16=not args.float32,
    )
