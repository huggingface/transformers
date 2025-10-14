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
from transformers import VibeVoiceConfig, VibeVoiceModel, VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel


def update_state_dict_for_hf_model(state_dict):
    """
    Update the state_dict to match the HuggingFace model structure.
    
    Changes:
    - Remove .conv layer nesting: 'layer.conv.conv.weight' -> 'layer.conv.weight'
    - This is needed because the HF model now has simplified SConv1d structure
    """
    updated_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Handle conv.conv -> conv mapping for semantic tokenizer SConv1d layers only
        # This removes one level of .conv nesting
        if ".conv.conv." in key and "semantic_tokenizer" in key:
            new_key = key.replace(".conv.conv.", ".conv.")
        
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
    # -- Use prepared acoustic and semantic tokenizers
    acoustic_tokenizer = VibeVoiceAcousticTokenizerModel.from_pretrained("bezzam/VibeVoiceAcousticTokenizer").to(dtype)
    semantic_tokenizer = VibeVoiceSemanticTokenizerModel.from_pretrained("bezzam/VibeVoiceSemanticTokenizer").to(dtype)
    model_config["acoustic_tokenizer_config"] = acoustic_tokenizer.config.to_dict()
    model_config["semantic_tokenizer_config"] = semantic_tokenizer.config.to_dict()

    # 3) create config
    model_config = VibeVoiceConfig(**model_config)

    # 4) create model
    model = VibeVoiceModel(model_config).to(dtype)

    # 5) Update state dict to match HuggingFace model structure
    updated_state_dict = update_state_dict_for_hf_model(original_state_dict)
    missing, unexpected = model.load_state_dict(updated_state_dict, strict=False)
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"missing keys found: {missing}")

    # TODO create audio feature extractor here??

    # push to hub
    if push_to_hub is not None:
        model.push_to_hub(push_to_hub)

"""
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
