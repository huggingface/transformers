# coding=utf-8
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""
Weights conversion script for Phi

This script downloads both Phi-1 and Phi-1.5 checkpoints to "checkpoint_path" and then converts the weights to
HugfgingFace model's format and saves them in "pytorch_dump_folder_path".
"""

import argparse
import gc
import os

import torch
from huggingface_hub import hf_hub_download

from transformers import PhiConfig, PhiForCausalLM


_MODELS = {
    "microsoft/phi-1": "https://huggingface.co/microsoft/phi-1/blob/main/pytorch_model.bin",
    "microsoft/phi-1_5": "https://huggingface.co/microsoft/phi-1_5/blob/main/pytorch_model.bin",
}


PHI_MAPPING = {
    "layers.0.wte.weight": "model.embed_tokens.weight",
    "layers.25.linear.bias": "lm_head.bias",
    "layers.25.linear.weight": "lm_head.weight",
    "layers.25.ln.bias": "model.final_layernorm.bias",
    "layers.25.ln.weight": "model.final_layernorm.weight",
    "layers": "model.layers",
    "ln": "input_layernorm",
    "mixer": "self_attn",
    "Wqkv": "query_key_value",
    "out_proj": "dense",
}


def convert_weights(original_weights, mapping, config):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())

    # we change names (1-24) -> layers(0-23) for Phi model layers
    range_change = {
        f"layers.{k}.": f"layers.{v}."
        for k, v in zip(range(1, config.num_hidden_layers + 1), range(0, config.num_hidden_layers))
    }

    mapping.update(**range_change)

    for original_weights_key in original_weights_keys:
        new_key = original_weights_key

        if "rotary_emb" in new_key:
            continue

        if "Wqkv" in new_key:
            if "weight" in new_key:
                weight = original_weights[new_key]
                weights_shape = weight.shape
                weight = (
                    weight.view(3, config.num_attention_heads, -1, config.hidden_size)
                    .transpose(0, 1)
                    .reshape(*weights_shape)
                )
                original_weights[new_key] = weight
            elif "bias" in new_key:
                bias = original_weights[new_key]
                bias_shape = bias.shape
                bias = bias.view(3, config.num_attention_heads, -1).transpose(0, 1).reshape(*bias_shape)
                original_weights[new_key] = bias

        for k, v in mapping.items():
            if k in new_key:
                new_key = new_key.replace(k, v)

        converted_weights[new_key] = original_weights.pop(original_weights_key)

    return converted_weights


def _download(url: str, root: str):
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-1]}"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )


def convert_phi_weights(checkpoint_path, pytorch_dump_folder_path, use_cuda, save_weights_directly):
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    for each_model_name, each_model_url in _MODELS.items():
        converted_checkpoint = {}

        model_path = os.path.join(checkpoint_path, each_model_name + "_" + each_model_url.split("/")[-1])
        if not os.path.exists(model_path):
            print(f"\n{each_model_name} was not found! Downloading it to {model_path}")
            _download(url=each_model_url, root=model_path)
        model_checkpoint = torch.load(model_path, map_location=device)
        model_type = each_model_name.split("/")[1]  # phi-1 or phi-1_5
        config = PhiConfig.from_pretrained(f"susnato/{model_type}_dev")

        # Converting the weights
        converted_checkpoint.update(**convert_weights(model_checkpoint, PHI_MAPPING, config))

        # Save either the whole model or the converted weights
        if save_weights_directly:
            save_weights_path = os.path.join(
                pytorch_dump_folder_path, each_model_name.split("/")[-1] + "_" + each_model_url.split("/")[-1]
            )
            torch.save(converted_checkpoint, save_weights_path)
            print(f"Model weights saved at {save_weights_path}!")

        else:
            model = PhiForCausalLM(config).to(device)
            model.load_state_dict(converted_checkpoint, strict=True)
            save_model_path = os.path.join(pytorch_dump_folder_path, model_type)
            model.save_pretrained(save_model_path)
            print(f"Model saved at {save_model_path}!")

            # release GPU memory for the 2nd model if cuda was used.
            del config, model

        # release GPU memory for the 2nd model if cuda was used.
        del model_checkpoint, converted_checkpoint
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    parser.add_argument(
        "--use_cuda",
        default=False,
        type=bool,
        help="Whether to load the weights on GPU during conversion or not, False by default",
    )
    parser.add_argument(
        "--save_weights_directly",
        default=True,
        type=bool,
        help="Whether to save the weights directly after conversion or load the weight to the Phi model and then save "
        "the Phi model along with weights. True by default",
    )

    args = parser.parse_args()
    convert_phi_weights(args.checkpoint_path, args.pytorch_dump_folder_path, args.use_cuda, args.save_weights_directly)
