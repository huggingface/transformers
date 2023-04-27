# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import re

import torch
from CLAP import create_model

from transformers import AutoFeatureExtractor, ClapConfig, ClapModel


KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

processor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused", truncation="rand_trunc")


def init_clap(checkpoint_path, enable_fusion=False):
    model, model_cfg = create_model(
        "HTSAT-tiny",
        "roberta",
        checkpoint_path,
        precision="fp32",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enable_fusion=enable_fusion,
        fusion_type="aff_2d" if enable_fusion else None,
    )
    return model, model_cfg


def rename_state_dict(state_dict):
    model_state_dict = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in state_dict.items():
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # replace sequential layers with list
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # Because in CLAP they use `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            model_state_dict[key] = value

    return model_state_dict


def convert_clap_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, enable_fusion=False):
    clap_model, clap_model_cfg = init_clap(checkpoint_path, enable_fusion=enable_fusion)

    clap_model.eval()
    state_dict = clap_model.state_dict()
    state_dict = rename_state_dict(state_dict)

    transformers_config = ClapConfig()
    transformers_config.audio_config.enable_fusion = enable_fusion
    model = ClapModel(transformers_config)

    # ignore the spectrogram embedding layer
    model.load_state_dict(state_dict, strict=False)

    model.save_pretrained(pytorch_dump_folder_path)
    transformers_config.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--enable_fusion", action="store_true", help="Whether to enable fusion or not")
    args = parser.parse_args()

    convert_clap_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.enable_fusion)
