#!/usr/bin/env python
"""Converts a Moonshine model in Useful Sensors format to Hugging Face format."""
# Copyright 2022 The HuggingFace Inc. team and the OpenAI team. All rights reserved.
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

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from transformers.models.moonshine.modeling_moonshine import MoonshineConfig, MoonshineForConditionalGeneration


# Copied from https://github.com/usefulsensors/moonshine/blob/a1d77cc573b0471ac4602b86f67b3f48d67df1a9/moonshine/model.py
def _get_weights(model_name):
    repo = "UsefulSensors/moonshine"

    return (
        hf_hub_download(repo, f"{x}.weights.h5", subfolder=model_name) for x in ("preprocessor", "encoder", "decoder")
    )


def _read_h5_weights(group, current_key="", weights={}):
    for key in group.keys():
        full_key = f"{current_key}.{key}" if current_key else key
        if isinstance(group[key], h5py.Dataset):
            w = np.array(group[key])
            w = torch.from_numpy(w)
            if len(w.shape) > 1:
                if len(w.shape) == 3:
                    hidden_size = max(list(w.shape))
                    try:
                        w = w.reshape(hidden_size, hidden_size)
                    except RuntimeError:
                        # meaning its a conv layers
                        pass
                w = w.transpose(0, -1)
            weights[full_key] = w
        else:
            _read_h5_weights(group[key], full_key, weights)
    return weights


def _convert_layer_names(name, gated_mlp=False):
    name = re.sub(
        r"layers\.functional(?:_(\d+))?\.layers",
        lambda m: f'layers.{m.group(1) if m.group(1) else "0"}',
        name,
        count=1,
    )
    if gated_mlp:
        name = re.sub(r"functional\.layers\.dense\.", "mlp.fc1.", name)
        name = re.sub(r"functional\.layers\.dense_1\.", "mlp.fc2.", name)
    else:
        name = re.sub(r"functional\.layers\.sequential\.layers\.dense\.", "mlp.fc1.", name)
        name = re.sub(r"functional\.layers\.sequential\.layers\.dense_1\.", "mlp.fc2.", name)
    name = re.sub(r"layers\.sequential\.layers\.conv1d\.", "conv1.", name)
    name = re.sub(r"layers\.sequential\.layers\.conv1d_1\.", "conv2.", name)
    name = re.sub(r"layers\.sequential\.layers\.conv1d_2\.", "conv3.", name)
    name = re.sub(r"layers\.sequential\.layers\.group_normalization\.", "groupnorm.", name)
    name = re.sub(r"mha_with_rope\.key_dense", "self_attn.k_proj", name)
    name = re.sub(r"mha_with_rope\.query_dense", "self_attn.q_proj", name)
    name = re.sub(r"mha_with_rope\.value_dense", "self_attn.v_proj", name)
    name = re.sub(r"mha_with_rope\.output_dense", "self_attn.o_proj", name)
    name = re.sub(r"mha_precomputed_kv\.key_dense", "encoder_attn.k_proj", name)
    name = re.sub(r"mha_precomputed_kv\.query_dense", "encoder_attn.q_proj", name)
    name = re.sub(r"mha_precomputed_kv\.value_dense", "encoder_attn.v_proj", name)
    name = re.sub(r"mha_precomputed_kv\.output_dense", "encoder_attn.o_proj", name)
    name = re.sub(r"mha_causal_with_rope\.key_dense", "self_attn.k_proj", name)
    name = re.sub(r"mha_causal_with_rope\.query_dense", "self_attn.q_proj", name)
    name = re.sub(r"mha_causal_with_rope\.value_dense", "self_attn.v_proj", name)
    name = re.sub(r"mha_causal_with_rope\.output_dense", "self_attn.o_proj", name)
    name = re.sub(r"layer_normalization\.", "input_layernorm.", name)
    name = re.sub(r"layer_normalization_1\.", "post_attention_layernorm.", name)
    name = re.sub(r"layer_normalization_2\.", "final_layernorm.", name)
    name = re.sub(r"vars\.0", "weight", name)
    name = re.sub(r"vars\.1", "bias", name)
    name = re.sub(r"layers\.reversible_embedding", "embed_tokens", name)

    return name


def _convert_weights(weights, encoder=True):
    if "layers.rotary_embedding.vars.0" in weights:
        weights.pop("layers.rotary_embedding.vars.0")

    converted_weights = {}
    if encoder:
        converted_weights["layer_norm.weight"] = weights.pop("layers.layer_normalization.vars.0")
    else:
        converted_weights["norm.weight"] = weights.pop("layers.layer_normalization.vars.0")

    for name, w in weights.items():
        if encoder:
            new_name = _convert_layer_names(name)
        else:
            new_name = _convert_layer_names(name, gated_mlp=True)
        converted_weights[new_name] = w

    return converted_weights


def convert_usefulsensors_moonshine_to_hf(model_name, pytorch_dump_folder_path):
    preprocessor_weights_path, encoder_weights_path, decoder_weights_path = _get_weights(model_name)

    with h5py.File(preprocessor_weights_path, "r") as f:
        loaded_preprocessor_weights = _read_h5_weights(f, weights={})

    with h5py.File(encoder_weights_path, "r") as f:
        loaded_encoder_weights = _read_h5_weights(f, weights={})

    with h5py.File(decoder_weights_path, "r") as f:
        loaded_decoder_weights = _read_h5_weights(f, weights={})

    encoder_state_dict = {**loaded_encoder_weights, **loaded_preprocessor_weights}
    converted_encoder_state_dict = _convert_weights(encoder_state_dict)

    converted_decoder_state_dict = _convert_weights(loaded_decoder_weights, encoder=False)
    converted_decoder_state_dict["embed_tokens.weight"] = converted_decoder_state_dict["embed_tokens.weight"].T

    final_weights = {}
    for k, v in converted_encoder_state_dict.items():
        final_weights[f"model.encoder.{k}"] = v

    for k, v in converted_decoder_state_dict.items():
        final_weights[f"model.decoder.{k}"] = v

    if model_name == "tiny":
        config = MoonshineConfig()
    elif model_name == "base":
        config = MoonshineConfig(
            hidden_size=416,
            intermediate_size=1664,
            encoder_num_hidden_layers=8,
            decoder_num_hidden_layers=8,
            encoder_num_attention_heads=8,
            decoder_num_attention_heads=8,
            partial_rotary_factor=0.62,
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")

    final_weights["proj_out.weight"] = converted_decoder_state_dict["embed_tokens.weight"]

    model = MoonshineForConditionalGeneration(config)
    model.load_state_dict(final_weights)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--model_name", type=str, help="Path to the downloaded checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    convert_usefulsensors_moonshine_to_hf(args.model_name, args.pytorch_dump_folder_path)