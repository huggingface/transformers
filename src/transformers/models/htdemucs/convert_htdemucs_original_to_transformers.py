# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Wav2Vec2 checkpoint."""


import argparse
import re

import torch
from demucs.pretrained import get_model

from transformers import logging
from transformers.models.htdemucs.configuration_htdemucs import HtdemucsConfig
from transformers.models.htdemucs.modeling_htdemucs import HtdemucsModel


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "freq_emb.embedding": "freq_embedding.embedding",
    "channel_downsampler": "freq_downsampler",
    "channel_upsampler": "freq_upsampler",
    "channel_downsampler_t": "temp_downsampler",
    "channel_upsampler_t": "temp_upsampler",
    "encoder.*.conv": "freq_encoder.*.conv_in",
    "encoder.*.rewrite": "freq_encoder.*.conv_out",
    "encoder.*.dconv": "freq_encoder.*.residual_conv",
    "tencoder.*.conv": "temp_encoder.*.conv_in",
    "tencoder.*.rewrite": "temp_encoder.*.conv_out",
    "tencoder.*.dconv": "temp_encoder.*.residual_conv",
    "decoder.*.conv_tr": "freq_decoder.*.conv_out",
    "decoder.*.rewrite": "freq_decoder.*.conv_in",
    "decoder.*.dconv": "freq_decoder.*.residual_conv",
    "tdecoder.*.conv_tr": "temp_decoder.*.conv_out",
    "tdecoder.*.rewrite": "temp_decoder.*.conv_in",
    "tdecoder.*.dconv": "temp_decoder.*.residual_conv",
    "crosstransformer.norm_in_t": "transformer.temp_layernorm_embedding",
    "crosstransformer.norm_in": "transformer.freq_layernorm_embedding",
    "crosstransformer.layers.*.self_attn.q_proj": "transformer.layers.*.freq_attn.attn.q_proj",
    "crosstransformer.layers.*.self_attn.k_proj": "transformer.layers.*.freq_attn.attn.k_proj",
    "crosstransformer.layers.*.self_attn.v_proj": "transformer.layers.*.freq_attn.attn.v_proj",
    "crosstransformer.layers.*.self_attn.out_proj": "transformer.layers.*.freq_attn.attn.out_proj",
    "crosstransformer.layers.*.cross_attn.q_proj": "transformer.layers.*.freq_attn.attn.q_proj",
    "crosstransformer.layers.*.cross_attn.k_proj": "transformer.layers.*.freq_attn.attn.k_proj",
    "crosstransformer.layers.*.cross_attn.v_proj": "transformer.layers.*.freq_attn.attn.v_proj",
    "crosstransformer.layers.*.cross_attn.out_proj": "transformer.layers.*.freq_attn.attn.out_proj",
    "crosstransformer.layers.*.gamma_1.scale": "transformer.layers.*.freq_attn.layer_scale_1",
    "crosstransformer.layers.*.gamma_2.scale": "transformer.layers.*.freq_attn.layer_scale_2",
    "crosstransformer.layers.*.linear1": "transformer.layers.*.freq_attn.fc1",
    "crosstransformer.layers.*.linear2": "transformer.layers.*.freq_attn.fc2",
    "crosstransformer.layers.*.norm1": "transformer.layers.*.freq_attn.attn_layer_norm",
    "crosstransformer.layers.*.norm2": [
        "transformer.layers.*.freq_attn.final_layer_norm",
        "transformer.layers.*.freq_attn.cross_attn_layer_norm",
    ],
    "crosstransformer.layers.*.norm3": "transformer.layers.*.freq_attn.final_layer_norm",
    "crosstransformer.layers.*.norm_out": "transformer.layers.*.freq_attn.group_norm",
    "crosstransformer.layers_t.*.self_attn.q_proj": "transformer.layers.*.temp_attn.attn.q_proj",
    "crosstransformer.layers_t.*.self_attn.k_proj": "transformer.layers.*.temp_attn.attn.k_proj",
    "crosstransformer.layers_t.*.self_attn.v_proj": "transformer.layers.*.temp_attn.attn.v_proj",
    "crosstransformer.layers_t.*.self_attn.out_proj": "transformer.layers.*.temp_attn.attn.out_proj",
    "crosstransformer.layers_t.*.cross_attn.q_proj": "transformer.layers.*.temp_attn.attn.q_proj",
    "crosstransformer.layers_t.*.cross_attn.k_proj": "transformer.layers.*.temp_attn.attn.k_proj",
    "crosstransformer.layers_t.*.cross_attn.v_proj": "transformer.layers.*.temp_attn.attn.v_proj",
    "crosstransformer.layers_t.*.cross_attn.out_proj": "transformer.layers.*.temp_attn.attn.out_proj",
    "crosstransformer.layers_t.*.gamma_1.scale": "transformer.layers.*.temp_attn.layer_scale_1",
    "crosstransformer.layers_t.*.gamma_2.scale": "transformer.layers.*.temp_attn.layer_scale_2",
    "crosstransformer.layers_t.*.linear1": "transformer.layers.*.temp_attn.fc1",
    "crosstransformer.layers_t.*.linear2": "transformer.layers.*.temp_attn.fc2",
    "crosstransformer.layers_t.*.norm1": "transformer.layers.*.temp_attn.attn_layer_norm",
    "crosstransformer.layers_t.*.norm2": [
        "transformer.layers.*.temp_attn.final_layer_norm",
        "transformer.layers.*.temp_attn.cross_attn_layer_norm",
    ],
    "crosstransformer.layers_t.*.norm3": "transformer.layers.*.temp_attn.final_layer_norm",
    "crosstransformer.layers_t.*.norm_out": "transformer.layers.*.temp_attn.group_norm",
}

RESIDUAL_CONV_MAPPING = {
    ".layers.*.0": ".conv_in.*",
    ".layers.*.1": ".norm_in.*",
    ".layers.*.3": ".conv_out.*",
    ".layers.*.4": ".norm_out.*",
    ".layers.*.6": ".layer_scales.*",
}

FUSED_PROJECTION_MAPPING = {
    "in_proj_weight": ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
    "in_proj_bias": ["q_proj.bias", "k_proj.bias", "v_proj.bias"],
}


def unfuse_projections(state_dict):
    state_dict = dict(state_dict)
    keys = list(state_dict.keys())
    for key in keys:
        for fused_key, unfused_keys in FUSED_PROJECTION_MAPPING.items():
            if fused_key in key:
                val = state_dict.pop(key)
                hidden_size = val.shape[0] // len(unfused_keys)
                for idx, replacement_key in enumerate(unfused_keys):
                    state_dict[key.replace(fused_key, replacement_key)] = val[
                        idx * hidden_size : (idx + 1) * hidden_size, ...
                    ]
    return state_dict


def set_recursively(key, value, full_name, weight_type, hf_pointer):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_param_name = None
    if weight_type is not None and weight_type != "param":
        hf_shape = getattr(hf_pointer, weight_type).shape
    elif weight_type is not None and weight_type == "param":
        shape_pointer = hf_pointer
        for attribute in hf_param_name.split("."):
            shape_pointer = getattr(shape_pointer, attribute)
        hf_shape = shape_pointer.shape

        # let's reduce dimension
        value = value[0]
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "param":
        for attribute in hf_param_name.split("."):
            hf_pointer = getattr(hf_pointer, attribute)
        hf_pointer.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def load_demucs_layer(name, value, hf_model=None):
    is_used = False
    prefix = name.split(".")[0]
    for key, mapped_key in MAPPING.items():
        if "*" in key:
            layer_index = re.findall(r"\d+", name)
            if layer_index:
                layer_index = layer_index[0]
                key = key.replace("*", layer_index)
                if isinstance(mapped_key, list):
                    mapped_key = mapped_key[int(layer_index) % 2]
                mapped_key = mapped_key.replace("*", layer_index)
        if "dconv" in name:
            for residual_key, mapped_residual_key in RESIDUAL_CONV_MAPPING.items():
                if "*" in residual_key:
                    layer_index = name.split(".")[4]
                    residual_key = residual_key.replace("*", layer_index)
                    mapped_residual_key = mapped_residual_key.replace("*", layer_index)
                if residual_key in name:
                    mapped_key += mapped_residual_key
        if key in name and key.split(".")[0] == prefix:
            is_used = True
            if "weight_g" in name:
                weight_type = "weight_g"
            elif "weight_v" in name:
                weight_type = "weight_v"
            elif "bias" in name:
                weight_type = "bias"
            elif "weight" in name:
                weight_type = "weight"
            else:
                weight_type = None
            set_recursively(mapped_key, value, name, weight_type, hf_model)
            return is_used
    return is_used


def recursively_load_weights(model, hf_model):
    unused_weights = []
    demucs_dict = model.models[0].state_dict()
    demucs_dict = unfuse_projections(demucs_dict)

    for name, value in demucs_dict.items():
        is_used = load_demucs_layer(name, value, hf_model)
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


@torch.no_grad()
def convert_demucs_checkpoint(checkpoint, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = HtdemucsConfig.from_pretrained(config_path)
    else:
        config = HtdemucsConfig()

    model = get_model(checkpoint)
    hf_demucs = HtdemucsModel(config)

    recursively_load_weights(model, hf_demucs)
    # hack to save tied weights from the k/q/v proj
    hf_demucs.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)
    hf_demucs = hf_demucs.from_pretrained(pytorch_dump_folder_path, use_safetensors=False)
    hf_demucs.save_pretrained(pytorch_dump_folder_path, safe_serialization=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default="htdemucs", type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_demucs_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
    )
