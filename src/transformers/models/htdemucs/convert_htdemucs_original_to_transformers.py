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
import json
import os

import torch
from demucs.pretrained import get_model

from transformers.models.htdemucs.modeling_htdemucs import HtdemucsModel
from transformers.models.htdemucs.configuration_htdemucs import HtdemucsConfig
from transformers import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "channel_downsampler": "freq_downsampler",
    "channel_upsampler": "freq_upsampler",
    "channel_downsampler_t": "temp_downsampler",
    "channel_upsampler_t": "temp_upsampler",
    "encoder.*.conv": "freq_encoder.*.conv_in",
    "encoder.*.rewrite": "freq_encoder.*.conv_out",
    "encoder.*.dconv": "temp_encoder.*.residual_conv",
    "tencoder.*.conv": "temp_encoder.*.conv_in",
    "tencoder.*.rewrite": "temp_encoder.*.conv_out",
    "tencoder.*.dconv": "temp_encoder.*.residual_conv",
    "decoder.*.conv_tr": "freq_decoder.*.conv_out",
    "decoder.*.rewrite": "freq_decoder.*.conv_in",
    "decoder.*.dconv": "freq_decoder.*.residual_conv",
    "tdecoder.*.conv_tr": "temp_decoder.*.conv_out",
    "tdecoder.*.rewrite": "temp_decoder.*.conv_in",
    "tdecoder.*.dconv": "temp_decoder.*.residual_conv",
    "crosstransformer.norm_in": "transformer.freq_layernorm_embedding",
    "crosstransformer.norm_in_t": "transformer.temp_layernorm_embedding",
}

RESIDUAL_CONV_MAPPING = {
    ".layers.*.0": ".conv_in.*",
    ".layers.*.1": ".norm_in.*",
    ".layers.*.3": ".conv_out.*",
    ".layers.*.4": ".norm_out.*",
    ".layers.*.6": ".layer_scales.*"
}

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
            layer_index = name.split(".")[1]
            key = key.replace("*", layer_index)
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
    hf_demucs.save_pretrained(pytorch_dump_folder_path)


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
