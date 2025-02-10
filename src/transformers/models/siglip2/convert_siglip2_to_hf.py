# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Convert Siglip2 checkpoints from the original repository.

URL: https://github.com/google-research/big_vision/tree/main
"""
import re
import argparse
import collections
import os

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw

from transformers import GemmaTokenizerFast, Siglip2Config, Siglip2ImageProcessor, Siglip2Model, Siglip2Processor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


config_options = {
    "base": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    "large": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
}

model_name_to_checkpoint = {
    # base checkpoints
    "siglip2-base-patch-16-naflex-256": "./checkpoints/siglip2/siglip2_b16_naflex.npz",
}

# fmt: off
expected_outputs = {
    "siglip2-base-patch-16-naflex-256": torch.tensor([
        [  1.0775,   0.0974,  -1.7726],
        [ -4.3421,  -6.1043,  -2.1243],
        [  4.1455,   4.8611,   3.1851],
        [  9.3390,  10.0336,   6.0143],
        [  2.3163,   2.9762,   4.0904],
        [-12.1292, -13.6398, -14.2740],
        [  1.0461,   1.0337,  -2.6771]
    ]),
}
# fmt: on

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision embeddings
    r"params/img/embedding/kernel":                                                             r"vision_model.embeddings.patch_embedding.weight",
    r"params/img/embedding/bias":                                                               r"vision_model.embeddings.patch_embedding.bias",
    r"params/img/pos_embedding":                                                                r"vision_model.embeddings.position_embedding.weight",
    # Vision encoder
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_0/scale":                             r"vision_model.encoder.layers.\1.layer_norm1.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_0/bias":                              r"vision_model.encoder.layers.\1.layer_norm1.bias",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_1/scale":                             r"vision_model.encoder.layers.\1.layer_norm2.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_1/bias":                              r"vision_model.encoder.layers.\1.layer_norm2.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_0/kernel":                     r"vision_model.encoder.layers.\1.mlp.fc1.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_0/bias":                       r"vision_model.encoder.layers.\1.mlp.fc1.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_1/kernel":                     r"vision_model.encoder.layers.\1.mlp.fc2.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_1/bias":                       r"vision_model.encoder.layers.\1.mlp.fc2.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/key/kernel":     r"vision_model.encoder.layers.\1.self_attn.k_proj.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/key/bias":       r"vision_model.encoder.layers.\1.self_attn.k_proj.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/value/kernel":   r"vision_model.encoder.layers.\1.self_attn.v_proj.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/value/bias":     r"vision_model.encoder.layers.\1.self_attn.v_proj.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/query/kernel":   r"vision_model.encoder.layers.\1.self_attn.q_proj.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/query/bias":     r"vision_model.encoder.layers.\1.self_attn.q_proj.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/out/kernel":     r"vision_model.encoder.layers.\1.self_attn.out_proj.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/out/bias":       r"vision_model.encoder.layers.\1.self_attn.out_proj.bias",
    # Vision norm
    r"params/img/Transformer/encoder_norm/scale":                                               r"vision_model.post_layernorm.weight",
    r"params/img/Transformer/encoder_norm/bias":                                                r"vision_model.post_layernorm.bias",
    # Vision head
    r"params/img/MAPHead_0/probe":                                                              r"vision_model.head.probe",
    r"params/img/MAPHead_0/LayerNorm_0/scale":                                                  r"vision_model.head.layernorm.weight",
    r"params/img/MAPHead_0/LayerNorm_0/bias":                                                   r"vision_model.head.layernorm.bias",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel":                                          r"vision_model.head.mlp.fc1.weight",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_0/bias":                                            r"vision_model.head.mlp.fc1.bias",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel":                                          r"vision_model.head.mlp.fc2.weight",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_1/bias":                                            r"vision_model.head.mlp.fc2.bias",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel":                          r"vision_model.head.attention.out_proj.weight",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias":                            r"vision_model.head.attention.out_proj.bias",
    # Text embeddings
    r"params/txt/Embed_0/embedding":                                                            r"text_model.embeddings.token_embedding.weight",
    r"params/txt/pos_embedding":                                                                r"text_model.embeddings.position_embedding.weight",
    # Text encoder
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_0/scale":                               r"text_model.encoder.layers.\1.layer_norm1.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_0/bias":                                r"text_model.encoder.layers.\1.layer_norm1.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_1/scale":                               r"text_model.encoder.layers.\1.layer_norm2.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_1/bias":                                r"text_model.encoder.layers.\1.layer_norm2.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_0/kernel":                       r"text_model.encoder.layers.\1.mlp.fc1.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_0/bias":                         r"text_model.encoder.layers.\1.mlp.fc1.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_1/kernel":                       r"text_model.encoder.layers.\1.mlp.fc2.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_1/bias":                         r"text_model.encoder.layers.\1.mlp.fc2.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/key/kernel":       r"text_model.encoder.layers.\1.self_attn.k_proj.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/key/bias":         r"text_model.encoder.layers.\1.self_attn.k_proj.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/value/kernel":     r"text_model.encoder.layers.\1.self_attn.v_proj.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/value/bias":       r"text_model.encoder.layers.\1.self_attn.v_proj.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/query/kernel":     r"text_model.encoder.layers.\1.self_attn.q_proj.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/query/bias":       r"text_model.encoder.layers.\1.self_attn.q_proj.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/out/kernel":       r"text_model.encoder.layers.\1.self_attn.out_proj.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/out/bias":         r"text_model.encoder.layers.\1.self_attn.out_proj.bias",
    # Text encoder norm and head
    r"params/txt/Encoder_0/encoder_norm/scale":                                                 r"text_model.final_layer_norm.weight",
    r"params/txt/Encoder_0/encoder_norm/bias":                                                  r"text_model.final_layer_norm.bias",
    r"params/txt/head/kernel":                                                                  r"text_model.head.weight",
    r"params/txt/head/bias":                                                                    r"text_model.head.bias",
    # learned temperature and bias
    r"params/t":                                                                                r"logit_scale",
    r"params/b":                                                                                r"logit_bias",
}
# fmt: on


def get_siglip2_config(model_name: str) -> Siglip2Config:
    _, variant, _, patch_size, _, num_patches = model_name.split("-")
    patch_size = int(patch_size)
    num_patches = int(num_patches)

    common_options = config_options[variant]
    vision_config = {
        "patch_size": patch_size,
        "num_patches": num_patches,
        **common_options,
    }
    text_config = {
        "vocab_size": 256_000,
        **common_options,
    }
    config = Siglip2Config(
        vision_config=vision_config,
        text_config=text_config,
    )
    return config


def get_siglip2_tokenizer() -> GemmaTokenizerFast:
    # Load pretrained tokenizer
    gemma_checkpoint = "google/gemma-7b"
    tokenizer = GemmaTokenizerFast.from_pretrained(
        gemma_checkpoint,
        add_bos_token=False,
        add_eos_token=True,
        padding_side="right",
        # important: make tokenizer NOT return attention_mask since original one doesn't require it
        model_input_names=["input_ids"],
    )
    return tokenizer


def get_siglip2_image_processor(patch_size: int, max_num_patches: int) -> Siglip2ImageProcessor:
    image_processor = Siglip2ImageProcessor(
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_factor=1 / 255,
    )
    return image_processor


def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # vision encoder

    rename_keys.append(("params/img/embedding/kernel", "vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("params/img/embedding/bias", "vision_model.embeddings.patch_embedding.bias"))
    rename_keys.append(("params/img/pos_embedding", "vision_model.embeddings.position_embedding.weight"))

    for i in range(config.vision_config.num_hidden_layers):
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/scale", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/scale", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("params/img/Transformer/encoder_norm/scale", "vision_model.post_layernorm.weight"))
    rename_keys.append(("params/img/Transformer/encoder_norm/bias", "vision_model.post_layernorm.bias"))

    rename_keys.append(("params/img/MAPHead_0/probe", "vision_model.head.probe"))
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/scale", "vision_model.head.layernorm.weight"))
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/bias", "vision_model.head.layernorm.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel", "vision_model.head.mlp.fc1.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/bias", "vision_model.head.mlp.fc1.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel", "vision_model.head.mlp.fc2.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/bias", "vision_model.head.mlp.fc2.bias"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel", "vision_model.head.attention.out_proj.weight"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias", "vision_model.head.attention.out_proj.bias"))

    # text encoder

    rename_keys.append(("params/txt/Embed_0/embedding", "text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("params/txt/pos_embedding", "text_model.embeddings.position_embedding.weight"))

    for i in range(config.text_config.num_hidden_layers):
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/scale", f"text_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/bias", f"text_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/scale", f"text_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/bias", f"text_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"text_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"text_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"text_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"text_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"text_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"text_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("params/txt/Encoder_0/encoder_norm/scale", "text_model.final_layer_norm.weight"))
    rename_keys.append(("params/txt/Encoder_0/encoder_norm/bias", "text_model.final_layer_norm.bias"))
    rename_keys.append(("params/txt/head/kernel", "text_model.head.weight"))
    rename_keys.append(("params/txt/head/bias", "text_model.head.bias"))

    # learned temperature and bias
    rename_keys.append(("params/t", "logit_scale"))
    rename_keys.append(("params/b", "logit_bias"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new, config):
    val = dct.pop(old)

    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if "patch_embedding.weight" in new:
        val = val.T  # transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if new.endswith("bias"):
        val = val.reshape(-1)

    dct[new] = torch.from_numpy(val)


def split_to_layers(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        if "/encoderblock/" in key:
            weight = state_dict.pop(key)
            for i, weight_i in enumerate(weight):
                new_name = key.replace("encoderblock", f"encoderblock_{i}")
                state_dict[new_name] = weight_i
    return state_dict


def read_in_q_k_v_head(state_dict, config):
    # read in individual input projection layers
    key_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    key_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/bias").reshape(-1)
    value_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    value_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/bias").reshape(-1)
    query_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    query_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/bias").reshape(-1)

    # next, add them to the state dict as a single matrix + vector
    state_dict["vision_model.head.attention.in_proj_weight"] = torch.from_numpy(
        np.concatenate([query_proj_weight, key_proj_weight, value_proj_weight], axis=0)
    )
    state_dict["vision_model.head.attention.in_proj_bias"] = torch.from_numpy(
        np.concatenate([query_proj_bias, key_proj_bias, value_proj_bias], axis=0)
    )


def create_image(width, height):
    image = Image.new('RGB', (width, height), color='red')
    draw = ImageDraw.Draw(image)
    center_x = image.width // 2
    center_y = image.height // 2
    radius = min(center_x, center_y) // 8 * 7
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill='blue',
        outline='green',
        width=image.width // 20,
    )
    return image


def prepare_inputs():
    text = [
        'circle',
        'ellipsoid',
        'blue circle on red background',
        'blue circle with green border on red background',
        'green circle on red background',
        'a dog',
        'a blue dog with a green border on a red background',
    ]
    img224 = create_image(224, 224)
    img1024 = create_image(1024, 1024)
    img224_1024 = create_image(1024, 224)

    images = [img224, img1024, img224_1024]
    return text, images


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_old_keys_to_new_keys(state_dict_keys: list):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


@torch.no_grad()
def convert_siglip2_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Siglip2 structure.
    """

    # Define Siglip2 configuration
    config = get_siglip2_config(model_name)

    # --------------------------------------------------------------------------------------------
    # Convert model
    # --------------------------------------------------------------------------------------------

    checkpoint = model_name_to_checkpoint[model_name]

    print(f"Loading checkpoint from {checkpoint}...")
    data = np.load(checkpoint)
    state_dict = flatten_nested_dict(data)
    state_dict = split_to_layers(state_dict)

    # Rename and transform weights
    print("Renaming and transforming weights...")

    original_keys = list(state_dict.keys())
    hf_keys = convert_old_keys_to_new_keys(original_keys)

    new_state_dict = {}
    for original_key in original_keys:
        new_key = hf_keys[original_key]
        parameter = state_dict[original_key] # change to pop
        
        if any(layer_name in new_key for layer_name in ("out_proj", "q_proj", "k_proj", "v_proj")):
            if "vision" in new_key:
                parameter = parameter.reshape(-1, config.vision_config.hidden_size)
            elif "text" in new_key:
                parameter = parameter.reshape(-1, config.text_config.hidden_size)
        if "patch_embedding.weight" in new_key:
            parameter = parameter.T
        elif new_key.endswith("weight") and "position_embedding" not in new_key and "token_embedding" not in new_key:
            parameter = parameter.T
        if "position_embedding" in new_key and "vision" in new_key:
            parameter = parameter.reshape(-1, config.vision_config.hidden_size)
        if "position_embedding" in new_key and "text" in new_key:
            parameter = parameter.reshape(-1, config.text_config.hidden_size)
        if new_key.endswith("bias"):
            parameter = parameter.reshape(-1)

        new_state_dict[new_key] = torch.from_numpy(parameter)

    state_dict = new_state_dict
    # rename_keys = create_rename_keys(config)
    # for src, dest in rename_keys:
    #     rename_key(state_dict, src, dest, config)

    # for key, old_param in state_dict.items():
    #     new_param = new_state_dict[key]
    #     old_param = torch.tensor(old_param)
    #     new_param = torch.tensor(new_param)
    #     if old_param.shape != new_param.shape:
    #         print(f"Shape mismatch for {key}: {old_param.shape} != {new_param.shape}")
    #     elif not torch.allclose(old_param, new_param):
    #         print(f"Value mismatch for {key}: {old_param} != {new_param}")

    # raise
    # qkv matrices of attention pooling head need special treatment
    read_in_q_k_v_head(state_dict, config)

    # load HuggingFace model
    print("Loading HuggingFace model...")
    model = Siglip2Model(config).eval()
    model.load_state_dict(state_dict)

    # Create processor
    print("Creating processor...")
    # TODO: update with more checkpoints
    tokenizer = get_siglip2_tokenizer()
    image_processor = get_siglip2_image_processor(config.vision_config.patch_size, max_num_patches=256)
    processor = Siglip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # Verify logits
    if verify_logits:
        print(f"Verifying logits for {model_name}...")
        text, images = prepare_inputs()
        inputs = processor(text=text, images=images, padding="max_length", max_length=64, return_tensors="pt")
        outputs = model(**inputs)
        torch.testing.assert_close(outputs.logits_per_text, expected_outputs[model_name], atol=1e-3, rtol=1e-3)

    # Save model
    if pytorch_dump_folder_path is not None:
        dst_dir = os.path.join(pytorch_dump_folder_path, model_name)
        print(f"Saving model {model_name} to {dst_dir}...")
        model.save_pretrained(dst_dir)
        print(f"Saving processor to {dst_dir}...")
        processor.save_pretrained(dst_dir)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to the HuggingFace Hub...")
        model.push_to_hub(f"s0225/{model_name}", private=True)
        processor.push_to_hub(f"s0225/{model_name}", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="siglip2-base-patch-16-naflex-256",
        type=str,
        choices=model_name_to_checkpoint.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="./checkpoints/siglip2-hf/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether to verify logits against the original implementation.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_siglip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
