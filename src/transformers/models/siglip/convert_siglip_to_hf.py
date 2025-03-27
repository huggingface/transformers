# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert SigLIP checkpoints from the original repository.

URL: https://github.com/google-research/big_vision/tree/main
"""

import argparse
import collections
import os
from typing import Tuple

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image

from transformers import (
    GemmaTokenizerFast,
    SiglipConfig,
    SiglipImageProcessor,
    SiglipModel,
    SiglipProcessor,
    SiglipTokenizer,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


MODEL_CONFIGS = {
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
    "giant-opt": {
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
    },
    "so400m": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
    },
}

model_name_to_checkpoint = {
    # base checkpoints
    "siglip-base-patch16-224": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_224_63724782.npz",
    "siglip-base-patch16-256": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_256_60500360.npz",
    "siglip-base-patch16-384": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_384_68578854.npz",
    "siglip-base-patch16-512": "/Users/nielsrogge/Documents/SigLIP/webli_en_b16_512_68580893.npz",
    # large checkpoints
    "siglip-large-patch16-256": "/Users/nielsrogge/Documents/SigLIP/webli_en_l16_256_60552751.npz",
    "siglip-large-patch16-384": "/Users/nielsrogge/Documents/SigLIP/webli_en_l16_384_63634585.npz",
    # multilingual checkpoint
    "siglip-base-patch16-256-i18n": "/Users/nielsrogge/Documents/SigLIP/webli_i18n_b16_256_66117334.npz",
    # so400m checkpoints
    "siglip-so400m-patch14-384": "/Users/nielsrogge/Documents/SigLIP/webli_en_so400m_384_58765454.npz",
    # ----------------- v2 -----------------
    # base checkpoints
    "siglip2-base-patch32-256": "gv-hf/siglip2/siglip2_b32_256.npz",
    "siglip2-base-patch16-224": "gv-hf/siglip2/siglip2_b16_224.npz",
    "siglip2-base-patch16-256": "gv-hf/siglip2/siglip2_b16_256.npz",
    "siglip2-base-patch16-384": "gv-hf/siglip2/siglip2_b16_384.npz",
    "siglip2-base-patch16-512": "gv-hf/siglip2/siglip2_b16_512.npz",
    # large checkpoints
    "siglip2-large-patch16-256": "gv-hf/siglip2/siglip2_l16_256.npz",
    "siglip2-large-patch16-384": "gv-hf/siglip2/siglip2_l16_384.npz",
    "siglip2-large-patch16-512": "gv-hf/siglip2/siglip2_l16_512.npz",
    # giant opt checkpoints
    "siglip2-giant-opt-patch16-256": "gv-hf/siglip2/siglip2_g-opt16_256.npz",
    "siglip2-giant-opt-patch16-384": "gv-hf/siglip2/siglip2_g-opt16_384.npz",
    # so400m checkpoints
    "siglip2-so400m-patch14-224": "gv-hf/siglip2/siglip2_so400m14_224.npz",
    "siglip2-so400m-patch14-384": "gv-hf/siglip2/siglip2_so400m14_384.npz",
    "siglip2-so400m-patch16-256": "gv-hf/siglip2/siglip2_so400m16_256.npz",
    "siglip2-so400m-patch16-384": "gv-hf/siglip2/siglip2_so400m16_384.npz",
    "siglip2-so400m-patch16-512": "gv-hf/siglip2/siglip2_so400m16_512.npz",
}

# ------------------------------------------------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------------------------------------------------


def get_image_size_from_model_name(model_name: str) -> int:
    if "-i18n" not in model_name:
        size = model_name.split("-")[-1]
    else:
        size = model_name.split("-")[-2]
    return int(size)


def get_patch_size_from_model_name(model_name: str) -> int:
    patch_str = [x for x in model_name.split("-") if "patch" in x][0]
    return int(patch_str[-2:])


def get_vocab_size_from_model_name(model_name: str) -> int:
    if "siglip2" in model_name:
        vocab_size = 256000
    elif "-i18n" in model_name:
        vocab_size = 250000
    else:
        vocab_size = 32000
    return vocab_size


def get_vocab_file_from_model_name(model_name: str) -> str:
    # get vocab file
    if "i18n" in model_name:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/multilingual_vocab/sentencepiece.model"
    else:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/english_vocab/sentencepiece.model"
    return vocab_file


def get_text_and_vision_vit_variants(model_name: str) -> Tuple[str, str]:
    variant = model_name.split("-")[1] if "giant-opt" not in model_name else "giant-opt"
    return {
        "base": ("base", "base"),
        "large": ("large", "large"),
        "so400m": ("so400m", "so400m"),
        # g-opt siglip2 is not symmetric
        "giant-opt": ("so400m", "giant-opt"),
    }[variant]


def get_siglip_config(model_name):
    text_variant, vision_variant = get_text_and_vision_vit_variants(model_name)
    text_config = MODEL_CONFIGS[text_variant].copy()
    vision_config = MODEL_CONFIGS[vision_variant].copy()

    text_config["vocab_size"] = get_vocab_size_from_model_name(model_name)
    vision_config["image_size"] = get_image_size_from_model_name(model_name)
    vision_config["patch_size"] = get_patch_size_from_model_name(model_name)

    if text_config["hidden_size"] != vision_config["hidden_size"]:
        text_config["projection_size"] = vision_config["hidden_size"]

    return SiglipConfig(text_config=text_config, vision_config=vision_config)


# ------------------------------------------------------------------------------------------------------
#  PROCESSING
# ------------------------------------------------------------------------------------------------------


def get_tokenizer(model_name: str) -> GemmaTokenizerFast:
    if "siglip2" in model_name:
        tokenizer = GemmaTokenizerFast.from_pretrained(
            "google/gemma-2-9b-it",
            add_bos_token=False,
            add_eos_token=True,
            padding_side="right",
            do_lower_case=True,
            # important: make tokenizer NOT return attention_mask since original one doesn't require it
            model_input_names=["input_ids"],
        )
    else:
        # for siglip v1
        vocab_file = get_vocab_file_from_model_name(model_name)
        # important: make tokenizer not return attention_mask since original one doesn't require it
        tokenizer = SiglipTokenizer(vocab_file=vocab_file, model_input_names=["input_ids"])
    return tokenizer


def get_image_processor(model_name: str) -> SiglipImageProcessor:
    image_size = get_image_size_from_model_name(model_name)
    size = {"height": image_size, "width": image_size}
    if "siglip2" in model_name:
        image_processor = SiglipImageProcessor(size=size, resample=2)  # bilinear resampling
    else:
        image_processor = SiglipImageProcessor(size=size)
    return image_processor


# ------------------------------------------------------------------------------------------------------
#  CONVERT FUNCTIONS
# ------------------------------------------------------------------------------------------------------


def split_encoderblock_layers(state_dict: dict) -> dict:
    """
    Split the encoderblock weight into layers. In some cases they are concatenated in
    the original checkpoints.
    """
    # Make shallow copy
    state_dict = state_dict.copy()
    # Split encoderblock weight into layers
    keys = list(state_dict.keys())
    for key in keys:
        if "/encoderblock/" in key:
            weight = state_dict.pop(key)
            for i, weight_i in enumerate(weight):
                new_name = key.replace("encoderblock", f"encoderblock_{i}")
                state_dict[new_name] = weight_i
    return state_dict


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
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if new.endswith("bias"):
        val = val.reshape(-1)

    dct[new] = torch.from_numpy(val)


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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@torch.no_grad()
def convert_siglip_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """

    # Define default SigLIP configuration
    config = get_siglip_config(model_name)

    # Get checkpoint
    checkpoint = model_name_to_checkpoint[model_name]
    if not os.path.exists(checkpoint):
        org, repo_id, *filepath = checkpoint.split("/")
        checkpoint = hf_hub_download(repo_id=f"{org}/{repo_id}", filename="/".join(filepath))

    # Load original state dict
    data = load(checkpoint)
    state_dict = flatten_nested_dict(data)
    state_dict = split_encoderblock_layers(state_dict)

    # Remove and rename some keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest, config)

    # qkv matrices of attention pooling head need special treatment
    read_in_q_k_v_head(state_dict, config)

    # Load HuggingFace model
    model = SiglipModel(config).eval()
    model.load_state_dict(state_dict)

    # Create processor
    image_processor = get_image_processor(model_name)
    tokenizer = get_tokenizer(model_name)
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Verify forward pass on dummy images and texts
    url_1 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
    image_1 = Image.open(requests.get(url_1, stream=True).raw).convert("RGB")
    url_2 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
    image_2 = Image.open(requests.get(url_2, stream=True).raw).convert("RGB")
    texts = ["an apple", "a picture of an apple"]

    inputs = processor(images=[image_1, image_2], text=texts, padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    if verify_logits:
        image_size = config.vision_config.image_size

        # verify input_ids against original ones
        if image_size == 224:
            filename = "siglip_pixel_values.pt"
        elif image_size == 256:
            filename = "siglip_pixel_values_256.pt"
        elif image_size == 384:
            filename = "siglip_pixel_values_384.pt"
        elif image_size == 512:
            filename = "siglip_pixel_values_512.pt"
        else:
            raise ValueError("Image size not supported")

        filepath = hf_hub_download(repo_id="nielsr/test-image", filename=filename, repo_type="dataset")
        original_pixel_values = torch.load(filepath)
        filepath = hf_hub_download(repo_id="nielsr/test-image", filename="siglip_input_ids.pt", repo_type="dataset")
        original_input_ids = torch.load(filepath)

        if "i18n" not in model_name:
            assert inputs.input_ids.tolist() == original_input_ids.tolist()

        print("Mean of original pixel values:", original_pixel_values.mean())
        print("Mean of new pixel values:", inputs.pixel_values.mean())

        # note: we're testing with original pixel values here since we don't have exact pixel values
        with torch.no_grad():
            outputs = model(input_ids=original_input_ids, pixel_values=original_pixel_values)
        print(outputs.logits_per_image[:3, :3])

        probs = torch.sigmoid(outputs.logits_per_image)  # these are the probabilities
        print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")

        if model_name == "siglip-base-patch16-224":
            expected_slice = torch.tensor(
                [[-2.9621, -2.1672], [-0.2713, 0.2910]],
            )
        elif model_name == "siglip-base-patch16-256":
            expected_slice = torch.tensor(
                [[-3.1146, -1.9894], [-0.7312, 0.6387]],
            )
        elif model_name == "siglip-base-patch16-384":
            expected_slice = torch.tensor(
                [[-2.8098, -2.1891], [-0.4242, 0.4102]],
            )
        elif model_name == "siglip-base-patch16-512":
            expected_slice = torch.tensor(
                [[-2.7899, -2.2668], [-0.4295, -0.0735]],
            )
        elif model_name == "siglip-large-patch16-256":
            expected_slice = torch.tensor(
                [[-1.5827, -0.5801], [-0.9153, 0.1363]],
            )
        elif model_name == "siglip-large-patch16-384":
            expected_slice = torch.tensor(
                [[-2.1523, -0.2899], [-0.2959, 0.7884]],
            )
        elif model_name == "siglip-so400m-patch14-384":
            expected_slice = torch.tensor([[-1.2441, -0.6649], [-0.7060, 0.7374]])
        elif model_name == "siglip-base-patch16-256-i18n":
            expected_slice = torch.tensor(
                [[-0.9064, 0.1073], [-0.0299, 0.5304]],
            )

        assert torch.allclose(outputs.logits_per_image[:3, :3], expected_slice, atol=1e-4)
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        pytorch_dump_folder_path = os.path.join(pytorch_dump_folder_path, model_name)
        os.makedirs(pytorch_dump_folder_path, exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"s0225/{model_name}", private=True)
        processor.push_to_hub(f"s0225/{model_name}", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="siglip-base-patch16-224",
        type=str,
        choices=model_name_to_checkpoint.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
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
    convert_siglip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
