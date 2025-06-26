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
"""Convert MLCD checkpoints from the original repository.

URL: https://github.com/deepglint/unicom/tree/main
"""

import argparse
import collections
import os
import re

import numpy as np
import requests
import torch
from PIL import Image

from transformers import CLIPImageProcessor

from ...utils import logging
from .configuration_mlcd import MLCDVisionConfig
from .modeling_mlcd import MLCDVisionModel


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


COMMON_CONFIG_PARAMS = {
    "mlcd-vit-bigG-patch14-336": {
        "hidden_size": 1664,
        "image_size": 336,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "patch_size": 14,
        "projection_dim": 1024,
    },
    "mlcd-vit-bigG-patch14-448": {
        "hidden_size": 1664,
        "image_size": 448,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "patch_size": 14,
        "projection_dim": 1024,
    },
}

MODEL_NAME_TO_CHECKPOINT_PATH = {
    # base checkpoints
    "mlcd-vit-bigG-patch14-336": "MLCD_ViT_bigG_14_336px_pytorch.pt",
    "mlcd-vit-bigG-patch14-448": "MLCD_ViT_bigG_14_448px_pytorch.pt",
}

# fmt: off
EXPECTED_OUTPUTS = {
    "mlcd-vit-bigG-patch14-336": torch.tensor([
        [-0.8921, -0.1069,  0.2989,  0.6018, -0.5892],
        [ 0.4093, -1.4592,  0.6048, -0.5147, -0.5929],
        [ 0.7796, -0.7133, -0.5649, -0.7843, -0.5548],
        [ 0.0041,  0.0286,  0.4310, -0.1403, -0.2399],
        [ 0.0839,  0.2152, -0.3822, -0.1668, -0.7886]
    ]),
    "mlcd-vit-bigG-patch14-448": torch.tensor([
        [-0.8978, -0.1181,  0.4769,  0.4761, -0.5779],
        [ 0.2640, -2.6150,  0.4853,  0.5743, -1.1003],
        [ 0.3314, -0.3328, -0.4305, -0.1874, -0.7701],
        [-1.5174, -1.0238, -1.1854,  0.1749, -0.8786],
        [ 0.2323, -0.8346, -0.9680, -0.2951,  0.0867],
    ]),
}
# fmt: on

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision embeddings
    r"conv1.weight":                                                r"vision_model.embeddings.patch_embedding.weight",
    r"class_embedding":                                             r"vision_model.embeddings.class_embedding",
    r"vision_rotary_embedding":                                     r"vision_model.vision_rotary_embedding",
    r"class_pos_emb":                                               r"vision_model.class_pos_emb",
    # Vision encoder
    r"transformer.resblocks_(\d+).ln_1.weight":                     r"vision_model.encoder.layers.\1.layer_norm1.weight",
    r"transformer.resblocks_(\d+).ln_1.bias":                       r"vision_model.encoder.layers.\1.layer_norm1.bias",
    r"transformer.resblocks_(\d+).ln_2.weight":                     r"vision_model.encoder.layers.\1.layer_norm2.weight",
    r"transformer.resblocks_(\d+).ln_2.bias":                       r"vision_model.encoder.layers.\1.layer_norm2.bias",
    r"transformer.resblocks_(\d+).mlp.c_fc.weight":                 r"vision_model.encoder.layers.\1.mlp.fc1.weight",
    r"transformer.resblocks_(\d+).mlp.c_fc.bias":                   r"vision_model.encoder.layers.\1.mlp.fc1.bias",
    r"transformer.resblocks_(\d+).mlp.c_proj.weight":               r"vision_model.encoder.layers.\1.mlp.fc2.weight",
    r"transformer.resblocks_(\d+).mlp.c_proj.bias":                 r"vision_model.encoder.layers.\1.mlp.fc2.bias",
    r"transformer.resblocks_(\d+).attn.(q|k|v|out)_proj.weight":    r"vision_model.encoder.layers.\1.self_attn.\2_proj.weight",
    r"transformer.resblocks_(\d+).attn.(q|k|v|out)_proj.bias":      r"vision_model.encoder.layers.\1.self_attn.\2_proj.bias",
    # Vision norm
    r"ln_post.weight":                                              r"vision_model.post_layernorm.weight",
    r"ln_post.bias":                                                r"vision_model.post_layernorm.bias",
    r"ln_pre.weight":                                               r"vision_model.pre_layernorm.weight",
    r"ln_pre.bias":                                                 r"vision_model.pre_layernorm.bias",
}
# fmt: on


# --------------------------------------------------------------------------------------------
# Model objects: configuration, image processor
# --------------------------------------------------------------------------------------------


def get_mlcd_config(model_name: str) -> MLCDVisionConfig:
    """
    Create a configuration for the MLCD model based on the model name.
    """
    assert model_name in COMMON_CONFIG_PARAMS, f"Model {model_name} not found in the list of COMMON_CONFIG_PARAMS."
    config_params = COMMON_CONFIG_PARAMS[model_name]
    config = MLCDVisionConfig(
        hidden_size=config_params["hidden_size"],
        image_size=config_params["image_size"],
        intermediate_size=config_params["intermediate_size"],
        num_attention_heads=config_params["num_attention_heads"],
        num_hidden_layers=config_params["num_hidden_layers"],
        patch_size=config_params["patch_size"],
        projection_dim=config_params["projection_dim"],
    )
    return config


def get_mlcd_image_processor(model_name: str) -> CLIPImageProcessor:
    """
    Create an image processor for the MLCD model based on the model name.
    """
    assert model_name in COMMON_CONFIG_PARAMS, f"Model {model_name} not found in the list of COMMON_CONFIG_PARAMS."
    config_params = COMMON_CONFIG_PARAMS[model_name]
    image_processor = CLIPImageProcessor(
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        feature_extractor_type="CLIPFeatureExtractor",
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        resample=3,
        size=config_params["image_size"],
        crop_size=config_params["image_size"],
    )
    return image_processor


# --------------------------------------------------------------------------------------------
# Helper functions for state dict conversion
# --------------------------------------------------------------------------------------------


def flatten_nested_dict(params: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested original checkpoint dictionary into a flat dictionary.
    """
    items = []
    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def split_resblocks_layers(state_dict: dict) -> dict:
    """
    Split the resblocks weight into layers. In some cases they are concatenated in
    the original checkpoints.
    """
    # Make shallow copy
    state_dict = state_dict.copy()
    # Split resblocks weight into layers
    keys = list(state_dict.keys())
    for key in keys:
        if ".resblocks." in key:
            weight = state_dict.pop(key)
            for i, weight_i in enumerate(weight):
                new_name = key.replace("resblocks", f"resblocks_{i}")
                state_dict[new_name] = weight_i
    return state_dict


def chunk_qkv_for_attn(state_dict: dict) -> dict:
    """
    Chunk the q/k/v weights and biases for the attention layers.
    """
    # Make shallow copy
    state_dict = state_dict.copy()
    # Read and process q/k/v weights and biases
    keys = list(state_dict.keys())
    for key in keys:
        if ".in_proj." in key:
            weight = state_dict.pop(key)
            qkv_weights = weight.chunk(3, dim=0)
            for name, weight_i in zip(["q_proj", "k_proj", "v_proj"], qkv_weights):
                new_name = key.replace("in_proj", name)
                state_dict[new_name] = weight_i
    return state_dict


def convert_old_keys_to_new_keys(state_dict_keys: list) -> dict:
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


# --------------------------------------------------------------------------------------------
# Convert model
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def convert_mlcd_checkpoint(model_name, input_dir, output_dir, verify_hidden_state=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MLCD structure.
    """

    # Define MLCD configuration
    config = get_mlcd_config(model_name)

    checkpoint = MODEL_NAME_TO_CHECKPOINT_PATH[model_name]
    checkpoint_path = os.path.join(input_dir, checkpoint)
    assert os.path.exists(checkpoint_path), f"Checkpoint path ({checkpoint_path}) not found."

    # Load original checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, "cpu")

    # Flatten nested dictionary
    print("Flattening nested dictionary...")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if "positional_embedding" in state_dict:
        state_dict.pop("positional_embedding")
    state_dict = flatten_nested_dict(state_dict)
    state_dict = split_resblocks_layers(state_dict)
    state_dict = chunk_qkv_for_attn(state_dict)

    # Rename and transform weights
    print("Renaming and transforming weights...")
    original_keys = list(state_dict.keys())
    hf_keys = convert_old_keys_to_new_keys(original_keys)
    new_state_dict = {}
    for original_key in original_keys:
        new_key = hf_keys[original_key]
        parameter = state_dict.pop(original_key)
        new_state_dict[new_key] = torch.from_numpy(parameter)

    # load HuggingFace model
    print("Loading HuggingFace model...")
    model = MLCDVisionModel(config).eval()
    model.load_state_dict(new_state_dict)

    # Create processor
    print("Creating processor...")
    image_processor = get_mlcd_image_processor(model_name)

    # Verify hidden state
    if verify_hidden_state:
        print("Verifying hidden state for {model_name}...")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
        last_hidden_state = model(pixel_values, output_hidden_states=True).last_hidden_state[0, :5, :5]
        expected_hidden_state = EXPECTED_OUTPUTS[model_name]
        np.testing.assert_allclose(last_hidden_state.cpu().numpy(), expected_hidden_state.numpy(), atol=1e-4)

    # Save model
    if output_dir is not None:
        dst_dir = os.path.join(output_dir, model_name)
        print(f"Saving model {model_name} to {dst_dir}...")
        model.save_pretrained(dst_dir)
        print(f"Saving processor to {dst_dir}...")
        image_processor.save_pretrained(dst_dir)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to the HuggingFace Hub...")
        model.push_to_hub(f"deepglint-hf/{model_name}", private=True)
        image_processor.push_to_hub(f"deepglint-hf/{model_name}", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="mlcd-vit-bigG-patch14-448",
        type=str,
        choices=MODEL_NAME_TO_CHECKPOINT_PATH.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--input_dir",
        default="mlcd/original",
        help="Location of MLCD original weights",
    )
    parser.add_argument(
        "--output_dir",
        default="mlcd/checkpoint",
        help="Location to write HF model and processor",
    )
    parser.add_argument(
        "--verify_hidden_state",
        action="store_true",
        help="Whether to verify hidden_state against the original implementation.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_mlcd_checkpoint(
        args.model_name, args.input_dir, args.output_dir, args.verify_hidden_state, args.push_to_hub
    )
