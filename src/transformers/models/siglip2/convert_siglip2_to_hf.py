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

import argparse
import collections
import os
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from transformers import GemmaTokenizerFast, Siglip2Config, Siglip2ImageProcessorFast, Siglip2Model, Siglip2Processor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


COMMON_CONFIG_PARAMS = {
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
    "so400m": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
    },
}

MODEL_NAME_TO_CHECKPOINT_PATH = {
    # base checkpoints
    "siglip2-base-patch16-naflex": "gv-hf/siglip2/siglip2_b16_naflex.npz",
    "siglip2-so400m-patch16-naflex": "gv-hf/siglip2/siglip2_so400m16_naflex.npz",
}

# fmt: off
EXPECTED_OUTPUTS = {
    "siglip2-base-patch16-naflex": torch.tensor([
        [  1.0195,  -0.0280,  -1.4468],
        [ -4.5395,  -6.2269,  -1.5667],
        [  4.1757,   5.0358,   3.5159],
        [  9.4264,  10.1879,   6.3353],
        [  2.4409,   3.1058,   4.5491],
        [-12.3230, -13.7355, -13.4632],
        [  1.1520,   1.1687,  -1.9647],
    ]),
    "siglip2-so400m-patch16-naflex": torch.tensor([
        [  0.9422,   0.5540,  -2.4405],
        [ -7.3522,  -9.4931,  -6.3499],
        [  5.7852,   6.7288,   7.7893],
        [  9.9881,  10.8136,   9.2121],
        [  5.3660,   5.7746,   8.4130],
        [-12.7218, -14.2631, -13.6442],
        [  0.6384,   0.4278,  -0.9022],
    ]),
}
# fmt: on

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision embeddings
    r"params/img/embedding/kernel":                                                                         r"vision_model.embeddings.patch_embedding.weight",
    r"params/img/embedding/bias":                                                                           r"vision_model.embeddings.patch_embedding.bias",
    r"params/img/pos_embedding":                                                                            r"vision_model.embeddings.position_embedding.weight",
    # Vision encoder
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_0/scale":                                         r"vision_model.encoder.layers.\1.layer_norm1.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_0/bias":                                          r"vision_model.encoder.layers.\1.layer_norm1.bias",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_1/scale":                                         r"vision_model.encoder.layers.\1.layer_norm2.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_1/bias":                                          r"vision_model.encoder.layers.\1.layer_norm2.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_0/kernel":                                 r"vision_model.encoder.layers.\1.mlp.fc1.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_0/bias":                                   r"vision_model.encoder.layers.\1.mlp.fc1.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_1/kernel":                                 r"vision_model.encoder.layers.\1.mlp.fc2.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_1/bias":                                   r"vision_model.encoder.layers.\1.mlp.fc2.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/kernel":   r"vision_model.encoder.layers.\1.self_attn.\2_proj.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/bias":     r"vision_model.encoder.layers.\1.self_attn.\2_proj.bias",
    # Vision norm
    r"params/img/Transformer/encoder_norm/scale":                                                           r"vision_model.post_layernorm.weight",
    r"params/img/Transformer/encoder_norm/bias":                                                            r"vision_model.post_layernorm.bias",
    # Vision head
    r"params/img/MAPHead_0/probe":                                                                          r"vision_model.head.probe",
    r"params/img/MAPHead_0/LayerNorm_0/scale":                                                              r"vision_model.head.layernorm.weight",
    r"params/img/MAPHead_0/LayerNorm_0/bias":                                                               r"vision_model.head.layernorm.bias",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel":                                                      r"vision_model.head.mlp.fc1.weight",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_0/bias":                                                        r"vision_model.head.mlp.fc1.bias",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel":                                                      r"vision_model.head.mlp.fc2.weight",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_1/bias":                                                        r"vision_model.head.mlp.fc2.bias",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel":                                      r"vision_model.head.attention.out_proj.weight",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias":                                        r"vision_model.head.attention.out_proj.bias",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/qkv/kernel":                                      r"vision_model.head.attention.in_proj_weight",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/qkv/bias":                                        r"vision_model.head.attention.in_proj_bias",
    # Text embeddings
    r"params/txt/Embed_0/embedding":                                                                        r"text_model.embeddings.token_embedding.weight",
    r"params/txt/pos_embedding":                                                                            r"text_model.embeddings.position_embedding.weight",
    # Text encoder
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_0/scale":                                           r"text_model.encoder.layers.\1.layer_norm1.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_0/bias":                                            r"text_model.encoder.layers.\1.layer_norm1.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_1/scale":                                           r"text_model.encoder.layers.\1.layer_norm2.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_1/bias":                                            r"text_model.encoder.layers.\1.layer_norm2.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_0/kernel":                                   r"text_model.encoder.layers.\1.mlp.fc1.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_0/bias":                                     r"text_model.encoder.layers.\1.mlp.fc1.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_1/kernel":                                   r"text_model.encoder.layers.\1.mlp.fc2.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_1/bias":                                     r"text_model.encoder.layers.\1.mlp.fc2.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/kernel":     r"text_model.encoder.layers.\1.self_attn.\2_proj.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/bias":       r"text_model.encoder.layers.\1.self_attn.\2_proj.bias",
    # Text encoder norm and head
    r"params/txt/Encoder_0/encoder_norm/scale":                                                             r"text_model.final_layer_norm.weight",
    r"params/txt/Encoder_0/encoder_norm/bias":                                                              r"text_model.final_layer_norm.bias",
    r"params/txt/head/kernel":                                                                              r"text_model.head.weight",
    r"params/txt/head/bias":                                                                                r"text_model.head.bias",
    # learned temperature and bias
    r"params/t":                                                                                            r"logit_scale",
    r"params/b":                                                                                            r"logit_bias",
}
# fmt: on


# --------------------------------------------------------------------------------------------
# Model objects: configuration, tokenizer, image processor
# --------------------------------------------------------------------------------------------


def get_siglip2_config(model_name: str) -> Siglip2Config:
    """
    Create a configuration for the Siglip2 model based on the model name.
    """

    _, variant, patch, _ = model_name.split("-")
    patch_size = int(patch[-2:])
    num_patches = 256

    common_options = COMMON_CONFIG_PARAMS[variant]
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
        do_lower_case=True,
        # important: make tokenizer NOT return attention_mask since original one doesn't require it
        model_input_names=["input_ids"],
    )
    return tokenizer


def get_siglip2_image_processor(patch_size: int, max_num_patches: int) -> Siglip2ImageProcessorFast:
    image_processor = Siglip2ImageProcessorFast(
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_factor=1 / 255,
        resample=Image.Resampling.BILINEAR,
    )
    return image_processor


# --------------------------------------------------------------------------------------------
# Helper functions for state dict conversion
# --------------------------------------------------------------------------------------------


def flatten_nested_dict(params: dict, parent_key: str = "", sep: str = "/") -> dict:
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


def merge_qkv_for_head(state_dict: dict, config: Siglip2Config) -> dict:
    """
    Merge the q/k/v weights and biases for the attention head.
    """
    # Make shallow copy
    state_dict = state_dict.copy()
    # Read and process q/k/v weights and biases
    qkv_weights, qkv_biases = [], []
    for name in ["query", "key", "value"]:
        prefix = f"params/img/MAPHead_0/MultiHeadDotProductAttention_0/{name}"
        weight = state_dict.pop(f"{prefix}/kernel").reshape(-1, config.vision_config.hidden_size)
        bias = state_dict.pop(f"{prefix}/bias").reshape(-1)
        qkv_weights.append(weight)
        qkv_biases.append(bias)

    # Combine into single tensors
    state_dict["params/img/MAPHead_0/MultiHeadDotProductAttention_0/qkv/kernel"] = np.concatenate(qkv_weights, axis=1)
    state_dict["params/img/MAPHead_0/MultiHeadDotProductAttention_0/qkv/bias"] = np.concatenate(qkv_biases, axis=0)
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
# Helper functions for model verification
# --------------------------------------------------------------------------------------------


def create_image(width, height):
    """
    Helper function to create an image with a blue circle on a red background.
    """
    image = Image.new("RGB", (width, height), color="red")
    draw = ImageDraw.Draw(image)
    center_x = image.width // 2
    center_y = image.height // 2
    radius = min(center_x, center_y) // 8 * 7
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill="blue",
        outline="green",
        width=image.width // 20,
    )
    return image


def prepare_inputs():
    """
    Prepare inputs for the model.
    """
    text = [
        "circle",
        "ellipsoid",
        "blue circle on red background",
        "blue circle with green border on red background",
        "green circle on red background",
        "a dog",
        "a blue dog with a green border on a red background",
    ]
    img224 = create_image(224, 224)
    img1024 = create_image(1024, 1024)
    img224_1024 = create_image(1024, 224)

    images = [img224, img1024, img224_1024]
    return text, images


# --------------------------------------------------------------------------------------------
# Convert model
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def convert_siglip2_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Siglip2 structure.
    """

    # Define Siglip2 configuration
    config = get_siglip2_config(model_name)

    checkpoint = MODEL_NAME_TO_CHECKPOINT_PATH[model_name]
    if not os.path.exists(checkpoint):
        org, repo_id, *filepath = checkpoint.split("/")
        checkpoint = hf_hub_download(repo_id=f"{org}/{repo_id}", filename="/".join(filepath))

    print(f"Loading checkpoint from {checkpoint}...")
    data = np.load(checkpoint)
    state_dict = flatten_nested_dict(data)
    state_dict = split_encoderblock_layers(state_dict)
    state_dict = merge_qkv_for_head(state_dict, config)

    # Rename and transform weights
    print("Renaming and transforming weights...")

    original_keys = list(state_dict.keys())
    hf_keys = convert_old_keys_to_new_keys(original_keys)

    new_state_dict = {}
    for original_key in original_keys:
        new_key = hf_keys[original_key]
        parameter = state_dict.pop(original_key)

        hidden_size = config.vision_config.hidden_size if "vision" in new_key else config.text_config.hidden_size

        if any(k in new_key for k in ("out_proj", "q_proj", "k_proj", "v_proj", "position_embedding")):
            parameter = parameter.reshape(-1, hidden_size)

        # Transpose every weight except for position_embedding and token_embedding
        if new_key.endswith("weight") and "position_embedding" not in new_key and "token_embedding" not in new_key:
            parameter = parameter.T

        # Reshape every bias
        if new_key.endswith("bias"):
            parameter = parameter.reshape(-1)

        new_state_dict[new_key] = torch.from_numpy(parameter)

    # load HuggingFace model
    print("Loading HuggingFace model...")
    model = Siglip2Model(config).eval()
    model.load_state_dict(new_state_dict)

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
        torch.testing.assert_close(outputs.logits_per_text, EXPECTED_OUTPUTS[model_name], atol=1e-3, rtol=1e-3)

    # Save model
    if pytorch_dump_folder_path is not None:
        dst_dir = os.path.join(pytorch_dump_folder_path, model_name)
        print(f"Saving model {model_name} to {dst_dir}...")
        model.save_pretrained(dst_dir)
        print(f"Saving processor to {dst_dir}...")
        processor.save_pretrained(dst_dir)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to the HuggingFace Hub...")
        model.push_to_hub(f"qubvel-hf/{model_name}", private=True)
        processor.push_to_hub(f"qubvel-hf/{model_name}", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="siglip2-base-patch16-naflex",
        type=str,
        choices=MODEL_NAME_TO_CHECKPOINT_PATH.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="checkpoints/",
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
