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
"""Convert IJEPA checkpoints from the original repository.

URL: https://github.com/facebookresearch/ijepa
"""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import (
    IJepaConfig,
    IJepaModel,
    ViTImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # projection layer + position embeddings
    rename_keys.append(("pos_embed", "ijepa.embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "ijepa.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "ijepa.embeddings.patch_embeddings.projection.bias"))

    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (
                f"blocks.{i}.norm1.weight",
                f"ijepa.encoder.layer.{i}.layernorm_before.weight",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm1.bias",
                f"ijepa.encoder.layer.{i}.layernorm_before.bias",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.attn.proj.weight",
                f"ijepa.encoder.layer.{i}.attention.output.dense.weight",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.attn.proj.bias",
                f"ijepa.encoder.layer.{i}.attention.output.dense.bias",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm2.weight",
                f"ijepa.encoder.layer.{i}.layernorm_after.weight",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm2.bias",
                f"ijepa.encoder.layer.{i}.layernorm_after.bias",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc1.weight",
                f"ijepa.encoder.layer.{i}.intermediate.dense.weight",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc1.bias",
                f"ijepa.encoder.layer.{i}.intermediate.dense.bias",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc2.weight",
                f"ijepa.encoder.layer.{i}.output.dense.weight",
            )
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc2.bias",
                f"ijepa.encoder.layer.{i}.output.dense.bias",
            )
        )

    # layernorm + pooler
    rename_keys.extend(
        [
            ("norm.weight", "layernorm.weight"),
            ("norm.bias", "layernorm.bias"),
        ]
    )

    # if just the base model, we should remove "ijepa" from all keys that start with "ijepa"
    rename_keys = [(pair[0], pair[1][6:]) if pair[1].startswith("ijepa") else pair for pair in rename_keys]

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_ijepa_config(model_name):
    patch_size = int(model_name.split("_")[1][4:])
    config = IJepaConfig(patch_size=patch_size)
    if "vith" in model_name:
        config.hidden_size = 1280
        config.num_hidden_layers = 32
        config.num_attention_heads = 16
        config.layer_norm_eps = 1e-6
        config.mlp_ratio = 4
        config.intermediate_size = 5120
        if model_name == "ijepa_vith16_1k":
            config.image_size = 448
    elif "vitg" in model_name:
        config.hidden_size = 1408
        config.num_hidden_layers = 40
        config.num_attention_heads = 16
        config.layer_norm_eps = 1e-6
        config.mlp_ratio = 48 / 11
        config.intermediate_size = 6144
    else:
        raise ValueError("Model not supported, only supports huge and giant models.")
    return config


@torch.no_grad()
def convert_ijepa_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our IJEPA structure.
    """

    # define default IJEPA configuration
    config = get_ijepa_config(model_name)

    checkpoint_mapping = {
        "ijepa_vith14_1k": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
        "ijepa_vith14_22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
        "ijepa_vith16_1k": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar",
        "ijepa_vitg16_22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar",
    }

    # Load original checkpoint
    checkpoint_url = checkpoint_mapping[model_name]
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["encoder"]
    original_state_dict = {k.replace("module.", ""): v for k, v in original_state_dict.items()}

    # Rename keys
    state_dict = original_state_dict.copy()
    remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = IJepaModel(config, add_pooling_layer=False).eval()
    model.load_state_dict(state_dict)

    if verify_logits:
        # Check outputs on an image, prepared by ViTImageProcessor
        image_processor = ViTImageProcessor()
        encoding = image_processor(images=prepare_img(), return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        with torch.no_grad():
            outputs = model(pixel_values)

        expected_slice = torch.Tensor(
            [[-0.0621, -0.0054, -2.7513], [-0.1952, 0.0909, -3.9536], [0.0942, -0.0331, -1.2833]]
        )

        assert torch.allclose(
            expected_slice,
            outputs.last_hidden_state[0, :3, :3],
            atol=1e-4,
        )

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_name_to_hf_name = {
            "ijepa_vith14_1k": "ijepa_huge_patch14_1k",
            "ijepa_vith14_22k": "ijepa_huge_patch14_22k",
            "ijepa_vith16_1k": "ijepa_huge_patch16_1k",
            "ijepa_vitg16_22k": "ijepa_giant_patch16_22k",
        }
        name = model_name_to_hf_name[model_name]
        model.push_to_hub(f"jmtzt/{name}", use_temp_dir=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="ijepa_vith14_1k",
        type=str,
        choices=[
            "ijepa_vith14_1k",
            "ijepa_vith14_22k",
            "ijepa_vith16_1k",
            "ijepa_vitg16_22k",
        ],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the 🤗 Hub.",
    )
    parser.add_argument(
        "--verify_logits", action="store_false", help="Whether or not to verify logits after conversion."
    )

    parser.set_defaults()
    args = parser.parse_args()
    convert_ijepa_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
