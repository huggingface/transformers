# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert SegGPT checkpoints"""


import argparse

import requests
import torch
from PIL import Image

from transformers import SegGPTConfig, SegGPTForInstanceSegmentation
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off

    # rename embedding and its parameters
    rename_keys.append(("patch_embed.proj.weight", "model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("mask_token", "model.embeddings.mask_token"))
    rename_keys.append(("segment_token_x", "model.embeddings.segment_token_input"))
    rename_keys.append(("segment_token_y", "model.embeddings.segment_token_prompt"))
    rename_keys.append(("type_token_cls", "model.embeddings.type_token_semantic"))
    rename_keys.append(("type_token_ins", "model.embeddings.type_token_instance"))
    rename_keys.append(("pos_embed", "model.embeddings.position_embeddings"))

    # rename decoder and other
    rename_keys.append(("norm.weight", "model.encoder.layernorm.weight"))
    rename_keys.append(("norm.bias", "model.encoder.layernorm.bias"))
    rename_keys.append(("decoder_embed.weight", "decoder.decoder_embed.weight"))
    rename_keys.append(("decoder_embed.bias", "decoder.decoder_embed.bias"))
    rename_keys.append(("decoder_pred.0.weight", "decoder.decoder_pred.conv.weight"))
    rename_keys.append(("decoder_pred.0.bias", "decoder.decoder_pred.conv.bias"))
    rename_keys.append(("decoder_pred.1.weight", "decoder.decoder_pred.layernorm.weight"))
    rename_keys.append(("decoder_pred.1.bias", "decoder.decoder_pred.layernorm.bias"))
    rename_keys.append(("decoder_pred.3.weight", "decoder.decoder_pred.head.weight"))
    rename_keys.append(("decoder_pred.3.bias", "decoder.decoder_pred.head.bias"))

    # rename blocks
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"blocks.{i}.attn.qkv.weight", f"model.encoder.layers.{i}.attention.qkv.weight"))
        rename_keys.append((f"blocks.{i}.attn.qkv.bias", f"model.encoder.layers.{i}.attention.qkv.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"model.encoder.layers.{i}.attention.proj.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"model.encoder.layers.{i}.attention.proj.bias"))
        rename_keys.append((f"blocks.{i}.attn.rel_pos_h", f"model.encoder.layers.{i}.attention.rel_pos_h"))
        rename_keys.append((f"blocks.{i}.attn.rel_pos_w", f"model.encoder.layers.{i}.attention.rel_pos_w"))

        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"model.encoder.layers.{i}.mlp.fc2.bias"))

        rename_keys.append((f"blocks.{i}.norm1.weight", f"model.encoder.layers.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"model.encoder.layers.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"model.encoder.layers.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"model.encoder.layers.{i}.layernorm_after.bias"))

    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_seggpt_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub

    # Define default GroundingDINO configuation
    config = SegGPTConfig()

    # Load original checkpoint
    checkpoint_url = "https://huggingface.co/BAAI/SegGPT/blob/main/seggpt_vit_large.pth"
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    # Load HF model
    model = SegGPTForInstanceSegmentation(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"EduardoPacheco/{model_name}")
        # processor.push_to_hub(f"EduardoPacheco/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="grounding-dino-tiny",
        type=str,
        choices=["grounding-dino-tiny", "grounding-dino-base"],
        help="Name of the GroundingDINO model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_seggpt_checkpoint(args)
