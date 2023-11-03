# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ViT MAE checkpoints from the original repository: https://github.com/facebookresearch/mae"""

import argparse

import requests
import torch
from PIL import Image

from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEImageProcessor


def rename_key(name):
    if "cls_token" in name:
        name = name.replace("cls_token", "vit.embeddings.cls_token")
    if "mask_token" in name:
        name = name.replace("mask_token", "decoder.mask_token")
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "vit.embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "vit.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "vit.embeddings.norm")
    if "decoder_blocks" in name:
        name = name.replace("decoder_blocks", "decoder.decoder_layers")
    if "blocks" in name:
        name = name.replace("blocks", "vit.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    if "norm.weight" in name and "decoder" not in name:
        name = name.replace("norm.weight", "vit.layernorm.weight")
    if "norm.bias" in name and "decoder" not in name:
        name = name.replace("norm.bias", "vit.layernorm.bias")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            if "decoder_blocks" in key:
                dim = config.decoder_hidden_size
                prefix = "decoder.decoder_layers."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
                elif "bias" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.bias"] = val[-dim:]
            else:
                dim = config.hidden_size
                prefix = "vit.encoder.layer."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
                elif "bias" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.bias"] = val[-dim:]

        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_vit_mae_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    config = ViTMAEConfig()
    if "large" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "huge" in checkpoint_url:
        config.patch_size = 14
        config.hidden_size = 1280
        config.intermediate_size = 5120
        config.num_hidden_layers = 32
        config.num_attention_heads = 16

    model = ViTMAEForPreTraining(config)

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    image_processor = ViTMAEImageProcessor(size=config.image_size)

    new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()

    url = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = ViTMAEImageProcessor(size=config.image_size)
    inputs = image_processor(images=image, return_tensors="pt")

    # forward pass
    torch.manual_seed(2)
    outputs = model(**inputs)
    logits = outputs.logits

    if "large" in checkpoint_url:
        expected_slice = torch.tensor(
            [[-0.7309, -0.7128, -1.0169], [-1.0161, -0.9058, -1.1878], [-1.0478, -0.9411, -1.1911]]
        )
    elif "huge" in checkpoint_url:
        expected_slice = torch.tensor(
            [[-1.1599, -0.9199, -1.2221], [-1.1952, -0.9269, -1.2307], [-1.2143, -0.9337, -1.2262]]
        )
    else:
        expected_slice = torch.tensor(
            [[-0.9192, -0.8481, -1.1259], [-1.1349, -1.0034, -1.2599], [-1.1757, -1.0429, -1.2726]]
        )

    # verify logits
    assert torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4)

    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_mae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
