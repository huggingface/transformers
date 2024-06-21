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
"""Convert SegGPT checkpoints from the original repository.

URL: https://github.com/baaivision/Painter/tree/main/SegGPT
"""

import argparse

import requests
import torch
from PIL import Image

from transformers import SegGptConfig, SegGptForImageSegmentation, SegGptImageProcessor
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

        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"model.encoder.layers.{i}.mlp.lin1.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"model.encoder.layers.{i}.mlp.lin1.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"model.encoder.layers.{i}.mlp.lin2.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"model.encoder.layers.{i}.mlp.lin2.bias"))

        rename_keys.append((f"blocks.{i}.norm1.weight", f"model.encoder.layers.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"model.encoder.layers.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"model.encoder.layers.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"model.encoder.layers.{i}.layernorm_after.bias"))

    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on spongebob images
def prepare_input():
    image_input_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
    )
    image_prompt_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
    )
    mask_prompt_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"
    )

    image_input = Image.open(requests.get(image_input_url, stream=True).raw)
    image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
    mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw)

    return image_input, image_prompt, mask_prompt


@torch.no_grad()
def convert_seggpt_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    verify_logits = args.verify_logits
    push_to_hub = args.push_to_hub

    # Define default GroundingDINO configuation
    config = SegGptConfig()

    # Load original checkpoint
    checkpoint_url = "https://huggingface.co/BAAI/SegGpt/blob/main/seggpt_vit_large.pth"
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    # Load HF model
    model = SegGptForImageSegmentation(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    input_img, prompt_img, prompt_mask = prepare_input()
    image_processor = SegGptImageProcessor()
    inputs = image_processor(images=input_img, prompt_images=prompt_img, prompt_masks=prompt_mask, return_tensors="pt")

    expected_prompt_pixel_values = torch.tensor(
        [
            [[-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965]],
            [[1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583]],
            [[2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088]],
        ]
    )

    expected_pixel_values = torch.tensor(
        [
            [[1.6324, 1.6153, 1.5810], [1.6153, 1.5982, 1.5810], [1.5810, 1.5639, 1.5639]],
            [[1.2731, 1.2556, 1.2206], [1.2556, 1.2381, 1.2031], [1.2206, 1.2031, 1.1681]],
            [[1.6465, 1.6465, 1.6465], [1.6465, 1.6465, 1.6465], [1.6291, 1.6291, 1.6291]],
        ]
    )

    expected_prompt_masks = torch.tensor(
        [
            [[-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179]],
            [[-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357]],
            [[-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044]],
        ]
    )

    assert torch.allclose(inputs.pixel_values[0, :, :3, :3], expected_pixel_values, atol=1e-4)
    assert torch.allclose(inputs.prompt_pixel_values[0, :, :3, :3], expected_prompt_pixel_values, atol=1e-4)
    assert torch.allclose(inputs.prompt_masks[0, :, :3, :3], expected_prompt_masks, atol=1e-4)

    torch.manual_seed(2)
    outputs = model(**inputs)
    print(outputs)

    if verify_logits:
        expected_output = torch.tensor(
            [
                [[-2.1208, -2.1190, -2.1198], [-2.1237, -2.1228, -2.1227], [-2.1232, -2.1226, -2.1228]],
                [[-2.0405, -2.0396, -2.0403], [-2.0434, -2.0434, -2.0433], [-2.0428, -2.0432, -2.0434]],
                [[-1.8102, -1.8088, -1.8099], [-1.8131, -1.8126, -1.8129], [-1.8130, -1.8128, -1.8131]],
            ]
        )
        assert torch.allclose(outputs.pred_masks[0, :, :3, :3], expected_output, atol=1e-4)
        print("Looks good!")
    else:
        print("Converted without verifying logits")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"EduardoPacheco/{model_name}")
        image_processor.push_to_hub(f"EduardoPacheco/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="seggpt-vit-large",
        type=str,
        choices=["seggpt-vit-large"],
        help="Name of the SegGpt model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        help="Whether or not to verify the logits against the original implementation.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_seggpt_checkpoint(args)
