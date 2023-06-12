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
"""Convert VitMatte checkpoints from the original repository.

URL: https://github.com/hustvl/ViTMatte
"""

import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting


def get_config(model_name):
    hidden_size = 384 if "small" in model_name else 768
    num_attention_heads = 6 if "small" in model_name else 12

    backbone_config = VitDetConfig(
        num_channels=4,
        image_size=512,
        pretrain_image_size=224,
        patch_size=16,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=True,
        window_size=14,
        # 2, 5, 8, 11 for global attention
        window_block_indices=[0, 1, 3, 4, 6, 7, 9, 10],
        residual_block_indices=[2, 5, 8, 11],
        out_features=["stage12"],
    )

    return VitMatteConfig(backbone_config=backbone_config, hidden_size=hidden_size)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("backbone.pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("backbone.patch_embed.proj.weight", "backbone.embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "backbone.embeddings.projection.bias"))
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_vitmatte_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    config = get_config(model_name)

    # load original state dict
    filepath = hf_hub_download(repo_id="nielsr/vitmatte-checkpoints", filename="ViTMatte_S_Com.pth", repo_type="model")
    state_dict = torch.load(filepath, map_location="cpu")

    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "backbone.blocks" in key:
            key = key.replace("backbone.blocks", "backbone.encoder.layer")
        if "attn" in key:
            key = key.replace("attn", "attention")
        if "fusion_blks" in key:
            key = key.replace("fusion_blks", "fusion_blocks")
        state_dict[key] = val

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # create model
    model = VitMatteForImageMatting(config)
    model.eval()

    # load state dict
    model.load_state_dict(state_dict)

    # verify on dummy inputs
    # TODO use processor
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    filepath = hf_hub_download(repo_id="nielsr/vitmatte-dummy-data", filename="images.pt", repo_type="dataset")
    images = torch.load(filepath, map_location="cpu")

    print("Shape of images:", images.shape)
    with torch.no_grad():
        alphas = model(images).alphas

    print(alphas.shape)
    print(alphas[0, 0, :3, :3])

    expected_slice = torch.tensor(
        [
            [4.4248e-04, 8.3591e-06, 2.3893e-06],
            [8.2327e-06, 6.1929e-07, 2.4754e-07],
            [1.3892e-06, 1.1131e-07, 2.8105e-08],
        ]
    )

    assert torch.allclose(alphas[0, 0, :3, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving processor to {pytorch_dump_folder_path}")
        # processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"nielsr/{model_name}")
        # processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vitmatte-small-composition-1k",
        type=str,
        choices=[
            "vitmatte-small-composition-1k",
            "vitmatte-base-composition-1k",
            "vitmatte-small-distinctions-646",
            "vitmatte-base-distinctions-646",
        ],
        help="Name of the VitMatte model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_vitmatte_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
