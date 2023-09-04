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
"""Convert Swin SimMIM checkpoints from the original repository.

URL: https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#simmim-pretrained-swin-v1-models"""

import argparse

import requests
import torch
from PIL import Image

from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor


def get_swin_config(model_name):
    config = SwinConfig(image_size=192)

    if "base" in model_name:
        window_size = 6
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    elif "large" in model_name:
        window_size = 12
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
    else:
        raise ValueError("Model not supported, only supports base and large variants")

    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads

    return config


def rename_key(name):
    if "encoder.mask_token" in name:
        name = name.replace("encoder.mask_token", "embeddings.mask_token")
    if "encoder.patch_embed.proj" in name:
        name = name.replace("encoder.patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "encoder.patch_embed.norm" in name:
        name = name.replace("encoder.patch_embed.norm", "embeddings.norm")
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

    if name == "encoder.norm.weight":
        name = "layernorm.weight"
    if name == "encoder.norm.bias":
        name = "layernorm.bias"

    if "decoder" in name:
        pass
    else:
        name = "swin." + name

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "attn_mask" in key:
            pass
        elif "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            block_num = int(key_split[4])
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[
                    :dim
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[
                    -dim:
                ]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swin_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    config = get_swin_config(model_name)
    model = SwinForMaskedImageModeling(config)
    model.eval()

    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image_processor = ViTImageProcessor(size={"height": 192, "width": 192})
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    print(outputs.keys())
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and image processor for {model_name} to hub")
        model.push_to_hub(f"microsoft/{model_name}")
        image_processor.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="swin-base-simmim-window6-192",
        type=str,
        choices=["swin-base-simmim-window6-192", "swin-large-simmim-window12-192"],
        help="Name of the Swin SimMIM model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/SwinSimMIM/simmim_pretrain__swin_base__img192_window6__100ep.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_swin_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
