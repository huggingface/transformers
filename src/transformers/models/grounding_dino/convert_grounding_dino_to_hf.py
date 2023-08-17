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
"""Convert GroundingDINO SimMIM checkpoints from the original repository.

URL: https://github.com/microsoft/GroundingDINO-Transformer/blob/main/MODELHUB.md#simmim-pretrained-grounding_dino-v1-models"""

import argparse

import requests
import torch
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as F

from transformers import (
    GroundingDINOConfig, GroundingDINOModel, GroundingDINOForObjectDetection, ViTImageProcessor
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_grounding_dino_config(model_name):
    config = GroundingDINOConfig()

    if "tiny" in model_name:
        window_size = 6
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    elif "base" in model_name:
        window_size = 12
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    else:
        raise ValueError("Model not supported, only supports base and large variants")

    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads

    return config


def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("module.backbone.0.patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("module.backbone.0.patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("module.backbone.0.patch_embed.norm.weight", "embeddings.norm.weight"))
    rename_keys.append(("module.backbone.0.patch_embed.norm.bias", "embeddings.norm.bias"))

    for layer, depth in enumerate(config.depths):
        for block in range(depth):
            # layernorms
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm1.weight", 
                                f"encoder.layers.{layer}.blocks.{block}.layernorm_before.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm1.bias", 
                                f"encoder.layers.{layer}.blocks.{block}.layernorm_before.bias"))
            
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm2.weight", 
                                f"encoder.layers.{layer}.blocks.{block}.layernorm_after.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm2.bias", 
                                f"encoder.layers.{layer}.blocks.{block}.layernorm_after.bias"))
            # attention
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.relative_position_bias_table", 
                                f"encoder.layers.{layer}.blocks.{block}.attention.self.relative_position_bias_table"))
            # rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.relative_position_index", 
            #                     f"encoder.layers.{layer}.blocks.{block}.attention.relative_position_index"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.proj.weight", 
                            f"encoder.layers.{layer}.blocks.{block}.attention.output.dense.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.proj.bias", 
                            f"encoder.layers.{layer}.blocks.{block}.attention.output.dense.bias"))
            # intermidiate
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc1.weight", 
                            f"encoder.layers.{layer}.blocks.{block}.intermediate.dense.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc1.bias", 
                            f"encoder.layers.{layer}.blocks.{block}.intermediate.dense.bias"))
            
            # output
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc2.weight", 
                            f"encoder.layers.{layer}.blocks.{block}.output.dense.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc2.bias", 
                            f"encoder.layers.{layer}.blocks.{block}.output.dense.bias"))
            
        # downsample
        rename_keys.append((f"module.backbone.0.layers.{layer}.downsample.reduction.weight", 
                            f"encoder.layers.{layer}.downsample.reduction.weight"))
        rename_keys.append((f"module.backbone.0.layers.{layer}.downsample.reduction.bias", 
                            f"encoder.layers.{layer}.downsample.reduction.bias"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def image_processor(image: Image.Image) -> torch.Tensor:
    def resize(image, size, max_size=None):
        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        size = get_size_with_aspect_ratio(image.size, size, max_size)
        return F.resize(image, size)
    
    transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    image_resized = resize(image, 800, 1333)
    return transform(image_resized)

@torch.no_grad()
def convert_grounding_dino_checkpoint(model_name, checkpoint_path):
    #Define default GroundingDINO configuation
    config = get_grounding_dino_config(model_name)

    # Load original checkpoint
    original_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    # read_in_q_k_v(new_state_dict, config)

    # Load HF implementation with default config and converted state dict
    model = GroundingDINOModel(config).eval()
    model.load_state_dict(new_state_dict, strict=False)

    # Load and process test image
    image = prepare_img()
    inputs = image_processor(image)
    model(pixel_values=inputs.unsqueeze(0))

    # outputs = model(**inputs).logits

    # print(outputs.keys())
    # print("Looks ok!")

    # if pytorch_dump_folder_path is not None:
    #     print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    #     model.save_pretrained(pytorch_dump_folder_path)

    #     print(f"Saving image processor to {pytorch_dump_folder_path}")
    #     image_processor.save_pretrained(pytorch_dump_folder_path)

    # if push_to_hub:
    #     print(f"Pushing model and image processor for {model_name} to hub")
    #     model.push_to_hub(f"microsoft/{model_name}")
    #     image_processor.push_to_hub(f"microsoft/{model_name}")


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
        "--checkpoint_path",
        default="/home/eduardo/Desktop/Projects/GroundingDINO/weights/grounding_dino_tiny.pth",
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
    convert_grounding_dino_checkpoint(args.model_name, args.checkpoint_path)
