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
    GroundingDINOConfig, GroundingDINOForObjectDetection
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_grounding_dino_config(model_name):
    config = GroundingDINOConfig()

    if "tiny" in model_name:
        window_size = 7
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        image_size = 224
    elif "base" in model_name:
        window_size = 12
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        image_size = 384
    else:
        raise ValueError("Model not supported, only supports base and large variants")

    config.backbone_config.window_size = window_size
    config.backbone_config.image_size = image_size
    config.backbone_config.embed_dim = embed_dim
    config.backbone_config.depths = depths
    config.backbone_config.num_heads = num_heads
    config.backbone_config.out_indices = [2, 3, 4]

    return config


def create_rename_keys(config):
    rename_keys = []
    # fmt: off
    #TODO names might change after modifing GroundingDINOModel class
    ########################################## VISION BACKBONE - START
    # patch embedding layer
    rename_keys.append(("module.backbone.0.patch_embed.proj.weight", 
                        "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("module.backbone.0.patch_embed.proj.bias", 
                        "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("module.backbone.0.patch_embed.norm.weight", 
                        "model.backbone.conv_encoder.model.embeddings.norm.weight"))
    rename_keys.append(("module.backbone.0.patch_embed.norm.bias", 
                        "model.backbone.conv_encoder.model.embeddings.norm.bias"))

    for layer, depth in enumerate(config.backbone_config.depths):
        for block in range(depth):
            # layernorms
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm1.weight", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_before.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm1.bias", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_before.bias"))
            
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm2.weight", 
                                f"encoder.layers.{layer}.blocks.{block}.layernorm_after.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.norm2.bias", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_after.bias"))
            # attention
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.relative_position_bias_table", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.relative_position_bias_table"))
            # rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.relative_position_index", 
            #                     f"encoder.layers.{layer}.blocks.{block}.attention.relative_position_index"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.proj.weight", 
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.output.dense.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.attn.proj.bias", 
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.output.dense.bias"))
            # intermidiate
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc1.weight", 
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.intermediate.dense.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc1.bias", 
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.intermediate.dense.bias"))
            
            # output
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc2.weight", 
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.output.dense.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.blocks.{block}.mlp.fc2.bias", 
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.output.dense.bias"))
            
        # downsample
        if layer!=len(config.backbone_config.depths)-1:
            rename_keys.append((f"module.backbone.0.layers.{layer}.downsample.reduction.weight", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.downsample.reduction.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.downsample.norm.weight", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.downsample.norm.weight"))
            rename_keys.append((f"module.backbone.0.layers.{layer}.downsample.norm.bias", 
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.downsample.norm.bias"))
    
    for out_indice in config.backbone_config.out_indices:
        # Grounding DINO implementation of out_indices isn't aligned with transformers
        rename_keys.append((f"module.backbone.0.norm{out_indice-1}.weight", 
                        f"model.backbone.conv_encoder.model.hidden_states_norms.stage{out_indice}.weight"))
        rename_keys.append((f"module.backbone.0.norm{out_indice-1}.bias", 
                        f"model.backbone.conv_encoder.model.hidden_states_norms.stage{out_indice}.bias"))
        
    ########################################## VISION BACKBONE - END

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    ########################################## VISION BACKBONE - START
    embed_dim = config.backbone_config.embed_dim
    for layer, depth in enumerate(config.backbone_config.depths):
        hidden_size = embed_dim * 2**layer
        for block in range(depth):
            # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"module.backbone.0.layers.{layer}.blocks.{block}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"module.backbone.0.layers.{layer}.blocks.{block}.attn.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.query.weight"] = in_proj_weight[: hidden_size, :]
            state_dict[f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.query.bias"] = in_proj_bias[: hidden_size]

            state_dict[f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.key.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
            state_dict[f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.key.bias"] = in_proj_bias[hidden_size : hidden_size * 2]

            state_dict[f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.value.weight"] = in_proj_weight[-hidden_size :, :]
            state_dict[f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.value.bias"] = in_proj_bias[-hidden_size :]
    ########################################## VISION BACKBONE - END


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

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
    read_in_q_k_v(new_state_dict, config)

    # Load HF implementation with default config and converted state dict
    model = GroundingDINOForObjectDetection(config).eval()
    model.load_state_dict(new_state_dict, strict=False)

    # Load and process test image
    image = prepare_img()
    image_processor = T.Compose(
        [
            T.Resize(size=800, max_size=1333),
            T.ToTensor(), 
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]
    )
    inputs = image_processor(image)
    pixel_mask = torch.ones(((1, inputs.shape[1], inputs.shape[2])), dtype=torch.long, device=inputs.device)
    output= model.model.backbone.conv_encoder.model(pixel_values=inputs.unsqueeze(0))
    for feature_map in output.feature_maps:
        print(f"{feature_map.shape}")
        print(f"\t {feature_map[:, :5, 0, 0].cpu().numpy()}")

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