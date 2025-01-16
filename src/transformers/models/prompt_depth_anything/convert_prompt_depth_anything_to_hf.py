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
"""Convert Prompt Depth Anything checkpoints from the original repository. URL:
https://github.com/DepthAnything/PromptDA"""

import argparse
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    Dinov2Config,
    PromptDepthAnythingConfig,
    PromptDepthAnythingForDepthEstimation,
    PromptDepthAnythingImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    if "small" in model_name or "vits" in model_name:
        out_indices = [3, 6, 9, 12]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 64
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name or "vitb" in model_name:
        out_indices = [3, 6, 9, 12]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 128
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name or "vitl" in model_name:
        out_indices = [5, 12, 18, 24]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 256
        neck_hidden_sizes = [256, 512, 1024, 1024]
    else:
        raise NotImplementedError(f"Model not supported: {model_name}")

    depth_estimation_type = "metric"
    max_depth = None

    config = PromptDepthAnythingConfig(
        reassemble_hidden_size=backbone_config.hidden_size,
        patch_size=backbone_config.patch_size,
        backbone_config=backbone_config,
        fusion_hidden_size=fusion_hidden_size,
        neck_hidden_sizes=neck_hidden_sizes,
        depth_estimation_type=depth_estimation_type,
        max_depth=max_depth,
    )

    return config


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"pretrained.cls_token": r"backbone.embeddings.cls_token",
    r"pretrained.mask_token": r"backbone.embeddings.mask_token",
    r"pretrained.pos_embed": r"backbone.embeddings.position_embeddings",
    r"pretrained.patch_embed.proj.weight": r"backbone.embeddings.patch_embeddings.projection.weight",
    r"pretrained.patch_embed.proj.bias": r"backbone.embeddings.patch_embeddings.projection.bias",
    r"pretrained.norm.weight": r"backbone.layernorm.weight",
    r"pretrained.norm.bias": r"backbone.layernorm.bias",
    r"depth_head.scratch.output_conv1.weight": r"head.conv1.weight",
    r"depth_head.scratch.output_conv1.bias": r"head.conv1.bias",
    r"depth_head.scratch.output_conv2.0.weight": r"head.conv2.weight",
    r"depth_head.scratch.output_conv2.0.bias": r"head.conv2.bias",
    r"depth_head.scratch.output_conv2.2.weight": r"head.conv3.weight",
    r"depth_head.scratch.output_conv2.2.bias": r"head.conv3.bias",
    r"pretrained.blocks.(\d+).ls1.gamma": r"backbone.encoder.layer.\1.layer_scale1.lambda1",
    r"pretrained.blocks.(\d+).ls2.gamma": r"backbone.encoder.layer.\1.layer_scale2.lambda1",
    r"pretrained.blocks.(\d+).norm1.weight": r"backbone.encoder.layer.\1.norm1.weight",
    r"pretrained.blocks.(\d+).norm1.bias": r"backbone.encoder.layer.\1.norm1.bias",
    r"pretrained.blocks.(\d+).norm2.weight": r"backbone.encoder.layer.\1.norm2.weight",
    r"pretrained.blocks.(\d+).norm2.bias": r"backbone.encoder.layer.\1.norm2.bias",
    r"pretrained.blocks.(\d+).mlp.fc1.weight": r"backbone.encoder.layer.\1.mlp.fc1.weight",
    r"pretrained.blocks.(\d+).mlp.fc1.bias": r"backbone.encoder.layer.\1.mlp.fc1.bias",
    r"pretrained.blocks.(\d+).mlp.fc2.weight": r"backbone.encoder.layer.\1.mlp.fc2.weight",
    r"pretrained.blocks.(\d+).mlp.fc2.bias": r"backbone.encoder.layer.\1.mlp.fc2.bias",
    r"pretrained.blocks.(\d+).attn.proj.weight": r"backbone.encoder.layer.\1.attention.output.dense.weight",
    r"pretrained.blocks.(\d+).attn.proj.bias": r"backbone.encoder.layer.\1.attention.output.dense.bias",
    r"pretrained.blocks.(\d+).attn.qkv.weight": r"qkv_transform_\1",
    r"pretrained.blocks.(\d+).attn.qkv.bias": r"qkv_transform_bias_\1",
    r"depth_head.projects.(\d+).weight": r"neck.reassemble_stage.layers.\1.projection.weight",
    r"depth_head.projects.(\d+).bias": r"neck.reassemble_stage.layers.\1.projection.bias",
    r"depth_head.scratch.layer(\d+)_rn.weight": r"neck.convs.\0.weight",
    r"depth_head.resize_layers.(\d+).weight": r"neck.reassemble_stage.layers.\1.resize.weight",
    r"depth_head.resize_layers.(\d+).bias": r"neck.reassemble_stage.layers.\1.resize.bias",
    r"depth_head.scratch.refinenet(\d+).out_conv.weight": r"neck.fusion_stage.layers.\0.projection.weight",
    r"depth_head.scratch.refinenet(\d+).out_conv.bias": r"neck.fusion_stage.layers.\0.projection.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit1.conv1.weight": r"neck.fusion_stage.layers.\0.residual_layer1.convolution1.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit1.conv1.bias": r"neck.fusion_stage.layers.\0.residual_layer1.convolution1.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit1.conv2.weight": r"neck.fusion_stage.layers.\0.residual_layer1.convolution2.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit1.conv2.bias": r"neck.fusion_stage.layers.\0.residual_layer1.convolution2.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit2.conv1.weight": r"neck.fusion_stage.layers.\0.residual_layer2.convolution1.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit2.conv1.bias": r"neck.fusion_stage.layers.\0.residual_layer2.convolution1.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit2.conv2.weight": r"neck.fusion_stage.layers.\0.residual_layer2.convolution2.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit2.conv2.bias": r"neck.fusion_stage.layers.\0.residual_layer2.convolution2.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.0.weight": r"neck.fusion_stage.layers.\0.prompt_depth_layer.convolution1.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.0.bias": r"neck.fusion_stage.layers.\0.prompt_depth_layer.convolution1.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.2.weight": r"neck.fusion_stage.layers.\0.prompt_depth_layer.convolution2.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.2.bias": r"neck.fusion_stage.layers.\0.prompt_depth_layer.convolution2.bias",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.4.weight": r"neck.fusion_stage.layers.\0.prompt_depth_layer.convolution3.weight",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.4.bias": r"neck.fusion_stage.layers.\0.prompt_depth_layer.convolution3.bias",
}


def transform_qkv_weights(key, value, config):
    if not key.startswith("qkv_transform"):
        return value

    layer_idx = int(key.split("_")[-1])
    hidden_size = config.backbone_config.hidden_size

    if "bias" in key:
        # Handle bias
        return {
            f"backbone.encoder.layer.{layer_idx}.attention.attention.query.bias": value[:hidden_size],
            f"backbone.encoder.layer.{layer_idx}.attention.attention.key.bias": value[hidden_size : hidden_size * 2],
            f"backbone.encoder.layer.{layer_idx}.attention.attention.value.bias": value[-hidden_size:],
        }
    else:
        # Handle weights
        return {
            f"backbone.encoder.layer.{layer_idx}.attention.attention.query.weight": value[:hidden_size, :],
            f"backbone.encoder.layer.{layer_idx}.attention.attention.key.weight": value[
                hidden_size : hidden_size * 2, :
            ],
            f"backbone.encoder.layer.{layer_idx}.attention.attention.value.weight": value[-hidden_size:, :],
        }


name_to_checkpoint = {
    "prompt-depth-anything-vits": "model.ckpt",
    "prompt-depth-anything-vits-transparent": "model.ckpt",
    "prompt-depth-anything-vitl": "model.ckpt",
}


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # define DPT configuration
    config = get_dpt_config(model_name)

    model_name_to_repo = {
        "prompt-depth-anything-vits": "depth-anything/prompt-depth-anything-vits",
        "prompt-depth-anything-vits-transparent": "depth-anything/prompt-depth-anything-vits-transparent",
        "prompt-depth-anything-vitl": "depth-anything/prompt-depth-anything-vitl",
    }

    # load original state_dict
    repo_id = model_name_to_repo[model_name]
    filename = name_to_checkpoint[model_name]
    filepath = hf_hub_download(
        repo_id=repo_id,
        filename=f"{filename}",
    )

    state_dict = torch.load(filepath, map_location="cpu")["state_dict"]
    state_dict = {key[9:]: state_dict[key] for key in state_dict}

    # Convert state dict using mappings
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in ORIGINAL_TO_CONVERTED_KEY_MAPPING:
            new_key = ORIGINAL_TO_CONVERTED_KEY_MAPPING[key]
            transformed_value = transform_qkv_weights(new_key, value, config)
            if isinstance(transformed_value, dict):
                new_state_dict.update(transformed_value)
            else:
                new_state_dict[new_key] = transformed_value

    # load HuggingFace model
    model = PromptDepthAnythingForDepthEstimation(config)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    processor = PromptDepthAnythingImageProcessor(
        do_resize=True,
        size=756,
        ensure_multiple_of=14,
        keep_aspect_ratio=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt_depth_url = (
        "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
    )
    prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)

    inputs = processor(image, return_tensors="pt", prompt_depth=prompt_depth)

    # Verify forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values:", predicted_depth[0, :3, :3])

    # assert logits
    if verify_logits:
        expected_shape = torch.Size([1, 756, 1008])
        if model_name == "prompt-depth-anything-vits":
            expected_slice = torch.tensor(
                [[3.0100, 3.0016, 3.0219], [3.0046, 3.0137, 3.0275], [3.0083, 3.0191, 3.0292]]
            )
        elif model_name == "prompt-depth-anything-vits-transparent":
            expected_slice = torch.tensor(
                [[3.0058, 3.0397, 3.0460], [3.0314, 3.0393, 3.0504], [3.0326, 3.0465, 3.0545]]
            )
        elif model_name == "prompt-depth-anything-vitl":
            expected_slice = torch.tensor(
                [[3.1336, 3.1358, 3.1363], [3.1368, 3.1267, 3.1414], [3.1397, 3.1385, 3.1448]]
            )
        else:
            raise ValueError("Not supported")
        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=5e-3)  # 5mm tolerance
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(repo_id=f"{model_name.title()}-hf")
        processor.push_to_hub(repo_id=f"{model_name.title()}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="prompt_depth_anything_vits",
        type=str,
        choices=name_to_checkpoint.keys(),
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
        help="Whether to push the model to the hub after conversion.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Whether to verify the logits after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
