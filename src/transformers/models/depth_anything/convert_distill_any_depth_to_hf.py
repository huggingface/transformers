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
"""Convert Distill Any Depth checkpoints from the original repository. URL:
https://github.com/Westlake-AGI-Lab/Distill-Any-Depth"""

import argparse
import re
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"(backbone|pretrained)\.cls_token": r"backbone.embeddings.cls_token",
    r"(backbone|pretrained)\.mask_token": r"backbone.embeddings.mask_token",
    r"(backbone|pretrained)\.pos_embed": r"backbone.embeddings.position_embeddings",
    r"(backbone|pretrained)\.patch_embed\.proj\.(weight|bias)": r"backbone.embeddings.patch_embeddings.projection.\2",
    r"(backbone|pretrained)\.norm\.(weight|bias)": r"backbone.layernorm.\2",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.attn\.proj\.(weight|bias)": r"backbone.encoder.layer.\4.attention.output.dense.\5",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.ls(1|2)\.gamma": r"backbone.encoder.layer.\4.layer_scale\5.lambda1",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.mlp\.fc(1|2)\.(weight|bias)": r"backbone.encoder.layer.\4.mlp.fc\5.\6",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.norm(1|2)\.(weight|bias)": r"backbone.encoder.layer.\4.norm\5.\6",
    r"depth_head\.projects\.(\d+)\.(weight|bias)": r"neck.reassemble_stage.layers.\1.projection.\2",
    r"depth_head\.resize_layers\.(?!2)(\d+)\.(weight|bias)": r"neck.reassemble_stage.layers.\1.resize.\2",
    r"depth_head\.scratch\.layer(\d+)_rn\.weight": lambda m: f"neck.convs.{int(m[1]) - 1}.weight",
    r"depth_head\.scratch\.output_conv(\d+)(?:\.(\d+))?\.(weight|bias)": lambda m: (
        f"head.conv{int(m[1]) + (int(m[2]) // 2 if m[2] else 0)}.{m[3]}" if m[1] == "2" else f"head.conv{m[1]}.{m[3]}"
    ),
    r"depth_head\.scratch\.refinenet(\d+)\.out_conv\.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{3 - (int(m[1]) - 1)}.projection.{m[2]}",
    r"depth_head\.scratch\.refinenet(\d+)\.resConfUnit(\d+)\.conv(\d+)\.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{3 - (int(m[1]) - 1)}.residual_layer{m[2]}.convolution{m[3]}.{m[4]}",
}


def get_dpt_config(model_name):
    if "small" in model_name:
        out_indices = [3, 6, 9, 12]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 64
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name:
        out_indices = [3, 6, 9, 12]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 128
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name:
        out_indices = [5, 12, 18, 24]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 256
        neck_hidden_sizes = [256, 512, 1024, 1024]
    else:
        raise NotImplementedError(f"Model not supported: {model_name}")

    depth_estimation_type = "relative"
    max_depth = None

    config = DepthAnythingConfig(
        reassemble_hidden_size=backbone_config.hidden_size,
        patch_size=backbone_config.patch_size,
        backbone_config=backbone_config,
        fusion_hidden_size=fusion_hidden_size,
        neck_hidden_sizes=neck_hidden_sizes,
        depth_estimation_type=depth_estimation_type,
        max_depth=max_depth,
    )

    return config


def convert_key_pattern(key, mapping):
    for pattern, replacement in mapping.items():
        match = re.fullmatch(pattern, key)
        if match:
            if callable(replacement):
                return replacement(match)
            return re.sub(pattern, replacement, key)
    return None


def convert_keys(state_dict, config):
    new_state_dict = {}
    qkv_pattern = r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.attn\.qkv\.(weight|bias)"
    qkv_keys = [k for k in list(state_dict.keys()) if re.match(qkv_pattern, k)]
    for old_key in qkv_keys:
        value = state_dict.pop(old_key)
        match = re.match(qkv_pattern, old_key)
        _, _, _, layer, attr = match.groups()
        hidden_size = config.backbone_config.hidden_size
        q = value[:hidden_size]
        k = value[hidden_size : hidden_size * 2]
        v = value[-hidden_size:]

        for proj, tensor in zip(["query", "key", "value"], [q, k, v]):
            new_key = f"backbone.encoder.layer.{layer}.attention.attention.{proj}.{attr}"
            new_state_dict[new_key] = tensor

    for old_key in list(state_dict.keys()):
        value = state_dict.pop(old_key)
        new_key = convert_key_pattern(old_key, ORIGINAL_TO_CONVERTED_KEY_MAPPING)

        new_state_dict[new_key] = value

    return new_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(requests.get(url, stream=True).raw)


name_to_checkpoint = {
    "distill-any-depth-small": "small/model.safetensors",
    "distill-any-depth-base": "base/model.safetensors",
    "distill-any-depth-large": "large/model.safetensors",
}


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    config = get_dpt_config(model_name)

    repo_id = "xingyang1/Distill-Any-Depth"
    filepath = hf_hub_download(repo_id=repo_id, filename=name_to_checkpoint[model_name])
    state_dict = load_file(filepath)

    converted_state_dict = convert_keys(state_dict, config)

    model = DepthAnythingForDepthEstimation(config)
    model.load_state_dict(converted_state_dict)
    model.eval()

    processor = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        ensure_multiple_of=14,
        keep_aspect_ratio=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values:", predicted_depth[0, :3, :3])

    if verify_logits:
        print("Verifying logits...")
        expected_shape = torch.Size([1, 518, 686])

        if model_name == "distill-any-depth-small":
            expected_slice = torch.tensor(
                [[2.5653, 2.5249, 2.5570], [2.4897, 2.5235, 2.5355], [2.5255, 2.5261, 2.5422]]
            )
        elif model_name == "distill-any-depth-base":
            expected_slice = torch.tensor(
                [[4.8976, 4.9075, 4.9403], [4.8872, 4.8906, 4.9448], [4.8712, 4.8898, 4.8838]]
            )
        elif model_name == "distill-any-depth-large":
            expected_slice = torch.tensor(
                [[55.1067, 51.1828, 51.6803], [51.9098, 50.7529, 51.4494], [50.1745, 50.5491, 50.8818]]
            )
        else:
            raise ValueError("Not supported")

        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-4)
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
    parser.add_argument(
        "--model_name",
        default="distill-any-depth-small",
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
        action="store_true",
        required=False,
        help="Whether to verify the logits after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
