# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Convert DINOv3 checkpoints from the original repository.

URL: https://github.com/facebookresearch/dinov3/tree/main
"""

import argparse
import os
import re
from typing import Optional

import requests
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel, DINOv3ViTImageProcessorFast


HUB_MODELS = {
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

HUB_CHECKPOINTS = {
    "convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"dwconv":                              r"depthwise_conv",
    r"pwconv":                              r"pointwise_conv",
    r"norm":                                r"layer_norm",
    r"stages.(\d+).(\d+)":                  r"stages.\1.layers.\2",
    r"downsample_layers.(\d+).(\d+)":       r"stages.\1.downsample_layers.\2",
}
# fmt: on


def get_dinov3_config(model_name: str) -> DINOv3ConvNextConfig:
    # size of the architecture
    if model_name == "convnext_tiny":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 9, 3],
            hidden_sizes=[96, 192, 384, 768],
        )
    elif model_name == "convnext_small":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 27, 3],
            hidden_sizes=[96, 192, 384, 768],
        )
    elif model_name == "convnext_base":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 27, 3],
            hidden_sizes=[128, 256, 512, 1024],
        )
    elif model_name == "convnext_large":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 27, 3],
            hidden_sizes=[192, 384, 768, 1536],
        )
    else:
        raise ValueError("Model not supported")


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def get_transform(resize_size: int = 224):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


def get_image_processor(resize_size: int = 224):
    return DINOv3ViTImageProcessorFast(
        do_resize=True,
        size={"height": resize_size, "width": resize_size},
        resample=2,  # BILINEAR
    )


def convert_old_keys_to_new_keys(state_dict_keys: Optional[dict] = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


@torch.no_grad()
def convert_and_test_dinov3_checkpoint(args):
    expected_outputs = {
        "convnext_tiny_cls": [-6.372119, 1.300791, 2.074303, -0.079975, 0.607205],
        "convnext_tiny_patch": [0.490530, -3.713466, 1.848513, -1.040319, -1.090818],
        "convnext_small_cls": [-0.903914, 1.412183, 0.287465, 0.175296, -2.397940],
        "convnext_small_patch": [-1.081114, 0.637362, 3.748765, 0.170179, 1.445153],
        "convnext_base_cls": [0.155366, -0.378771, -0.735157, -2.818718, 0.015095],
        "convnext_base_patch": [3.039118, 0.778155, -1.961322, -1.607147, -2.411941],
        "convnext_large_cls": [-2.219094, -0.594451, -2.300294, -0.957415, -0.520473],
        "convnext_large_patch": [-1.477349, -0.217038, -3.128137, 0.418962, 0.334949],
    }
    model_name = args.model_name
    config = get_dinov3_config(model_name)
    # print(config)

    model = DINOv3ConvNextModel(config).eval()
    state_dict_path = hf_hub_download(repo_id=HUB_MODELS[model_name], filename=HUB_CHECKPOINTS[model_name])
    original_state_dict = torch.load(state_dict_path)
    original_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(original_keys)

    converted_state_dict = {}
    for key in original_keys:
        new_key = new_keys[key]
        weight_tensor = original_state_dict[key]
        if key == "norms.3.weight" or key == "norms.3.bias":
            continue
        converted_state_dict[new_key] = weight_tensor
    model.load_state_dict(converted_state_dict, strict=True)
    model = model.eval()

    transform = get_transform()
    image_processor = get_image_processor()
    image = prepare_img()

    # check preprocessing
    original_pixel_values = transform(image).unsqueeze(0)  # add batch dimension
    inputs = image_processor(image, return_tensors="pt")

    torch.testing.assert_close(original_pixel_values, inputs["pixel_values"], atol=1e-6, rtol=1e-6)
    print("Preprocessing looks ok!")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float):
        model_output = model(**inputs)

    last_layer_class_token = model_output.pooler_output
    last_layer_patch_tokens = model_output.last_hidden_state[:, 1:]

    actual_outputs = {}
    actual_outputs[f"{model_name}_cls"] = last_layer_class_token[0, :5].tolist()
    actual_outputs[f"{model_name}_patch"] = last_layer_patch_tokens[0, 0, :5].tolist()

    print("Actual:  ", [round(x, 6) for x in actual_outputs[f"{model_name}_cls"]])
    print("Expected:", expected_outputs[f"{model_name}_cls"])

    torch.testing.assert_close(
        torch.Tensor(actual_outputs[f"{model_name}_cls"]),
        torch.Tensor(expected_outputs[f"{model_name}_cls"]),
        atol=1e-3,
        rtol=1e-3,
    )
    print("Actual:  ", [round(x, 6) for x in actual_outputs[f"{model_name}_patch"]])
    print("Expected:", expected_outputs[f"{model_name}_patch"])

    torch.testing.assert_close(
        torch.Tensor(actual_outputs[f"{model_name}_patch"]),
        torch.Tensor(expected_outputs[f"{model_name}_patch"]),
        atol=1e-3,
        rtol=1e-3,
    )
    print("Forward pass looks ok!")

    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    if args.push_to_hub:
        api = HfApi()
        repo = HUB_MODELS[model_name]
        api.upload_folder(folder_path=save_dir, repo_id=repo, repo_type="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default="convnext_tiny",
        type=str,
        choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--save-dir",
        default="converted_models",
        type=str,
        help="Directory to save the converted model.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted model to the Hugging Face Hub.",
    )
    args = parser.parse_args()
    convert_and_test_dinov3_checkpoint(args)
