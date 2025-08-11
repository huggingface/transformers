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

from transformers import (
    DINOv3ConvNextConfig,
    DINOv3ViTImageProcessorFast,
    DINOv3ConvNextModel,
)


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


@torch.no_grad()
def convert_and_test_dinov3_checkpoint(args):
    expected_outputs = {
        "convnext_tiny_cls": [
            -6.372119903564453,
            1.3007919788360596,
            2.074303388595581,
            -0.0799759104847908,
            0.6072055697441101,
        ],
        "convnext_tiny_patch": [
            0.4905306398868561,
            -3.7134664058685303,
            1.8485137224197388,
            -1.0403193235397339,
            -1.0908184051513672,
        ],
        "convnext_small_cls": [
            -0.9039149284362793,
            1.4121832847595215,
            0.2874654531478882,
            0.17529653012752533,
            -2.3979403972625732,
        ],
        "convnext_small_patch": [
            -1.081114649772644,
            0.6373621821403503,
            3.7487659454345703,
            0.1701796054840088,
            1.4451534748077393,
        ],
        "convnext_base_cls": [
            0.15536683797836304,
            -0.37877172231674194,
            -0.7351579070091248,
            -2.818718671798706,
            0.015095720998942852,
        ],
        "convnext_base_patch": [
            3.0391180515289307,
            0.7781552672386169,
            -1.9613221883773804,
            -1.6071475744247437,
            -2.4119417667388916,
        ],
        "convnext_large_cls": [
            -2.219094753265381,
            -0.5944517254829407,
            -2.3002943992614746,
            -0.9574159979820251,
            -0.5204737782478333,
        ],
        "convnext_large_patch": [
            -1.477349042892456,
            -0.21703894436359406,
            -3.1281375885009766,
            0.41896212100982666,
            0.3349491357803345,
        ],
    }
    model_name = args.model_name
    config = get_dinov3_config(model_name)
    # print(config)

    model = DINOv3ConvNextModel(config).eval()
    state_dict_path = hf_hub_download(
        repo_id=HUB_MODELS[model_name], filename=HUB_CHECKPOINTS[model_name]
    )
    original_state_dict = torch.load(state_dict_path)
    original_keys = list(original_state_dict.keys())
    converted_state_dict = {}
    for key in original_keys:
        weight_tensor = original_state_dict[key]
        if key == "norms.3.weight" or key == "norms.3.bias":
            continue
        converted_state_dict[key] = weight_tensor
    model.load_state_dict(converted_state_dict, strict=True)
    model = model.eval()

    transform = get_transform()
    image_processor = get_image_processor()
    image = prepare_img()

    # check preprocessing
    original_pixel_values = transform(image).unsqueeze(0)  # add batch dimension
    inputs = image_processor(image, return_tensors="pt")

    torch.testing.assert_close(
        original_pixel_values, inputs["pixel_values"], atol=1e-6, rtol=1e-6
    )
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
