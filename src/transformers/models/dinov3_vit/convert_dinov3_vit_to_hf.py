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

from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast, DINOv3ViTModel


HUB_MODELS = {
    "vits16_lvd1689m": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus_lvd1689m": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16_lvd1689m": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16_lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vitl16_sat493m": "facebook/dinov3-vitl16-pretrain-sat493m",
    "vith16plus_lvd1689m": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16_lvd1689m": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "vit7b16_sat493m": "facebook/dinov3-vit7b16-pretrain-sat493m",
}

HUB_CHECKPOINTS = {
    "vits16_lvd1689m": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vits16plus_lvd1689m": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "vitb16_lvd1689m": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vitl16_lvd1689m": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "vitl16_sat493m": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "vith16plus_lvd1689m": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "vit7b16_lvd1689m": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "vit7b16_sat493m": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"cls_token":                   r"embeddings.cls_token",
    r"mask_token":                  r"embeddings.mask_token",
    r"storage_tokens":              r"embeddings.register_tokens",
    r"patch_embed.proj":            r"embeddings.patch_embeddings",
    r"periods":                     r"inv_freq",
    r"rope_embed":                  r"rope_embeddings",
    r"blocks.(\d+).attn.proj":      r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.":          r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+).gamma":  r"layer.\1.layer_scale\2.lambda1",
    r"blocks.(\d+).mlp.fc1":        r"layer.\1.mlp.up_proj",
    r"blocks.(\d+).mlp.fc2":        r"layer.\1.mlp.down_proj",
    r"blocks.(\d+).mlp":            r"layer.\1.mlp",
    r"blocks.(\d+).norm":           r"layer.\1.norm",
    r"w1":                          r"gate_proj",
    r"w2":                          r"up_proj",
    r"w3":                          r"down_proj",
}
# fmt: on


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


def split_qkv(state_dict: dict):
    keys = [x for x in state_dict.keys() if "qkv" in x]
    for key in keys:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v
    return state_dict


def get_dinov3_config(model_name: str) -> DINOv3ViTConfig:
    # size of the architecture
    if model_name == "vits16_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            proj_bias=True,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
        )
    elif model_name == "vits16plus_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            num_register_tokens=4,
            use_gated_mlp=True,
            hidden_act="silu",
        )
    elif model_name == "vitb16_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            proj_bias=True,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
        )
    elif model_name in ("vitl16_lvd1689m", "vitl16_sat493m"):
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
        )
    elif model_name == "vith16plus_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=20,
            num_register_tokens=4,
            use_gated_mlp=True,
            hidden_act="silu",
        )
    elif model_name in ("vit7b16_lvd1689m", "vit7b16_sat493m"):
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=4096,
            intermediate_size=8192,
            num_hidden_layers=40,
            num_attention_heads=32,
            query_bias=False,
            value_bias=False,
            num_register_tokens=4,
            use_gated_mlp=True,
            hidden_act="silu",
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
        "vits16_lvd1689m_cls": [0.463561, -0.415609, 0.408236, -0.126613, -0.286636],
        "vits16_lvd1689m_patch": [-0.038754, -0.250895, -0.016392, -0.455473, 0.571582],
        "vits16plus_lvd1689m_cls": [-0.471349, -1.365778, -0.317983, 0.377219, -0.769085],
        "vits16plus_lvd1689m_patch": [0.144551, -0.388117, -0.393433, -0.157695, -0.600380],
        "vitb16_lvd1689m_cls": [1.034643, -0.180609, -0.341018, -0.066376, -0.011383],
        "vitb16_lvd1689m_patch": [-0.082523, -0.456272, -0.728029, -0.430680, -0.152880],
        "vitl16_lvd1689m_cls": [0.484527, -0.582214, 0.480636, 0.592040, 0.945166],
        "vitl16_lvd1689m_patch": [-0.211367, -0.490863, -0.257131, 0.101763, 0.154511],
        "vith16plus_lvd1689m_cls": [-0.064575, -0.148866, -0.621524, 0.634878, 0.152695],
        "vith16plus_lvd1689m_patch": [-0.093817, 0.287407, -0.050036, 0.428043, 0.094561],
        "vit7b16_lvd1689m_cls": [0.275439, -0.261353, 0.067772, 0.049936, -0.158747],
        "vit7b16_lvd1689m_patch": [0.044442, -0.052542, 0.070777, -0.065111, -0.026546],
        "vitl16_sat493m_cls": [-0.33235, 0.34052, -0.22087, 0.21434, 0.09003],
        "vitl16_sat493m_patch": [0.18488, 0.30309, -0.20689, 0.12848, 0.06207],
        "vit7b16_sat493m_cls": [-0.19779, 0.11819, -0.00581, -0.21055, -0.03971],
        "vit7b16_sat493m_patch": [-0.12423, 0.07879, -0.10057, 0.02835, -0.11727],
    }

    model_name = args.model_name
    config = get_dinov3_config(model_name)

    model = DINOv3ViTModel(config).eval()
    state_dict_path = hf_hub_download(repo_id=HUB_MODELS[model_name], filename=HUB_CHECKPOINTS[model_name])
    original_state_dict = torch.load(state_dict_path, mmap=True)

    original_state_dict = split_qkv(original_state_dict)
    original_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(original_keys)

    converted_state_dict = {}
    for key in original_keys:
        new_key = new_keys[key]
        weight_tensor = original_state_dict[key]

        if "bias_mask" in key or "attn.k_proj.bias" in key or "local_cls_norm" in key:
            continue
        if "embeddings.mask_token" in new_key:
            weight_tensor = weight_tensor.unsqueeze(1)
        if "inv_freq" in new_key:
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
    last_layer_patch_tokens = model_output.last_hidden_state[:, config.num_register_tokens + 1 :]

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
        default="vith16plus_lvd1689m",
        type=str,
        choices=[
            "vits16_lvd1689m",
            "vits16plus_lvd1689m",
            "vitb16_lvd1689m",
            "vitl16_lvd1689m",
            "vitl16_sat493m",
            "vith16plus_lvd1689m",
            "vit7b16_lvd1689m",
            "vit7b16_sat493m",
        ],
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
