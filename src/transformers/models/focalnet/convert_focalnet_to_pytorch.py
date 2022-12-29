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
"""Convert FocalNet checkpoints from the original repository. URL: https://github.com/microsoft/FocalNet/tree/main"""

import argparse
import json

import torch
from PIL import Image

from huggingface_hub import hf_hub_download

from torchvision import transforms

import requests
from transformers import AutoImageProcessor, FocalNetConfig, FocalNetForImageClassification


def get_focalnet_config(model_name):
    config = FocalNetConfig()

    if "tiny" in model_name:
        embed_dim = 96
        depths = [2, 2, 6, 2]
        focal_levels = [2, 2, 2, 2]
        focal_windows = [3, 3, 3, 3]
    elif "small" in model_name:
        embed_dim = 96
        depths = (2, 2, 18, 2)
    elif "base" in model_name:
        embed_dim = 128
        depths = (2, 2, 18, 2)
    else:
        embed_dim = 192
        depths = (2, 2, 18, 2)

    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    config.embed_dim = embed_dim
    config.depths = depths
    config.focal_levels = focal_levels
    config.focal_windows = focal_windows

    return config


def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder." + name
    if "downsample.proj" in name:
        name = name.replace("downsample.proj", "downsample.projection")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "focalnet." + name

    return name


def convert_focalnet_checkpoint(model_name, pytorch_dump_folder_path):
    model_name_to_url = {
        "focalnet-tiny": (
            "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth"
        ),
    }

    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val

    config = get_focalnet_config(model_name)
    model = FocalNetForImageClassification(config)
    model.eval()

    # load state dict
    model.load_state_dict(state_dict)

    # verify conversion
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    image = Image.open(requests.get(url, stream=True).raw)
    #inputs = processor(images=image, return_tensors="pt")

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pixel_values = image_transforms(image).unsqueeze(0)

    outputs = model(pixel_values)

    predicted_class_idx = outputs.logits.argmax(-1).item()
    print("Predicted class:", predicted_class_idx)

    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="focalnet-tiny",
        type=str,
        help="Name of the FocalNet model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_focalnet_checkpoint(args.model_name, args.pytorch_dump_folder_path)
