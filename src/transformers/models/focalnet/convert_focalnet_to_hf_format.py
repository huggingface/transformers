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
"""Convert FocalNet checkpoints from the original repository. URL: https://github.com/microsoft/FocalNet/tree/main"""

import argparse
import json

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling


def get_focalnet_config(model_name):
    depths = [2, 2, 6, 2] if "tiny" in model_name else [2, 2, 18, 2]
    use_conv_embed = True if "large" in model_name or "huge" in model_name else False
    use_post_layernorm = True if "large" in model_name or "huge" in model_name else False
    use_layerscale = True if "large" in model_name or "huge" in model_name else False

    if "large" in model_name or "xlarge" in model_name or "huge" in model_name:
        if "fl3" in model_name:
            focal_levels = [3, 3, 3, 3]
            focal_windows = [5, 5, 5, 5]
        elif "fl4" in model_name:
            focal_levels = [4, 4, 4, 4]
            focal_windows = [3, 3, 3, 3]

    if "tiny" in model_name or "small" in model_name or "base" in model_name:
        focal_windows = [3, 3, 3, 3]
        if "lrf" in model_name:
            focal_levels = [3, 3, 3, 3]
        else:
            focal_levels = [2, 2, 2, 2]

    if "tiny" in model_name:
        embed_dim = 96
    elif "small" in model_name:
        embed_dim = 96
    elif "base" in model_name:
        embed_dim = 128
    elif "large" in model_name:
        embed_dim = 192
    elif "xlarge" in model_name:
        embed_dim = 256
    elif "huge" in model_name:
        embed_dim = 352

    # set label information
    repo_id = "huggingface/label-files"
    if "large" in model_name or "huge" in model_name:
        filename = "imagenet-22k-id2label.json"
    else:
        filename = "imagenet-1k-id2label.json"

    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    config = FocalNetConfig(
        embed_dim=embed_dim,
        depths=depths,
        focal_levels=focal_levels,
        focal_windows=focal_windows,
        use_conv_embed=use_conv_embed,
        id2label=id2label,
        label2id=label2id,
        use_post_layernorm=use_post_layernorm,
        use_layerscale=use_layerscale,
    )

    return config


def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder." + name
    if "encoder.layers" in name:
        name = name.replace("encoder.layers", "encoder.stages")
    if "downsample.proj" in name:
        name = name.replace("downsample.proj", "downsample.projection")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "modulation.f.weight" in name or "modulation.f.bias" in name:
        name = name.replace("modulation.f", "modulation.projection_in")
    if "modulation.h.weight" in name or "modulation.h.bias" in name:
        name = name.replace("modulation.h", "modulation.projection_context")
    if "modulation.proj.weight" in name or "modulation.proj.bias" in name:
        name = name.replace("modulation.proj", "modulation.projection_out")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "focalnet." + name

    return name


def convert_focalnet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    # fmt: off
    model_name_to_url = {
        "focalnet-tiny": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth",
        "focalnet-tiny-lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_lrf.pth",
        "focalnet-small": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_srf.pth",
        "focalnet-small-lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_lrf.pth",
        "focalnet-base": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_srf.pth",
        "focalnet-base-lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_lrf.pth",
        "focalnet-large-lrf-fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384.pth",
        "focalnet-large-lrf-fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth",
        "focalnet-xlarge-lrf-fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384.pth",
        "focalnet-xlarge-lrf-fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384_fl4.pth",
    }
    # fmt: on

    checkpoint_url = model_name_to_url[model_name]
    print("Checkpoint URL: ", checkpoint_url)
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

    processor = BitImageProcessor(
        do_resize=True,
        size={"shortest_edge": 256},
        resample=PILImageResampling.BILINEAR,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
    )
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    image_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    original_pixel_values = image_transforms(image).unsqueeze(0)

    # verify pixel_values
    assert torch.allclose(inputs.pixel_values, original_pixel_values, atol=1e-4)

    outputs = model(**inputs)

    predicted_class_idx = outputs.logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    print("First values of logits:", outputs.logits[0, :3])

    if model_name == "focalnet-tiny":
        expected_slice = torch.tensor([0.2166, -0.4368, 0.2191])
    elif model_name == "focalnet-tiny-lrf":
        expected_slice = torch.tensor([1.1669, 0.0125, -0.1695])
    elif model_name == "focalnet-small":
        expected_slice = torch.tensor([0.4917, -0.0430, 0.1341])
    elif model_name == "focalnet-small-lrf":
        expected_slice = torch.tensor([-0.2588, -0.5342, -0.2331])
    elif model_name == "focalnet-base":
        expected_slice = torch.tensor([-0.1655, -0.4090, -0.1730])
    elif model_name == "focalnet-base-lrf":
        expected_slice = torch.tensor([0.5306, -0.0483, -0.3928])
    assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor of {model_name} to the hub...")
        model.push_to_hub(f"{model_name}")
        processor.push_to_hub(f"{model_name}")


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
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub.",
    )

    args = parser.parse_args()
    convert_focalnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
