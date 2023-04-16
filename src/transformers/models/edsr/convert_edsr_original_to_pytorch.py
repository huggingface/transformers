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
"""Convert EDSR checkpoints from the original repository. URL: https://github.com/sanghyun-son/EDSR-PyTorch
"""

import argparse

import requests
import torch
import torchvision
import numpy as np
from PIL import Image

from transformers import EDSRConfig, EDSRForImageSuperResolution, EDSRImageProcessor


def get_edsr_config(checkpoint_url):
    config = EDSRConfig()

    print(checkpoint_url)
    if "edsr_baseline_x2" in checkpoint_url:
        config.n_resblocks = 16
        config.n_feats = 64
        config.scale = 2
    elif "edsr_baseline_x3" in checkpoint_url:
        config.n_resblocks = 16
        config.n_feats = 64
        config.scale = 3
    elif "edsr_baseline_x4" in checkpoint_url:
        config.n_resblocks = 16
        config.n_feats = 64
        config.scale = 4
    elif "edsr_x2" in checkpoint_url:
        config.n_resblocks = 32
        config.n_feats = 256
        config.scale = 2
    elif "edsr_x3" in checkpoint_url:
        config.n_resblocks = 32
        config.n_feats = 256
        config.scale = 3
    elif "edsr_x4" in checkpoint_url:
        config.n_resblocks = 32
        config.n_feats = 256
        config.scale = 4

    return config


def rename_key(name):
    if "model" in name:
        name = name.replace("model", "edsr_model")
    if "head" in name:
        name = name.replace("head", "edsr_head")
    if "body" in name:
        name = name.replace("body", "edsr_body")
    if "tail" in name:
        name = name.replace("tail", "upsampler")

    return name

# def load_sample_image():
#     url = "https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true"
#     image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     return image

def load_sample_image():
    SAMPLE_IMAGE_PATH = "/home/sri/Music/Set14/image_SRF_2/"
    bicubic_path = SAMPLE_IMAGE_PATH + "img_001_SRF_2_bicubic.png"
    gt_path = SAMPLE_IMAGE_PATH + "img_001_SRF_2_HR.png"

    bicubic_image = torchvision.io.read_image(bicubic_path).float().unsqueeze(0)
    bicubic_image = torch.nn.functional.interpolate(bicubic_image, scale_factor=0.5, mode="bilinear")
    # print(bicubic_image.shape)
    return bicubic_image.float()

def save_image(out_tensor):
    numpy_image = out_tensor.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
    out_image = torchvision.transforms.functional.to_pil_image(np.uint8(numpy_image))
    save_path = "model_out_pil.png"
    out_image.save()
    print



@torch.no_grad()
def convert_edsr_checkpoint(checkpoint_url: str, pytorch_dump_folder_path: str, push_to_hub: bool):
    config = get_edsr_config(checkpoint_url)
    name_to_url = {
        "edsr_baseline_x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
        "edsr_baseline_x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt",
        "edsr_baseline_x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
        "edsr_x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
        "edsr_x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
        "edsr_x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
    }

    model = EDSRForImageSuperResolution(config)
    model.eval()

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # for key in state_dict.copy().keys():
    #     val = state_dict.pop(key)
    #     new_key_name = key if "upsampler" in key else "model."+key
    #     state_dict[rename_key(new_key_name)] = val

    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # add prefix to all keys except the head
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("upsampler"):
            key = "edsr_model." + key
        state_dict[key] = val

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    print("Missing:", missing_keys)
    print("Unexpected:", unexpected_keys)

    if len(missing_keys) > 0:
        raise ValueError("Missing keys when converting: {}".format(missing_keys))
    for key in unexpected_keys:
        if not ("relative_position_index" in key or "relative_coords_table" in key or "self_mask" in key):
            raise ValueError(f"Unexpected key {key} in state_dict")

    # verify values
    processor = EDSRImageProcessor()
    pixel_values = load_sample_image()

    # pixel_values = processor(image).pixel_values

    # pixel_values = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
    print(pixel_values.shape)
    outputs = model(pixel_values)
    print(outputs)
    save_image(outputs.reconstruction)
    # assert values
    expected_slice = torch.tensor(
        [[65.5774, 47.5999, 34.9567],
        [79.2120, 63.8901, 49.3842],
        [86.0216, 82.4572, 69.4604]]
    )
    expected_shape = torch.Size([1, 3, pixel_values.shape[-2] * config.upscale, pixel_values.shape[-1] * config.upscale])
    if "edsr_baseline_x2" in checkpoint_url:
        expected_slice = torch.tensor(
            [[65.5774, 47.5999, 34.9567],
            [79.2120, 63.8901, 49.3842],
            [86.0216, 82.4572, 69.4604]]
        )
    print("Shape of reconstruction:", outputs.reconstruction.shape)
    print("Actual values of the reconstruction:\n", outputs.reconstruction[0, 0, :3, :3])

    assert (
        outputs.reconstruction.shape == expected_shape
    ), f"Shape of reconstruction should be {expected_shape}, but is {outputs.reconstruction.shape}"
    assert torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-3)
    print("Looks ok!")


# def convert_swin2sr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
#     get_edsr_config(checkpoint_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
        type=str,
        help="URL of the original EDSR checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")

    args = parser.parse_args()
    convert_edsr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
