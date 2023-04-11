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

def load_sample_image():
    url = "https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

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

    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val


    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Missing:", missing_keys)
    print("Unexpected:", unexpected_keys)

    if len(missing_keys) > 0:
        raise ValueError("Missing keys when converting: {}".format(missing_keys))
    for key in unexpected_keys:
        if not ("relative_position_index" in key or "relative_coords_table" in key or "self_mask" in key):
            raise ValueError(f"Unexpected key {key} in state_dict")

    # verify values
    processor = EDSRImageProcessor()
    url = "https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/test/0853x4.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    pixel_values = processor(image).pixel_values

    outputs = model(pixel_values)

    # assert values
    if "r16f64x2" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor(
            [[-0.7087, -0.7138, -0.6721], [-0.8340, -0.8095, -0.7298], [-0.9149, -0.8414, -0.7940]]
        )
    print("Shape of reconstruction:", outputs.reconstruction.shape)
    print("Actual values of the reconstruction:", outputs.reconstruction[0, 0, :3, :3])

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
