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
"""Convert ConvNeXTV2 checkpoints from the original repository.

URL: https://github.com/facebookresearch/ConvNeXt"""

import argparse
import json
import os

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnextv2_config(checkpoint_url):
    config = ConvNextV2Config()

    if "atto" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [40, 80, 160, 320]
    if "femto" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [48, 96, 192, 384]
    if "pico" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [64, 128, 256, 512]
    if "nano" in checkpoint_url:
        depths = [2, 2, 8, 2]
        hidden_sizes = [80, 160, 320, 640]
    if "tiny" in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "base" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if "large" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if "huge" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [352, 704, 1408, 2816]

    num_labels = 1000
    filename = "imagenet-1k-id2label.json"
    expected_shape = (1, 1000)

    repo_id = "huggingface/label-files"
    config.num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths

    return config, expected_shape


def rename_key(name):
    if "downsample_layers.0.0" in name:
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")
    if "downsample_layers.0.1" in name:
        name = name.replace("downsample_layers.0.1", "embeddings.norm")  # we rename to layernorm later on
    if "downsample_layers.1.0" in name:
        name = name.replace("downsample_layers.1.0", "stages.1.downsampling_layer.0")
    if "downsample_layers.1.1" in name:
        name = name.replace("downsample_layers.1.1", "stages.1.downsampling_layer.1")
    if "downsample_layers.2.0" in name:
        name = name.replace("downsample_layers.2.0", "stages.2.downsampling_layer.0")
    if "downsample_layers.2.1" in name:
        name = name.replace("downsample_layers.2.1", "stages.2.downsampling_layer.1")
    if "downsample_layers.3.0" in name:
        name = name.replace("downsample_layers.3.0", "stages.3.downsampling_layer.0")
    if "downsample_layers.3.1" in name:
        name = name.replace("downsample_layers.3.1", "stages.3.downsampling_layer.1")
    if "stages" in name and "downsampling_layer" not in name:
        # stages.0.0. for instance should be renamed to stages.0.layers.0.
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    if "head" in name:
        name = name.replace("head", "classifier")

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def convert_preprocessor(checkpoint_url):
    if "224" in checkpoint_url:
        size = 224
        crop_pct = 224 / 256
    elif "384" in checkpoint_url:
        size = 384
        crop_pct = None
    else:
        size = 512
        crop_pct = None

    return ConvNextImageProcessor(
        size=size,
        crop_pct=crop_pct,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        resample=PILImageResampling.BICUBIC,
    )


@torch.no_grad()
def convert_convnextv2_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ConvNeXTV2 structure.
    """
    print("Downloading original model from checkpoint...")
    # define ConvNeXTV2 configuration based on URL
    config, expected_shape = get_convnextv2_config(checkpoint_url)
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]

    print("Converting model parameters...")
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # add prefix to all keys expect classifier head
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "convnextv2." + key
        state_dict[key] = val

    # load HuggingFace model
    model = ConvNextV2ForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Check outputs on an image, prepared by ConvNextImageProcessor
    preprocessor = convert_preprocessor(checkpoint_url)
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")
    logits = model(**inputs).logits

    # note: the logits below were obtained without center cropping
    if checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.3930, 0.1747, -0.5246, 0.4177, 0.4295])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.1727, -0.5341, -0.7818, -0.4745, -0.6566])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.0333, 0.1563, -0.9137, 0.1054, 0.0381])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.1744, -0.1555, -0.0713, 0.0950, -0.1431])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt":
        expected_logits = torch.tensor([0.9996, 0.1966, -0.4386, -0.3472, 0.6661])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.2553, -0.6708, -0.1359, 0.2518, -0.2488])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.0673, -0.5627, -0.3753, -0.2722, 0.0178])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.6377, -0.7458, -0.2150, 0.1184, -0.0597])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt":
        expected_logits = torch.tensor([1.0799, 0.2322, -0.8860, 1.0219, 0.6231])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt":
        expected_logits = torch.tensor([0.3766, 0.4917, -1.1426, 0.9942, 0.6024])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt":
        expected_logits = torch.tensor([0.4220, -0.6919, -0.4317, -0.2881, -0.6609])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt":
        expected_logits = torch.tensor([0.1082, -0.8286, -0.5095, 0.4681, -0.8085])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt":
        expected_logits = torch.tensor([-0.2419, -0.6221, 0.2176, -0.0980, -0.7527])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt":
        expected_logits = torch.tensor([0.0391, -0.4371, 0.3786, 0.1251, -0.2784])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt":
        expected_logits = torch.tensor([-0.0504, 0.5636, -0.1729, -0.6507, -0.3949])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt":
        expected_logits = torch.tensor([0.3560, 0.9486, 0.3149, -0.2667, -0.5138])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt":
        expected_logits = torch.tensor([-0.2469, -0.4550, -0.5853, -0.0810, 0.0309])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt":
        expected_logits = torch.tensor([-0.3090, 0.0802, -0.0682, -0.1979, -0.2826])
    else:
        raise ValueError(f"Unknown URL: {checkpoint_url}")

    assert torch.allclose(logits[0, :5], expected_logits, atol=1e-3)
    assert logits.shape == expected_shape
    print("Model outputs match the original results!")

    if save_model:
        print("Saving model to local...")
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    model_name = "convnextv2"
    if "atto" in checkpoint_url:
        model_name += "-atto"
    if "femto" in checkpoint_url:
        model_name += "-femto"
    if "pico" in checkpoint_url:
        model_name += "-pico"
    if "nano" in checkpoint_url:
        model_name += "-nano"
    elif "tiny" in checkpoint_url:
        model_name += "-tiny"
    elif "base" in checkpoint_url:
        model_name += "-base"
    elif "large" in checkpoint_url:
        model_name += "-large"
    elif "huge" in checkpoint_url:
        model_name += "-huge"
    if "22k" in checkpoint_url and "1k" not in checkpoint_url:
        model_name += "-22k"
    elif "22k" in checkpoint_url and "1k" in checkpoint_url:
        model_name += "-22k-1k"
    elif "1k" in checkpoint_url:
        model_name += "-1k"
    if "224" in checkpoint_url:
        model_name += "-224"
    elif "384" in checkpoint_url:
        model_name += "-384"
    elif "512" in checkpoint_url:
        model_name += "-512"

    if push_to_hub:
        print(f"Pushing {model_name} to the hub...")
        model.push_to_hub(model_name)
        preprocessor.push_to_hub(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",
        type=str,
        help="URL of the original ConvNeXTV2 checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_convnextv2_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
