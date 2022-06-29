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
"""Convert ConvNextMaskRCNN checkpoints from the mmdetection repository.

URL: https://github.com/open-mmlab/mmdetection"""


import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import ConvNextMaskRCNNConfig, ConvNextMaskRCNNForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnext_maskrcnn_config():
    config = ConvNextMaskRCNNConfig()

    config.num_labels = 80
    repo_id = "datasets/huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name):
    if "backbone.norm" in name:
        name = name.replace("backbone.norm", "convnext.encoder.layernorms.")
    if "backbone" in name:
        name = name.replace("backbone", "convnext")
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
        # convnext.stages.0.0. for instance should be renamed to convnext.stages.0.layers.0.
        name = name[: len("convnext.stages.0")] + ".layers" + name[len("convnext.stages.0") :]
        name = name
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    if "depthwise_conv" in name:
        name = name.replace("depthwise_conv", "dwconv")
    if "pointwise_conv" in name:
        name = name.replace("pointwise_conv", "pwconv")
    if "norm" in name and "layernorm" not in name:
        name = name.replace("norm", "layernorm")
    if "gamma" in name:
        name = name.replace("gamma", "layer_scale_parameter")
    if "convnext.layernorm" in name:
        name = name.replace("layernorm", "encoder.layernorms.")
    # neck (simply remove "conv" attribute due to use of `ConvModule` in mmdet)
    if "lateral" in name or "fpn" in name or "mask_head" in name:
        if "conv.weight" in name:
            name = name.replace("conv.weight", "weight")
        if "conv.bias" in name:
            name = name.replace("conv.bias", "bias")

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_convnext_maskrcnn_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ConvNextMaskRCNN structure.
    """

    # define ConvNextMaskRCNN configuration based on URL
    config = get_convnext_maskrcnn_config()
    # load original state_dict
    state_dict = torch.load(checkpoint_path)["state_dict"]
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val

    # load HuggingFace model
    model = ConvNextMaskRCNNForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    pixel_values = transform(image).unsqueeze(0)

    outputs = model(pixel_values, output_hidden_states=True)

    # for i in outputs.hidden_states[1:]:
    #     print(i.shape)

    expected_slice = torch.tensor(
        [[-0.0836, -0.1298, -0.1237], [-0.0743, -0.1090, -0.0873], [-0.0231, 0.0851, 0.0792]]
    )
    assert torch.allclose(outputs.hidden_states[-1][0, 0, :3, :3], expected_slice, atol=1e-3)
    print("Looks ok!")
    # assert logits.shape == expected_shape

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and feature extractor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model_name = "convnext-tiny-maskrcnn"
        model.push_to_hub(model_name, organization="nielsr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/home/niels/checkpoints/convnext_maskrcnn/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth",
        required=False,
        type=str,
        help="Path to the original ConvNextMaskRCNN checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        required=False,
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_convnext_maskrcnn_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
