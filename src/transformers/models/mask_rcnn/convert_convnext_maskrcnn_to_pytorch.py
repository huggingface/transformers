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
"""Convert ConvNeXt Mask R-CNN checkpoints from the mmdetection repository.

URL: https://github.com/open-mmlab/mmdetection"""


import argparse
import json
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ConvNextConfig, MaskRCNNConfig, MaskRCNNForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnext_maskrcnn_config():
    backbone_config = ConvNextConfig.from_pretrained("facebook/convnext-tiny-224", out_features=["stage1", "stage2", "stage3", "stage4"])
    config = MaskRCNNConfig(backbone_config=backbone_config)

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


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
        # backbone.stages.0.0. for instance should be renamed to backbone.stages.0.layers.0.
        name = name[: len("backbone.stages.0")] + ".layers" + name[len("backbone.stages.0") :]
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

    # backbone layernorms
    if "backbone.layernorm0" in name:
        name = name.replace("backbone.layernorm0", "backbone.hidden_states_norms.stage1")
    if "backbone.layernorm1" in name:
        name = name.replace("backbone.layernorm1", "backbone.hidden_states_norms.stage2")
    if "backbone.layernorm2" in name:
        name = name.replace("backbone.layernorm2", "backbone.hidden_states_norms.stage3")
    if "backbone.layernorm3" in name:
        name = name.replace("backbone.layernorm3", "backbone.hidden_states_norms.stage4")

    # neck (simply remove "conv" attribute due to use of `ConvModule` in mmdet)
    if "lateral" in name or "fpn" in name or "mask_head" in name:
        if "conv.weight" in name:
            name = name.replace("conv.weight", "weight")
        if "conv.bias" in name:
            name = name.replace("conv.bias", "bias")

    return name


@torch.no_grad()
def convert_convnext_maskrcnn_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our MaskRCNN structure.
    """

    # define MaskRCNN configuration based on URL
    config = get_convnext_maskrcnn_config()
    # load original state_dict
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["state_dict"]
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val

    # load HuggingFace model
    model = MaskRCNNForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    pixel_values = transform(image).unsqueeze(0)

    width, height = image.size
    pixel_values_height, pixel_values_width = pixel_values.shape[-2:]

    img_metas = [
        {
            "img_shape": pixel_values.shape[1:],
            "scale_factor": np.array([1.6671875, 1.6666666, 1.6671875, 1.6666666], dtype=np.float32),
            "ori_shape": (3, height, width),
        }
    ]

    outputs = model(pixel_values, img_metas=img_metas, output_hidden_states=True)

    # verify hidden states
    # expected_slice = torch.tensor(
    #     [[-0.0836, -0.1298, -0.1237], [-0.0743, -0.1090, -0.0873], [-0.0231, 0.0851, 0.0792]]
    # )
    # assert torch.allclose(outputs.hidden_states[-1][0, 0, :3, :3], expected_slice, atol=1e-3)

    # verify outputs
    expected_slice_logits = torch.tensor(
        [[-12.4785, -17.4976, -14.7001], [-10.9181, -16.7281, -13.2826], [-10.5053, -18.3817, -15.5554]],
    )
    expected_slice_boxes = torch.tensor(
        [[-0.8485, 0.6819, -1.1016], [1.4864, -0.1529, -1.2551], [0.0233, 0.4202, 0.2257]],
    )
    print("Logits:", outputs.logits[0, :3, :3])
    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)
    print("Looks ok!")

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
        "--checkpoint_url",
        default="https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth",
        required=False,
        type=str,
        help="Path to the original MaskRCNN checkpoint you'd like to convert.",
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
    convert_convnext_maskrcnn_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
