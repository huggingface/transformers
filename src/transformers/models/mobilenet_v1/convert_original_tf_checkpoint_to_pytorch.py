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
"""Convert MobileNetV1 checkpoints from the tensorflow/models library."""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import (
    MobileNetV1Config,
    MobileNetV1FeatureExtractor,
    MobileNetV1ForImageClassification,
    load_tf_weights_in_mobilenet_v1,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_mobilenet_v1_config(model_name):
    config = MobileNetV1Config(layer_norm_eps=0.001)

    # Size of the architecture
    if "1.0" in model_name:
        config.depth_multiplier = 1.0
    # elif "mobilenet_v1_xs" in model_name:
    #     config.hidden_sizes = [96, 120, 144]
    #     config.neck_hidden_sizes = [16, 32, 48, 64, 80, 96, 384]
    # elif "mobilenet_v1_xxs" in model_name:
    #     config.hidden_sizes = [64, 80, 96]
    #     config.neck_hidden_sizes = [16, 16, 24, 48, 64, 80, 320]
    #     config.hidden_dropout_prob = 0.05
    #     config.expand_ratio = 2.0
    # MIH?

    # TODO: use a regexp?
    config.image_size = 224

    # The TensorFlow version of MobileNetV1 predicts 1001 classes instead of
    # the usual 1000. The first class (index 0) is "background".
    config.num_labels = 1001
    filename = "imagenet-1k-id2label.json"
    repo_id = "datasets/huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k) + 1: v for k, v in id2label.items()}
    id2label[0] = "background"
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileNetV1 structure.
    """
    config = get_mobilenet_v1_config(model_name)

    # Load 🤗 model
    model = MobileNetV1ForImageClassification(config).eval()

    # Load weights from TensorFlow checkpoint
    load_tf_weights_in_mobilenet_v1(model, config, checkpoint_path)

    # Check outputs on an image, prepared by MobileNetV1FeatureExtractor
    feature_extractor = MobileNetV1FeatureExtractor(crop_size=config.image_size, size=config.image_size + 32)
#MIH
    # encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    # outputs = model(**encoding)
    # logits = outputs.logits

    # if model_name.startswith("deeplabv3_"):
    #     assert logits.shape == (1, 21, 32, 32)

    #     if model_name == "deeplabv3_mobilenet_v1_s":
    #         expected_logits = torch.tensor(
    #             [
    #                 [[6.2065, 6.1292, 6.2070], [6.1079, 6.1254, 6.1747], [6.0042, 6.1071, 6.1034]],
    #                 [[-6.9253, -6.8653, -7.0398], [-7.3218, -7.3983, -7.3670], [-7.1961, -7.2482, -7.1569]],
    #                 [[-4.4723, -4.4348, -4.3769], [-5.3629, -5.4632, -5.4598], [-5.1587, -5.3402, -5.5059]],
    #             ]
    #         )
    #     elif model_name == "deeplabv3_mobilenet_v1_xs":
    #         expected_logits = torch.tensor(
    #             [
    #                 [[5.4449, 5.5733, 5.6314], [5.1815, 5.3930, 5.5963], [5.1656, 5.4333, 5.4853]],
    #                 [[-9.4423, -9.7766, -9.6714], [-9.1581, -9.5720, -9.5519], [-9.1006, -9.6458, -9.5703]],
    #                 [[-7.7721, -7.3716, -7.1583], [-8.4599, -8.0624, -7.7944], [-8.4172, -7.8366, -7.5025]],
    #             ]
    #         )
    #     elif model_name == "deeplabv3_mobilenet_v1_xxs":
    #         expected_logits = torch.tensor(
    #             [
    #                 [[6.9811, 6.9743, 7.3123], [7.1777, 7.1931, 7.3938], [7.5633, 7.8050, 7.8901]],
    #                 [[-10.5536, -10.2332, -10.2924], [-10.2336, -9.8624, -9.5964], [-10.8840, -10.8158, -10.6659]],
    #                 [[-3.4938, -3.0631, -2.8620], [-3.4205, -2.8135, -2.6875], [-3.4179, -2.7945, -2.8750]],
    #             ]
    #         )
    #     else:
    #         raise ValueError(f"Unknown model_name: {model_name}")

    #     assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    # else:
    #     assert logits.shape == (1, 1000)

    #     if model_name == "mobilenet_v1_s":
    #         expected_logits = torch.tensor([-0.9866, 0.2392, -1.1241])
    #     elif model_name == "mobilenet_v1_xs":
    #         expected_logits = torch.tensor([-2.4761, -0.9399, -1.9587])
    #     elif model_name == "mobilenet_v1_xxs":
    #         expected_logits = torch.tensor([-1.9364, -1.2327, -0.4653])
    #     else:
    #         raise ValueError(f"Unknown model_name: {model_name}")

    #     assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_mapping = {
            "mobilenet_v1_s": "mobilenet_v1-small",
            "mobilenet_v1_xs": "mobilenet_v1-x-small",
            "mobilenet_v1_xxs": "mobilenet_v1-xx-small",
            "deeplabv3_mobilenet_v1_s": "deeplabv3-mobilenet_v1-small",
            "deeplabv3_mobilenet_v1_xs": "deeplabv3-mobilenet_v1-x-small",
            "deeplabv3_mobilenet_v1_xxs": "deeplabv3-mobilenet_v1-xx-small",
        }

        print("Pushing to the hub...")
        model_name = model_mapping[model_name]
        feature_extractor.push_to_hub(model_name, organization="apple")
        model.push_to_hub(model_name, organization="apple")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="mobilenet_v1_1.0_224",
        type=str,
        help=(
            "Name of the MobileNetV1 model you'd like to convert. Should be one of 'mobilenet_v1_1.0_224'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original TensorFlow checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
