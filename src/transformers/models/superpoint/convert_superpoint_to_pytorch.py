# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import argparse
import os

import requests
import torch
from PIL import Image

from transformers import SuperPointConfig, SuperPointImageProcessor, SuperPointModel


def get_superpoint_config():
    config = SuperPointConfig(
        hidden_sizes=[64, 64, 128, 128, 256],
        descriptor_dim=256,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        nms_radius=4,
        border_removal_distance=4,
        initializer_range=0.02,
    )

    return config


def create_rename_keys(config, state_dict):
    rename_keys = []

    # Encoder weights
    rename_keys.append(("conv1a.weight", "encoder.conv1a.weight"))
    rename_keys.append(("conv1b.weight", "encoder.conv1b.weight"))
    rename_keys.append(("conv2a.weight", "encoder.conv2a.weight"))
    rename_keys.append(("conv2b.weight", "encoder.conv2b.weight"))
    rename_keys.append(("conv3a.weight", "encoder.conv3a.weight"))
    rename_keys.append(("conv3b.weight", "encoder.conv3b.weight"))
    rename_keys.append(("conv4a.weight", "encoder.conv4a.weight"))
    rename_keys.append(("conv4b.weight", "encoder.conv4b.weight"))
    rename_keys.append(("conv1a.bias", "encoder.conv1a.bias"))
    rename_keys.append(("conv1b.bias", "encoder.conv1b.bias"))
    rename_keys.append(("conv2a.bias", "encoder.conv2a.bias"))
    rename_keys.append(("conv2b.bias", "encoder.conv2b.bias"))
    rename_keys.append(("conv3a.bias", "encoder.conv3a.bias"))
    rename_keys.append(("conv3b.bias", "encoder.conv3b.bias"))
    rename_keys.append(("conv4a.bias", "encoder.conv4a.bias"))
    rename_keys.append(("conv4b.bias", "encoder.conv4b.bias"))

    # Keypoint Decoder weights
    rename_keys.append(("convPa.weight", "keypoint_decoder.convSa.weight"))
    rename_keys.append(("convPb.weight", "keypoint_decoder.convSb.weight"))
    rename_keys.append(("convPa.bias", "keypoint_decoder.convSa.bias"))
    rename_keys.append(("convPb.bias", "keypoint_decoder.convSb.bias"))

    # Descriptor Decoder weights
    rename_keys.append(("convDa.weight", "descriptor_decoder.convDa.weight"))
    rename_keys.append(("convDb.weight", "descriptor_decoder.convDb.weight"))
    rename_keys.append(("convDa.bias", "descriptor_decoder.convDa.bias"))
    rename_keys.append(("convDb.bias", "descriptor_decoder.convDb.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_imgs():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im1 = Image.open(requests.get(url, stream=True).raw)
    url = "http://images.cocodataset.org/test-stuff2017/000000004016.jpg"
    im2 = Image.open(requests.get(url, stream=True).raw)
    return [im1, im2]


@torch.no_grad()
def convert_superpoint_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    TODO docs
    """

    print("Downloading original model from checkpoint...")
    config = get_superpoint_config()

    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    print("Converting model parameters...")
    rename_keys = create_rename_keys(config, original_state_dict)
    new_state_dict = original_state_dict.copy()
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    model = SuperPointModel(config)
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Successfully loaded weights in the model")

    preprocessor = SuperPointImageProcessor()
    inputs = preprocessor(images=prepare_imgs(), return_tensors="pt")
    outputs = model(**inputs)

    torch.count_nonzero(outputs.mask[0])
    expected_keypoints_shape = (2, 830, 2)
    expected_scores_shape = (
        2,
        830,
    )
    expected_descriptors_shape = (2, 830, 256)

    expected_keypoints_values = torch.tensor([[480.0, 9.0], [494.0, 9.0], [489.0, 16.0]])
    expected_scores_values = torch.tensor([0.0064, 0.0140, 0.0595, 0.0728, 0.5170, 0.0175, 0.1523, 0.2055, 0.0336])
    expected_descriptors_value = torch.tensor(-0.1096)

    assert outputs.keypoints.shape == expected_keypoints_shape
    assert outputs.scores.shape == expected_scores_shape
    assert outputs.descriptors.shape == expected_descriptors_shape

    assert torch.allclose(outputs.keypoints[0, :3], expected_keypoints_values, atol=1e-3)
    assert torch.allclose(outputs.scores[0, :9], expected_scores_values, atol=1e-3)
    assert torch.allclose(outputs.descriptors[0, 0, 0], expected_descriptors_value, atol=1e-3)
    print("Model outputs match the original results!")

    if save_model:
        print("Saving model to local...")
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

        model_name = "superpoint"
        if push_to_hub:
            print(f"Pushing {model_name} to the hub...")
        model.push_to_hub(model_name)
        preprocessor.push_to_hub(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
        type=str,
        help="URL of the original SuperPoint checkpoint you'd like to convert.",
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
    convert_superpoint_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
