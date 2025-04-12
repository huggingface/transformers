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
"""Convert ConvNext checkpoints from the original repository.

URL: https://github.com/facebookresearch/ConvNeXt"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnext_config(checkpoint_url):
    config = ConvNextConfig()

    if "tiny" in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "small" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "base" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if "large" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if "xlarge" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]

    if "1k" in checkpoint_url:
        num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        expected_shape = (1, 1000)
    else:
        num_labels = 21841
        filename = "imagenet-22k-id2label.json"
        expected_shape = (1, 21841)

    repo_id = "huggingface/label-files"
    config.num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    if "1k" not in checkpoint_url:
        # this dataset contains 21843 labels but the model only has 21841
        # we delete the classes as mentioned in https://github.com/google-research/big_transfer/issues/18
        del id2label[9205]
        del id2label[15027]
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths

    return config, expected_shape


def rename_key(name):
    if "stages" in name and "downsampling_layer" not in name:
        # stages.0.0. for instance should be renamed to stages.0.layers.0.
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    if "gamma" in name:
        name = name.replace("gamma", "layer_scale_parameter")
    if "head" in name:
        name = name.replace("head", "classifier")
    if "blocks" in name:
        name = name.replace("blocks.", "")
    if "conv_dw" in name:
        name = name.replace("conv_dw", "dwconv")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "pwconv1")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "pwconv2")
    if "layers.downsample" in name:
        name = name.replace("layers.downsample", "downsampling_layer")
    if "fc." in name:
        name = name.replace("fc.", "")
    if "classifier.layernorm" in name:
        name = name.replace("classifier.layernorm", "layernorm")
    if "stem.0" in name:
        name = name.replace("stem.0", "embeddings.patch_embeddings")
    if "stem.1" in name:
        name = name.replace("stem.1", "embeddings.layernorm")
    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_convnext_checkpoint(timm_model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our ConvNext structure.
    """

    # define default Convnext configuration
    config, expected_shape = get_convnext_config(timm_model_name)

    # load original model from timm
    timm_model = create_model(timm_model_name, pretrained=True)
    timm_model.eval()

    # load original state_dict from URL
    state_dict = timm_model.state_dict()

    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # add prefix to all keys expect classifier head
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if key not in ["classifier.weight", "classifier.bias"]:
            key = "convnext." + key
        state_dict[key] = val

    # load HuggingFace model
    model = ConvNextForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)

    # create image processor
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    timm_transforms = transform.transforms

    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # Check outputs on an image, prepared by ConvNextImageProcessor
    size = 384 if "384" in timm_model_name else 224
    image_processor = ConvNextImageProcessor(
        size={"shortest_edge": size},
    )
    image = prepare_img()
    timm_pixel_values = transform(image).unsqueeze(0)

    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    assert torch.allclose(timm_pixel_values, pixel_values)

    logits = model(pixel_values).logits

    # verify logits
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    print("Logits:", logits[0, :3])
    print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])
    timm_logits = timm_model(timm_pixel_values)
    assert timm_logits.shape == outputs.logits.shape
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    print("Looks ok!")

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)

    print("Pushing model to the hub...")
    model_name = "convnext"
    if "tiny" in timm_model_name:
        model_name += "-tiny"
    elif "small" in timm_model_name:
        model_name += "-small"
    elif "base" in timm_model_name:
        model_name += "-base"
    elif "xlarge" in timm_model_name:
        model_name += "-xlarge"
    elif "large" in timm_model_name:
        model_name += "-large"
    if "384" in timm_model_name:
        model_name += "-384"
    else:
        model_name += "-224"
    if "22k" in timm_model_name and "1k" not in timm_model_name:
        model_name += "-22k"
    if "22k" in timm_model_name and "1k" in timm_model_name:
        model_name += "-22k-1k"

    model.push_to_hub(
        repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
        organization="nielsr",
        commit_message="Add model",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timm_model_name",
        default="convnext_tiny.fb_in1k",
        type=str,
        help="Name of the ConvNext timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    args = parser.parse_args()
    convert_convnext_checkpoint(args.timm_model_name, args.pytorch_dump_folder_path, args.push_to_hub)
