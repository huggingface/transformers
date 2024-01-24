# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Perceiver checkpoints originally implemented in Haiku."""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import VMambaForImageClassification
from transformers.models.vmamba.configuration_vmamba import VMambaConfig
from transformers.models.vmamba.image_processing_vmamba import VMambaImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def prepare_img():
    # We will verify our results on an image of a dog
    url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def rename_keys(state_dict, architecture):
    new_state_dict = {}
    for name in state_dict:
        param = state_dict[name]
        if not name.startswith("head"):
            new_state_dict["vmamba." + name] = param
        else:
            new_state_dict[name] = param

    return new_state_dict


@torch.no_grad()
def convert_vmamba_checkpoint(pytorch_file, pytorch_dump_folder_path, architecture="image_classification"):
    """
    Copy/paste/tweak model's weights to our VMamba structure.
    """

    # load parameters
    checkpoint = torch.load(pytorch_file)

    # state = None
    # if isinstance(checkpoint, dict) and architecture in [
    #     "image_classification",
    #     "image_classification_fourier",
    #     "image_classification_conv",
    # ]:
    #     # the image classification_conv checkpoint also has batchnorm states (running_mean and running_var)
    #     params = checkpoint["params"]
    #     state = checkpoint["state"]
    # else:
    #     params = checkpoint
    # params = checkpoint

    # # turn into initial state dict
    # state_dict = {}
    # for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
    #     for param_name, param in parameters.items():
    #         state_dict[scope_name + "/" + param_name] = param

    # if state is not None:
    #     # add state variables
    #     for scope_name, parameters in hk.data_structures.to_mutable_dict(state).items():
    #         for param_name, param in parameters.items():
    #             state_dict[scope_name + "/" + param_name] = param

    state_dict = checkpoint["model"]

    # # rename keys
    state_dict = rename_keys(state_dict, architecture=architecture)

    # load HuggingFace model
    config = VMambaConfig()

    repo_id = "huggingface/label-files"
    if architecture == "image_classification":
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        config.qk_channels = None
        config.v_channels = None
        # set labels
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

        config.image_size = 224
        model = VMambaForImageClassification(config)
    # elif architecture == "optical_flow":
    #     config.num_latents = 2048
    #     config.d_latents = 512
    #     config.d_model = 322
    #     config.num_blocks = 1
    #     config.num_self_attends_per_block = 24
    #     config.num_self_attention_heads = 16
    #     config.num_cross_attention_heads = 1
    #     model = PerceiverForOpticalFlow(config)
    else:
        raise ValueError(f"Architecture {architecture} not supported")
    model.eval()

    # load weights
    model.load_state_dict(state_dict)

    # prepare dummy input
    # input_mask = None
    if architecture == "image_classification":
        image_processor = VMambaImageProcessor()
        image = prepare_img()
        encoding = image_processor(image, return_tensors="pt")
        inputs = encoding.pixel_values
    elif architecture == "optical_flow":
        inputs = torch.randn(1, 2, 27, 368, 496)
    elif architecture == "multimodal_autoencoding":
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        inputs = {"image": images, "audio": audio, "label": torch.zeros((images.shape[0], 700))}

    # forward pass
    outputs = model(inputs=inputs)

    # verify logits
    # if not isinstance(logits, dict):
    #     print("Shape of logits:", logits.shape)
    # else:
    #     for k, v in logits.items():
    #         print(f"Shape of logits of modality {k}", v.shape)

    if architecture == "image_classification":
        print("Predicted class:", model.config.id2label[outputs.argmax(-1).item()])

    # Finally, save files
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pytorch file of a VMamba checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )
    parser.add_argument(
        "--architecture",
        default="image_classification",
        type=str,
        help="""
        Architecture, provided as a string. One of 'MLM', 'image_classification', image_classification_fourier',
        image_classification_fourier', 'optical_flow' or 'multimodal_autoencoding'.
        """,
    )

    args = parser.parse_args()
    convert_vmamba_checkpoint(args.pytorch_file, args.pytorch_dump_folder_path, args.architecture)
