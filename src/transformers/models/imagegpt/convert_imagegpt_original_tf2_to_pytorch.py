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
"""Convert OpenAI Image GPT checkpoints."""


import argparse

import torch

from transformers import ImageGPTConfig, ImageGPTForCausalLM, load_tf_weights_in_imagegpt
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


logging.set_verbosity_info()


def convert_imagegpt_checkpoint_to_pytorch(imagegpt_checkpoint_path, model_size, pytorch_dump_folder_path):
    # Construct configuration depending on size
    MODELS = {"small": (512, 8, 24), "medium": (1024, 8, 36), "large": (1536, 16, 48)}
    n_embd, n_head, n_layer = MODELS[model_size]  # set model hyperparameters
    config = ImageGPTConfig(n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model = ImageGPTForCausalLM(config)

    # Load weights from numpy
    load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path)

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"Save configuration file to {pytorch_config_dump_path}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--imagegpt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--model_size",
        default=None,
        type=str,
        required=True,
        help="Size of the model (can be either 'small', 'medium' or 'large').",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_imagegpt_checkpoint_to_pytorch(
        args.imagegpt_checkpoint_path, args.model_size, args.pytorch_dump_folder_path
    )
