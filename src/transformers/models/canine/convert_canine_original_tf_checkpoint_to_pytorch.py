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
"""Convert CANINE checkpoint."""

import argparse

from transformers import CanineConfig, CanineModel, CanineTokenizer, load_tf_weights_in_canine
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    # Initialize PyTorch model
    config = CanineConfig()
    model = CanineModel(config)
    model.eval()

    print(f"Building PyTorch model from configuration: {config}")

    # Load weights from tf checkpoint
    load_tf_weights_in_canine(model, config, tf_checkpoint_path)

    # Save pytorch-model (weights and configuration)
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # Save tokenizer files
    tokenizer = CanineTokenizer()
    print(f"Save tokenizer files to {pytorch_dump_path}")
    tokenizer.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint. Should end with model.ckpt",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to a folder where the PyTorch model will be placed.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)
