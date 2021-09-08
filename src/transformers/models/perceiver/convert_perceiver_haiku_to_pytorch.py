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
import pickle
from pathlib import Path

import torch
import haiku as hk

from transformers import PerceiverConfig, PerceiverModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


@torch.no_grad()
def convert_perceiver_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """

    # define default Perceiver configuration
    config = PerceiverConfig()

    # load original parameters
    with open("language_perceiver_io_bytes.pickle", "rb") as f:
        params = pickle.loads(f.read())

    # TODO: rename keys

    # load HuggingFace model
    model = PerceiverModel(config)
    model.eval()
    model.load_state_dict(state_dict)

    # Verify outputs

    # Finally, save files
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://storage.googleapis.com/perceiver_io/language_perceiver_io_bytes.pickle",
        type=str,
        help="URL of the Perceiver checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_perceiver_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
