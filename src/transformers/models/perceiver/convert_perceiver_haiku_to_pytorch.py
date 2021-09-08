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
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch

import haiku as hk
import wget
from transformers import PerceiverConfig, PerceiverModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_haiku_weights_in_perceiver(model, state_dict):
    """Load haiku checkpoints in a PyTorch model."""
    names = []
    arrays = []
    for name, array in state_dict.items():
        if "~/" in name:
            name = name.replace("~/", "")
        print(f"Loading Haiku weight {name} with shape {array.shape}")
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        print("Name:", name)

        name = name.split("/")
        # we skip trainable position encodign + decoder parameters for now
        if any(n in ["trainable_position_encoding", "decoder"] for n in name):
            print(f"Skipping {'/'.join(name)}")
            continue
        # if first scope name starts with "perceiver_encoder", change it to "encoder"
        if name[0] == "perceiver_encoder":
            name[0] = "encoder"
        pointer = model
        for m_name in name:
            print("m_name:", m_name)
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b" or scope_names[0] == "offset":
                pointer = getattr(pointer, "bias")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


@torch.no_grad()
def convert_perceiver_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """

    # download pickle from URL if it doesn't exist already
    if os.path.exists("language_perceiver_io_bytes.pickle"):
        pickle_file = "language_perceiver_io_bytes.pickle"
    else:
        pickle_file = wget.download(checkpoint_url)

    # load parameters as FlatMapping data structure
    with open(pickle_file, "rb") as f:
        params = pickle.loads(f.read())

    # create initial state dict
    state_dict = dict()
    for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
        for param_name, param in parameters.items():
            state_dict[scope_name + "/" + param_name] = param

    # load HuggingFace model
    config = PerceiverConfig()
    model = PerceiverModel(config)
    model.eval()
    load_haiku_weights_in_perceiver(model, state_dict)

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
        type=str,
        default="https://storage.googleapis.com/perceiver_io/language_perceiver_io_bytes.pickle",
        help="URL of a Perceiver checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_perceiver_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
