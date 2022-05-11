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
"""Convert OPT checkpoint."""


import argparse
from pathlib import Path

import torch

from transformers import OPTConfig, OPTModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model" in sd.keys():
        sd = torch.load(checkpoint_path, map_location="cpu")["model"]

    # pop unnecessary weights
    if "decoder.version" in sd:
        sd.pop("decoder.version")
    return sd


@torch.no_grad()
def convert_opt_checkpoint(checkpoint_path, pytorch_dump_folder_path, config=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    state_dict = load_checkpoint(checkpoint_path)

    config = OPTConfig()

    model = OPTModel(config).half().eval()
    model.load_state_dict(state_dict)

    # Check results
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path", type=str, help="path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here: https://huggingface.co/models?other=opt_metasq"
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Define HF config."
    )
    args = parser.parse_args()
    convert_opt_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, config=args.hf_config)
