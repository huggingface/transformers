#!/usr/bin/env python
"""Converts a Dia model in OpenAI format to Hugging Face format."""
# Copyright 2025 The HuggingFace Inc. team and the OpenAI team. All rights reserved.
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

import torch

from transformers import AutoModel, DiaAudioProcessor, DiaProcessor, DiaTokenizer
from transformers.utils.import_utils import _is_package_available


def convert_dia_model_to_hf(checkpoint_path, pytorch_dump_folder_path):
    """
    Converts a Dia model in OpenAI format to Hugging Face format.
    Args:
        checkpoint_path (`str`):
            Path to the downloaded checkpoints.
        pytorch_dump_folder_path (`str`):
            Path to the output PyTorch model.
    """
    tokenizer = DiaTokenizer.from_pretrained(checkpoint_path)
    model = AutoModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16)

    # Save the model
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--checkpoint_path", type=str, help="Path to the downloaded checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=False,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    args = parser.parse_args()

    model = convert_dia_model_to_hf(args.checkpoint_path, args.pytorch_dump_folder_path)

    if args.convert_preprocessor:
        try:
            if not _is_package_available("tiktoken"):
                raise ModuleNotFoundError(
                    """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
                )
        except Exception as e:
            print(e)
        else:
            feature_extractor = DiaAudioProcessor(
                feature_size=model.config.num_mel_bins,
                # the rest of default parameters are the same as hardcoded in openai/dia
            )
            tokenizer = DiaTokenizer()
            processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
            processor.save_pretrained(args.pytorch_dump_folder_path)

    model.save_pretrained(args.pytorch_dump_folder_path)
