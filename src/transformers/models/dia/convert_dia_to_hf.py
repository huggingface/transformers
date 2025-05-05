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

from transformers import (
    DiaFeatureExtractor,
    DiaProcessor,
    DiaTokenizerFast,
)
from transformers.utils.import_utils import _is_package_available


# TODO if I do some checkpoint renaming


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

    model, is_multilingual, num_languages = convert_openai_dia_to_tfms(
        args.checkpoint_path, args.pytorch_dump_folder_path
    )

    if args.convert_preprocessor:
        try:
            if not _is_package_available("tiktoken"):
                raise ModuleNotFoundError(
                    """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
                )
        except Exception as e:
            print(e)
        else:
            tokenizer = convert_tiktoken_to_hf(is_multilingual, num_languages)
            feature_extractor = DiaFeatureExtractor(
                feature_size=model.config.num_mel_bins,
                # the rest of default parameters are the same as hardcoded in openai/dia
            )
            processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
            processor.save_pretrained(args.pytorch_dump_folder_path)

            # save fast tokenizer as well
            fast_tokenizer = DiaTokenizerFast.from_pretrained(args.pytorch_dump_folder_path)
            fast_tokenizer.save_pretrained(args.pytorch_dump_folder_path, legacy_format=False)

    model.save_pretrained(args.pytorch_dump_folder_path)
