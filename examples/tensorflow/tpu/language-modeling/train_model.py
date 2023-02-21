#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Script for preparing TFRecord shards for pre-tokenized examples."""

import argparse
import logging

import tensorflow as tf

from transformers import AutoTokenizer, AutoConfig, TFAutoModelForMaskedLM


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare TFRecord shards from pre-tokenized samples of the wikitext dataset."
    )
    parser.add_argument(
        "--pretrained_config",
        type=str,
        default="roberta-base",
        help="The model config to use. Note that we don't copy the model's weights, only the config!",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length. For training on TPUs, it helps to have a maximum"
        " sequence length that is a multiple of 8.",
    )
    parser.add_argument(
        "--output_dir",
        default="tf-tpu",
        type=str,
        help="Output directory where the TFRecord shards will be saved. If the"
        " path is appended with `gs://` ('gs://tf-tpu', for example) then the TFRecord"
        " shards will be directly saved to a Google Cloud Storage bucket.",
    )

    args = parser.parse_args()
    return args


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("unigram-tokenizer-wikitext")
    config = AutoConfig.from_pretrained(args.pretrained_config)
    config.vocab_size = tokenizer.vocab_size
    model = TFAutoModelForMaskedLM.from_config(config)



if __name__ == "__main__":
    args = parse_args()
    main(args)