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
        "--pretrained_model_config",
        type=str,
        default="roberta-base",
        help="The model config to use. Note that we don't copy the model's weights, only the config!",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="unigram-tokenizer-wikitext",
        help="The name of the tokenizer to load. We use the pretrained tokenizer to initialize the model's vocab size.",
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
    parser.add_argument(
        "--tpu_name",
        type=str,
        help="Name of TPU resource to initialize. Should be blank on Colab, and 'local' on TPU VMs."
    )

    parser.add_argument(
        "--tpu_zone",
        type=str,
        help="Google cloud zone that TPU resource is located in. Only used for non-Colab TPU nodes."
    )

    parser.add_argument(
        "--gcp_project",
        type=str,
        help="Google cloud project name. Only used for non-Colab TPU nodes."
    )

    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use mixed-precision bfloat16 for training. This is the recommended lower-precision format for TPU."
    )

    args = parser.parse_args()
    return args


def initialize_tpu(args):
    try:
        if args.tpu_name:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                args.tpu_name, zone=args.tpu_zone, project=args.gcp_project
            )
        else:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        raise RuntimeError(f"Couldn't connect to TPU!")

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    return tpu


def main(args):
    tpu = initialize_tpu(args)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    if args.bfloat16:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = AutoConfig.from_pretrained(args.pretrained_config)
    config.vocab_size = tokenizer.vocab_size

    with strategy.scope():
        model = TFAutoModelForMaskedLM.from_config(config)
        model(model.dummy_inputs)  # Pass some dummy inputs through the model to ensure all the weights are built








if __name__ == "__main__":
    args = parse_args()
    main(args)