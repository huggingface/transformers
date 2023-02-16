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
import os

import datasets
import tensorflow as tf

from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare TFRecord shards from pre-tokenized samples of the wikitext dataset."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="sayakpaul/unigram-tokenizer-wikitext",
        help="Tokenizer identifier. Can be a local filepath or a Hub identifier.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of entries to go in a single shard.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "validation"])
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Limit the number of shards (used for debugging).",
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


def get_serialized_examples(tokenizer):
    def fn(examples, max_length=128):
        tokenized_data = tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        records = []
        for i in range(len(examples)):
            features = {
                "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=tokenized_data["input_ids"][i])),
                "attention_mask": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=tokenized_data["attention_mask"][i])
                ),
            }
            features = tf.train.Features(feature=features)
            example = tf.train.Example(features=features)
            record_bytes = example.SerializeToString()
            records.append(record_bytes)
        return records

    return fn


def main(args):
    wikitext = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split)

    if args.limit is not None:
        max_samples = min(len(wikitext), args.limit)
        wikitext = wikitext.select(range(max_samples))
        logger.info(f"Limiting the dataset to {args.limit} entries.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Handle output directory creation.
    # For serializing into a Google Cloud Storage Bucket, one needs to first
    # create a bucket.
    if "gs" not in args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        split_dir = os.path.join(args.output_dir, args.split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    else:
        split_dir = os.path.join(args.output_dir, args.split)

    shard_count = 0
    get_serialized_examples_fn = get_serialized_examples(tokenizer)
    for shard in range(0, len(wikitext), args.shard_size):
        dataset_snapshot = wikitext[shard : shard + args.shard_size]["text"]
        shard_size = len(dataset_snapshot)
        filename = os.path.join(split_dir, f"wikitext-{args.limit}-{shard_count}-{shard_size}.tfrecord")
        serialized_examples = get_serialized_examples_fn(dataset_snapshot)

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                example = serialized_examples[i]
                out_file.write(example)
            logger.info("Wrote file {} containing {} records".format(filename, shard_size))

        shard_count += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
