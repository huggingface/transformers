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

"""Script for training a Unigram tokenizer."""

import argparse
import logging

import datasets
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

from transformers import AlbertTokenizerFast


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a unigram tokenizer on the wikitext dataset.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size during training.",
    )
    parser.add_argument(
        "-vs",
        "--vocab_size",
        type=int,
        default=10000,
        help="Size of the desired vocabulary.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Limit the number of shards (used for debugging).",
    )
    parser.add_argument(
        "--export_to_hub",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def get_unigram_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = normalizers.Sequence([normalizers.Replace("``", '"'), normalizers.Replace("''", '"')])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    return tokenizer


def get_unigram_trainer(vocab_size: int) -> UnigramTrainer:
    trainer = UnigramTrainer(
        unk_token="<unk>",
        special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"],
        vocab_size=vocab_size,
    )
    return trainer


def main(args):
    wikitext = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    if args.limit is not None:
        max_train_samples = min(len(wikitext), args.limit)
        wikitext = wikitext.select(range(max_train_samples))
        logger.info(f"Limiting the dataset to {args.limit} entries.")

    def batch_iterator():
        for i in range(0, len(wikitext), args.batch_size):
            yield wikitext[i : i + args.batch_size]["text"]

    logger.info("Training the tokenizer.")
    tokenizer = get_unigram_tokenizer()
    trainer = get_unigram_trainer(args.vocab_size)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    logger.info("Tokenizer training complete!")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()

    if args.export_to_hub:
        logger.info("Exporting the trained tokenzier to Hub.")
        new_tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
        new_tokenizer.push_to_hub("unigram-tokenizer-wikitext")


if __name__ == "__main__":
    args = parse_args()
    main(args)
