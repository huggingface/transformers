# coding=utf-8
# Copyright 2018 The Microsoft Reseach team and The HuggingFace Inc. team.
# Copyright (c) 2018 Microsoft and The HuggingFace Inc.  All rights reserved.
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
""" Finetuning seq2seq models for sequence generation."""

import argparse
from collections import deque
import logging
import pickle
import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, Model2Model

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


class TextDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init_(self, tokenizer_src, tokenizer_tgt, data_dir="", block_size=512):
        assert os.path.isdir(data_dir)

        # Load features that have already been computed if present
        cached_features_file = os.path.join(
            data_dir, "cached_lm_{}_{}".format(block_size, data_dir)
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as source:
                self.examples = pickle.load(source)
                return

        logger.info("Creating features from dataset at %s", data_dir)

        datasets = ["cnn", "dailymail"]
        for dataset in datasets:
            path_to_stories = os.path.join(data_dir, dataset, "stories")
            assert os.path.isdir(path_to_stories)

            story_filenames_list = os.listdir(path_to_stories)
            for story_filename in story_filenames_list:
                path_to_story = os.path.join(path_to_stories, story_filename)
                if not os.path.isfile(path_to_story):
                    continue

                with open(path_to_story, encoding="utf-8") as source:
                    try:
                        raw_story = source.read()
                        story, summary = process_story(raw_story)
                    except IndexError:  # skip ill-formed stories
                        continue

                story = tokenizer_src.convert_tokens_to_ids(tokenizer_src.tokenize(story))
                story_seq = _fit_to_block_size(story, block_size)

                summary = tokenizer_tgt.convert_tokens_to_ids(tokenizer_tgt.tokenize(summary))
                summary_seq = _fit_to_block_size(summary, block_size)

                self.examples.append((story_seq, summary_seq))

        logger.info("Saving features into cache file %s", cached_features_file)
        with open(cached_features_file, "wb") as sink:
            pickle.dump(self.examples, sink, protocole=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, items):
        return torch.tensor(self.examples[items])


def process_story(raw_story):
    """ Extract the story and summary from a story file.

    Attributes:
        raw_story (str): content of the story file as an utf-8 encoded string.

    Raises:
        IndexError: If the stoy is empty or contains no highlights.
    """
    file_lines = list(
        filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split("\n")])
    )

    # for some unknown reason some lines miss a period, add it
    file_lines = [_add_missing_period(line) for line in file_lines]

    # gather article lines
    story_lines = []
    lines = deque(file_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError as ie:  # if "@highlight" absent from file
            raise ie

    # gather summary lines
    highlights_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

    # join the lines
    story = " ".join(story_lines)
    summary = " ".join(highlights_lines)

    return story, summary


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', u"\u2019", u"\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."


def _fit_to_block_size(sequence, block_size):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter than the block size we pad it with -1 ids
    which correspond to padding tokens.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        return sequence.extend([-1] * [block_size - len(sequence)])


def load_and_cache_examples(args, tokenizer_src, tokenizer_tgt):
    dataset = TextDataset(tokenizer_src, tokenizer_tgt, file_path=args.data_dir)
    return dataset


# ------------
# Train
# ------------


def train(args, train_dataset, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Optional parameters
    parser.add_argument(
        "--decoder_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint to initialize the decoder's weights with.",
    )
    parser.add_argument(
        "--decoder_type",
        default="bert",
        type=str,
        help="The decoder architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--encoder_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder's weights with.",
    )
    parser.add_argument(
        "--encoder_type",
        default="bert",
        type=str,
        help="The encoder architecture to be fine-tuned.",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.encoder_type != 'bert' or args.decoder_type != 'bert':
        raise ValueError("Only the BERT architecture is currently supported for seq2seq.")

    # Set up training device
    # device = torch.device("cpu")

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    encoder_tokenizer_class = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
    decoder_tokenizer_class = AutoTokenizer.from_pretrained(args.decoder_name_or_path)
    model = Model2Model.from_pretrained(args.encoder_name_or_path, args.decoder_name_or_path)
    # model.to(device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    source, target = load_and_cache_examples(args, tokenizer)
    # global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
