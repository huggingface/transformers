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
from tqdm import tqdm, trange
import torch
from torch.utils.data import Dataset, RandomSampler

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

                story = tokenizer_src.convert_tokens_to_ids(
                    tokenizer_src.tokenize(story)
                )
                story_seq = _fit_to_block_size(story, block_size)

                summary = tokenizer_tgt.convert_tokens_to_ids(
                    tokenizer_tgt.tokenize(summary)
                )
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

    # Prepare the data loading
    args.train_bach_size = 1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_bach_size
    )

    # Prepare the optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=t_total
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            source = ([s for s, _ in batch]).to(args.device)
            target = ([t for _, t in batch]).to(args.device)
            model.train()
            outputs = model(source, target)
            loss = outputs[0]
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


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
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
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
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    args = parser.parse_args()

    if args.encoder_type != "bert" or args.decoder_type != "bert":
        raise ValueError(
            "Only the BERT architecture is currently supported for seq2seq."
        )

    # Set up training device
    # device = torch.device("cpu")

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    encoder_tokenizer_class = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
    decoder_tokenizer_class = AutoTokenizer.from_pretrained(args.decoder_name_or_path)
    model = Model2Model.from_pretrained(
        args.encoder_name_or_path, args.decoder_name_or_path
    )
    # model.to(device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, tokenizer)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
