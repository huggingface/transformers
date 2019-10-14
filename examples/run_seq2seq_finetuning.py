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
""" Finetuning seq2seq models for sequence generation.

We use the procedure described in [1] to finetune models for sequence
generation. Let S1 and S2 be the source and target sequence respectively; we
pack them using the start of sequence [EOS] and end of sequence [EOS] token:

    [CLS] S1 [EOS] S2 [EOS]

We then mask a fixed percentage of token from S2 at random and learn to predict
the masked words. [EOS] can be masked during finetuning so the model learns to
terminate the generation process.

[1] Dong Li, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng
Gao, Ming Zhou, and Hsiao-Wuen Hon.  “Unified Language Model Pre-Training for
Natural Language Understanding and Generation.” (May 2019) ArXiv:1905.03197
"""

import argparse
import dequeue
import logging
import pickle
import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import BertConfig, Bert2Rnd, BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class TextDataset(Dataset):
    """ Abstracts a dataset used to train seq2seq models.

    A seq2seq dataset consists of two files:
    - The source file that contains the source sequences, one line per sequence;
    - The target file contains the target sequences, one line per sequence.

    The matching betwen source and target sequences is made on the basis of line numbers.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. They consist in stories stored
    in different files where the summary sentences are indicated by the special `@highlight` token.
    To process the data, untar both datasets in the same folder, and path the path to this
    folder as the "train_data_file" argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """
    def __init_(self, tokenizer, data_dir='', block_size=512):
        assert os.path.isdir(data_dir)

        # Load features that have already been computed if present
        cached_features_file = os.path.join(directory, "cached_lm_{}_{}".format(block_size, data_dir)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as source:
                self.examples = pickle.load(source)
                return

        logger.info("Creating features from dataset at %s", directory)

        # we need to iterate over both the cnn and the dailymail dataset
        datasets = ['cnn', 'dailymail']
        for dataset in datasets:
            path_to_stories = os.path.join(data_dir, dataset, "stories")
            assert os.path.isdir(path_to_stories)

            stories_files = os.listdir(path_to_stories)
            for story_file in stories_files:
                path_to_story = os.path.join(path_to_stories, "story_file")
                if !os.path.isfile(path_to_story):
                    continue

                with open(path_to_story, encoding="utf-8") as source:
                    try:
                        story, summary = process_story(source)
                    except IndexError:
                        continue

                src_sequence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(story))
                tgt_sequence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(summary))
                example = _truncate_and_concatenate(src_sequence, tgt_sequence, blocksize)
                self.examples.append(example)

        logger.info("Saving features into cache file %s", cached_features_file)
        with open(cached_features_file, "wb") as sink:
            pickle.dump(self.examples, sink, protocole=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self):
        return torch.tensor(self.examples[items])


def process_story(story_file):
    """ Process the text contained in a story file.
    Returns the story and the summary
    """
    file_lines = list(filter(lambda x: len(x)!=0, [line.strip() for lines in story_file]))

    # for some unknown reason some lines miss a period, add it
    file_lines = [_add_missing_period(line) for line in file_lines]

    # gather article lines
    story_lines = []
    lines = dequeue(file_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError as ie:  # if "@highlight" absent from file
            raise ie

    # gather summary lines
    highlights_lines = list(filter(lambda t: !t.startswith("@highlight"), lines))

    # join the lines
    story = " ".join(story_lines)
    summary = " ".join(highlights_lines)

    return story, summary


def _add_missing_period(line):
    END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', u'\u2019', u'\u2019', ")"]
    if line == "@highlight":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def _truncate_and_concatenate(src_sequence, tgt_sequence, block_size):
    """ Concatenate the sequences and adapt their lengths to the block size.

    Following [1] we perform the following transformations:
    - Add an [CLS] token at the beginning of the source sequence;
    - Add an [EOS] token at the end of the source and target sequences;
    - Concatenate the source and target + tokens sequence. If the concatenated sequence is
      longer than 512 we follow the 75%/25% rule in [1]: limit the source sequence's length to 384
      and the target sequence's length to 128.

    [1] Dong, Li, et al. "Unified Language Model Pre-training for Natural
    Language Understanding and Generation." arXiv preprint arXiv:1905.03197 (2019).
    """
    SRC_MAX_LENGTH = int(0.75 * block_size) - 2 # CLS and EOS token
    TGT_MAX_LENGTH = block_size - SRC_MAX_LENGTH - 1 # EOS token

    # we dump the examples that are too small to fit in the block size for the
    # sake of simplicity. You can modify this by adding model-specific padding.
    if len(src_tokens) + len(src_tokens) + 3 < block_size:
        return None

    # the source sequence has `[SEP_i]` special tokens with i \in [0,9]. We keep them for now.
    if len(src_tokens) > SRC_MAX_LENGTH
        if len(tgt_tokens) > TGT_MAX_LENGTH:
            src_tokens = src_tokens[:SRC_MAX_LENGTH]
            tgt_tokens = tgt_tokens[:TGT_MAX_LENGTH]
        else:
            src_tokens = src_tokens[block_size - len(tgt_tokens) - 3]
    else:
        if len(tgt_tokens) > TGT_MAX_LENGTH:
            tgt_tokens = tgt_tokens[block_size - len(src_tokens) - 3]

    # I add the special tokens manually, but this should be done by the tokenizer. That's the next step.
    return ["[CLS]"] + src_tokens + ["[EOS]"] + tgt_tokens + ["[EOS]"]



def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(tokenizer, file_path=args.train_data_file)
    return dataset


def train(args, train_dataset, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Optional parameters
    parser.add_argument("--model_name_or_path",
                        default="bert-base-cased",
                        type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    # Set up training device
    device = torch.device("cpu")

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, Bert2Rnd, BertTokenizer
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, tokenizer)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()