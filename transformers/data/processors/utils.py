# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import sys
import copy
import json
import os
import logging
import math
import subprocess
import itertools
from functools import partial
from datetime import datetime
from multiprocessing import Pool

import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger()

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. 
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def tokenize_line(tokenizer, line):
    tokens = tokenizer.tokenize(line)
    tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
    return tokenized_text


def get_line_count(filename):
    try:
        out = subprocess.Popen(
            ["rg", "-cve" "^\s*$", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).communicate()[0]
        return int(out)
    except Exception:
        print("\n\n" +
              "COULDN'T FETCH LINE COUNT OF DATASET FILE, INSTALL RIPGREP TO DO SO EFFICIENTLY.\n" +
              "Install instructions: https://github.com/BurntSushi/ripgrep#installation" +
              "\n\n")
        return None


def line_buffer_generator(file_path, buffer_size):
    with open(file_path, encoding="utf-8") as f:
        buffer = []
        for line in f:
            if line == "\n":
                continue
            buffer.append(line)
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []
        yield buffer


class TextDataset(Dataset):
    def __init__(self, tokenizer, model_name_or_path, file_path='train', block_size=512, overwrite_cache=False):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            model_name_or_path
            + "_cached_lm_parallel_"
            + str(block_size)
            + "_"
            + filename,
        )

        line_buffer_size = 500000

        part = partial(tokenize_line, tokenizer)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                return
        else:
            logger.info("Creating features from dataset file at %s", directory)

            with Pool() as p:
                line_count = get_line_count(filename)
                total_line_chunks = math.ceil(line_count / line_buffer_size) if line_count else None
                tokenized_chunks = tqdm(
                    (
                        p.map(part, line_chunk, chunksize=(line_buffer_size // 10))
                        for line_chunk in line_buffer_generator(
                            file_path, line_buffer_size
                        )
                    ),
                    total=total_line_chunks,
                )
                tokenized_lines = list(
                    itertools.chain.from_iterable(tokenized_chunks)
                )  # flatten chunks
                tokenized_text = list(
                    itertools.chain.from_iterable(tokenized_lines)
                )  # flatten lines

                #  chunk tokens into block_size wide examples (truncate the last item if it doesn't fill the block_size)
                self.examples = [
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                    for i in range(0, len(tokenized_text) - block_size + 1, block_size)
                ]
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

