# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenization classes for DistilBERT."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open

from .tokenization_bert import BertTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        'distilbert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'distilbert-base-uncased': 512,
    'distilbert-base-uncased-distilled-squad': 512,
    'distilbert-base-multilingual-cased': 512,
}


class DistilBertTokenizer(BertTokenizer):
    r"""
    Constructs a DistilBertTokenizer.
    :class:`~transformers.DistilBertTokenizer` is identical to BertTokenizer and runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
