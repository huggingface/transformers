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
"""Tokenization classes for DPR."""


import collections
import logging
from typing import List, Optional, Union

from .file_utils import add_end_docstrings, add_start_docstrings
from .tokenization_bart import BartTokenizer, BartTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# TODO(piktus) Rename keys to facebook/default-rag or similar
# TODO(piktus) Figure out handling of doc separator and title separator
# see issues with adding tokens similar to https://github.com/huggingface/transformers/issues/3090

RAG_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    },
    "merges_file": {
        "facebook/bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
    },
}


class RagDefaultTokenizer(BartTokenizer):
    r"""
    Constructs a  DPRContextEncoderTokenizer.

    :class:`~transformers.DPRContextEncoderTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = RAG_PRETRAINED_VOCAB_FILES_MAP


class RagDefaultTokenizerFast(BartTokenizerFast):
    r"""
    Constructs a  DPRContextEncoderTokenizer.

    :class:`~transformers.DPRContextEncoderTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = RAG_PRETRAINED_VOCAB_FILES_MAP
