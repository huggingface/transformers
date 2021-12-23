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
"""Tokenization classes for RetriBERT."""

from ...utils import logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_retribert import RetriBertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "yjernite/retribert-base-uncased": "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "yjernite/retribert-base-uncased": "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "yjernite/retribert-base-uncased": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "yjernite/retribert-base-uncased": {"do_lower_case": True},
}


class RetriBertTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" RetriBERT tokenizer (backed by HuggingFace's *tokenizers* library).

    [`RetriBertTokenizerFast`] is identical to [`BertTokenizerFast`] and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = RetriBertTokenizer
    model_input_names = ["input_ids", "attention_mask"]
