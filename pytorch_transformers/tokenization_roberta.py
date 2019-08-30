# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RoBERTa."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
from io import open

from .tokenization_gpt2 import GPT2Tokenizer

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
    },
    'merges_file':
    {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'roberta-base': 512,
    'roberta-large': 512,
    'roberta-large-mnli': 512,
}


class RobertaTokenizer(GPT2Tokenizer):
    """
    RoBERTa BPE tokenizer, derived from the GPT-2 tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => will add a space is there isn't.
          As a consequence, this tokenizer `encode` and `decode` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, errors='replace', bos_token="<s>", eos_token="</s>", sep_token="</s>",
                 cls_token="<s>", unk_token="<unk>", pad_token='<pad>', mask_token='<mask>', **kwargs):
        super(RobertaTokenizer, self).__init__(vocab_file=vocab_file, merges_file=merges_file, errors=errors,
                                               bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
                                               sep_token=sep_token, cls_token=cls_token, pad_token=pad_token,
                                               mask_token=mask_token, **kwargs)

    def add_special_tokens_single_sentence(self, token_ids):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        A RoBERTa sequence has the following format: <s> X </s>
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: <s> A </s></s> B </s>
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep
