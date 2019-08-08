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

import json
import logging
import re
from io import open
import six
import os

from .tokenization_utils import PreTrainedTokenizer
from .tokenization_gpt2 import GPT2Tokenizer

logger = logging.getLogger(__name__)

DICT_FILES_NAMES = {
    'dict_file': 'dict.txt',
}

PRETRAINED_DICT_FILES_MAP = {
    'dict_file':
        {
            'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-dict.txt",
            'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-dict.txt",
            'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-dict.txt",
        },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'roberta-base': 512,
    'roberta-large': 512,
    'roberta-large-mnli': 512,
}

SPACE_NORMALIZER = re.compile(r"\s+")

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Dictionary(object):
    """
    A mapping from symbols to consecutive integers

    From Facebook's fairseq.
    """

    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f, ignore_utf_errors)
        return d

    def add_from_file(self, f, ignore_utf_errors=False):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, six.string_types):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        self.add_from_file(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))
            return

        lines = f.read().splitlines()
        for line in lines:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            self.indices[word] = len(self.symbols)
            self.symbols.append(word)
            self.count.append(count)

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=True, reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = [0] * (nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids


class RobertaTokenizer(PreTrainedTokenizer):
    """
    RoBERTa tokenizer. Peculiarities:
        - GPT-2 tokenizer with a different integer mapping on top.
    """
    vocab_files_names = DICT_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_DICT_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, dict_file, bpe_tokenizer=None, bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
                 unk_token="<unk>", **kwargs):
        super(RobertaTokenizer, self).__init__(cls_token=bos_token, sep_token=eos_token, eos_token=eos_token,
                                               unk_token=unk_token, **kwargs)

        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if bpe_tokenizer is None else bpe_tokenizer
        self.dictionary = Dictionary.load(dict_file)

    @property
    def vocab_size(self):
        return len(self.dictionary.indices)

    def _tokenize(self, text):
        """ Use GPT-2 Tokenizer """
        return self.gpt2_tokenizer._tokenize(text)

    def _convert_token_to_id(self, token):
        if self.dictionary.index(token) != 3:
            return self.dictionary.index(token)
        return self.dictionary.index(str(self.gpt2_tokenizer.convert_tokens_to_ids(token)))

    def _convert_id_to_token(self, index):
        symbol = self.dictionary[index]
        try:
            idx = int(symbol)
            return self.gpt2_tokenizer._convert_id_to_token(idx)
        except ValueError:
            return symbol

    def convert_tokens_to_string(self, tokens):
        return self.gpt2_tokenizer.convert_tokens_to_string(tokens)

    def convert_tokens_to_ids(self, tokens, no_sep_cls_tokens=False):
        cls = [self._convert_token_to_id(self.cls_token)]
        tokens = super().convert_tokens_to_ids(tokens)
        sep = [self._convert_token_to_id(self.sep_token)]
        return (cls + tokens + sep) if (isinstance(tokens, list) and not no_sep_cls_tokens) else tokens

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return super().convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)[1:-1]

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        dict_file = os.path.join(save_directory, DICT_FILES_NAMES['dict_file'])

        with open(dict_file, 'w', encoding='utf-8') as f:
            for i in range(self.dictionary.nspecial, len(self.dictionary.count)):
                f.write(f"{list(self.dictionary.indices.keys())[i]} {self.dictionary.count[i]}\n")

        vocab_files = self.gpt2_tokenizer.save_pretrained(save_directory)

        return vocab_files + (dict_file,)
