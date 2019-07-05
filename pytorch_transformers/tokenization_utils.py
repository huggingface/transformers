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
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
from io import open

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

from .file_utils import cached_path

logger = logging.getLogger(__name__)


class PreTrainedTokenizer(object):
    """ An abstract class to handle dowloading and loading pretrained tokenizers.
    """
    vocab_files_names = {}
    pretrained_vocab_files_map = {}
    max_model_input_sizes = {}

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedTokenizer from pre-trained vocabulary files.
        Download and cache the vocabulary files if needed.
        """
        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        if pretrained_model_name_or_path in s3_models:
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
        else:
            for file_id, file_name in cls.vocab_files_names.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                else:
                    full_file_name = pretrained_model_name_or_path
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We don't load it.".format(full_file_name))
                    full_file_name = None
                vocab_files[file_id] = full_file_name
        # redirect to the cache, if necessary
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(file_path, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                logger.error("Couldn't reach server to download vocabulary.")
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find files {} "
                    "at this path or url.".format(
                        pretrained_model_name_or_path, ', '.join(s3_models),
                        pretrained_model_name_or_path, str(vocab_files.keys())))
            return None

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info("loading file {} from cache at {}".format(
                    file_path, resolved_vocab_files[file_id]))

        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)

        # Merge resolved_vocab_files arguments in kwargs.
        for args_name, file_path in resolved_vocab_files.items():
            kwargs[args_name] = file_path

        # Instantiate tokenizer.
        tokenizer = cls(*inputs, **kwargs)

        return tokenizer

    def tokenize(self, text):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, token_ids, *input, **kwargs):
        raise NotImplementedError

    def save_vocabulary(self, vocab_path):
        raise NotImplementedError


def clean_up_tokenization(out_string):
    out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
                    ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
                    ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string
