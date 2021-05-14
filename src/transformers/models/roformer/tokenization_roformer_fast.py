# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for RoFormer."""
from ...utils import logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_roformer import RoFormerTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"junnyu/roformer_chinese_small": 1536, "junnyu/roformer_chinese_base": 1536}


PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
}


class RoFormerTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" RoFormer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.RoFormerTokenizerFast` is almost identical to :class:`~transformers.BertTokenizerFast` and
    runs end-to-end tokenization: punctuation splitting and wordpiece. There are some difference between them when
    tokenizing Chinese.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.

    Example::

        >>> from transformers import RoFormerTokenizerFast
        >>> tokenizer = RoFormerTokenizerFast.from_pretrained('junnyu/roformer_chinese_base')
        >>> tokenizer.tokenize("今天天气非常好。")
        # ['今', '天', '天', '气', '非常', '好', '。']

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = RoFormerTokenizer
