# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors.
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
""" Tokenization class for model LayoutLM."""


from ...utils import logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_layoutlm import LayoutLMTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

LAYOUTLM_PRETRAINED_TOKENIZER_ARCHIVE_LIST = [
    "microsoft/layoutlm-base-uncased",
    "microsoft/layoutlm-large-uncased",
    # See all LAYOUTLM models at https://huggingface.co/models?filter=layoutlm
]

class LayoutLMTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "Fast" LayoutLMTokenizer.

    :class:`~transformers.LayoutLMTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = 512
    slow_tokenizer_class = LayoutLMTokenizer
