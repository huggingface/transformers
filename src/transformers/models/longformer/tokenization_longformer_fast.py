# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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

from ...utils import logging
from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
from .tokenization_longformer import LongformerTokenizer


logger = logging.get_logger(__name__)


LONGFORMER_PRETRAINED_TOKENIZER_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # See all LONGFORMER models at https://huggingface.co/models?filter=longformer
]


class LongformerTokenizerFast(RobertaTokenizerFast):
    r"""
    Construct a "fast" Longformer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.LongformerTokenizerFast` is identical to :class:`~transformers.RobertaTokenizerFast`. Refer
    to the superclass for usage examples and documentation concerning parameters.
    """
    # merges and vocab same as Roberta
    max_model_input_sizes = 4096
    slow_tokenizer_class = LongformerTokenizer
