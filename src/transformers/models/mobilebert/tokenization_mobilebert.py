# coding=utf-8
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Tokenization classes for MobileBERT."""

import json
from typing import Optional

from tokenizers import normalizers

from ...utils import logging
from ...models.bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


# Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast with BERT->MobileBERT,Bert->MobileBert
class MobileBertTokenizer(BertTokenizer):
    pass

__all__ = ["MobileBertTokenizer"]
