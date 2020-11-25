# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


import json
import os
import unittest

from transformers import LongformerTokenizer, LongformerTokenizerFast, LONGFORMER_PRETRAINED_TOKENIZER_ARCHIVE_LIST
from transformers.testing_utils import require_tokenizers

from .test_tokenization_roberta import RobertaTokenizationTest


@require_tokenizers
class LongformerTokenizationTest(RobertaTokenizationTest):

    pretrained_vocab_checkpoints = LONGFORMER_PRETRAINED_TOKENIZER_ARCHIVE_LIST
    tokenizer_class = LongformerTokenizer
    rust_tokenizer_class = LongformerTokenizerFast
    test_rust_tokenizer = True
