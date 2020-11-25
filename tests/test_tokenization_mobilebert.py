# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Allegro.pl and The HuggingFace Inc. team.
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

from transformers import MobileBertTokenizer, MobileBertTokenizerFast, MOBILEBERT_PRETRAINED_TOKENIZER_ARCHIVE_LIST
from transformers.testing_utils import get_tests_dir, require_tokenizers, slow

from .test_tokenization_bert import BertTokenizationTest
from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class MobilebertTokenizationTest(BertTokenizationTest):

    pretrained_vocab_checkpoints = MOBILEBERT_PRETRAINED_TOKENIZER_ARCHIVE_LIST
    tokenizer_class = MobileBertTokenizer
    rust_tokenizer_class = MobileBertTokenizerFast
    test_rust_tokenizer = True
