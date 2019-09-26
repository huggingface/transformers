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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from io import open

from transformers.tokenization_distilbert import (DistilBertTokenizer)

from .tokenization_tests_commons import CommonTestCases
from .tokenization_bert_test import BertTokenizationTest

class DistilBertTokenizationTest(BertTokenizationTest):

    tokenizer_class = DistilBertTokenizer

    def get_tokenizer(self, **kwargs):
        return DistilBertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_sequence_builders(self):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.add_special_tokens_single_sequence(text)
        encoded_pair = tokenizer.add_special_tokens_sequence_pair(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + \
               text_2 + [tokenizer.sep_token_id]


if __name__ == '__main__':
    unittest.main()
