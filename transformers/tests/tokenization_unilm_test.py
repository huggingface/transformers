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

from transformers.tests.tokenization_bert_test import BertTokenizationTest
from transformers.tokenization_unilm import UnilmTokenizer

from .utils import slow

class UnilmTokenizationTest(BertTokenizationTest):
    tokenizer_class = UnilmTokenizer

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("unilm-base-cased")

        assert type(tokenizer) is UnilmTokenizer

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]


if __name__ == '__main__':
    unittest.main()
