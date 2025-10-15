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


import os
import unittest

from transformers import BertTokenizer, AutoTokenizer
from transformers.models.bert.tokenization_bert import (
    VOCAB_FILES_NAMES,
    BertTokenizer,
)
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin, filter_non_english

input_text = "UNwant\u00e9d,running"

expected_tokens = ['unwanted', ',', 'running']
expected_token_ids = [101, 18162, 1010, 2770, 102]

@require_tokenizers
class BertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google-bert/bert-base-uncased"
    tokenizer_class = BertTokenizer
    rust_tokenizer_class = BertTokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "google-bert/bert-base-uncased"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]
    
    def test_integration_expected_tokens(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.tokenize(input_text), expected_tokens)

    def test_integration_expected_token_ids(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.encode(input_text), expected_token_ids)
