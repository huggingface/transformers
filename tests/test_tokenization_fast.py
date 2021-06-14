# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import unittest

from transformers import PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class PreTrainedTokenizationFastTest(TokenizerTesterMixin, unittest.TestCase):
    rust_tokenizer_class = PreTrainedTokenizerFast
    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_vocab_key = "tokenizer_file"

    def setUp(self):
        self.test_rust_tokenizer = False  # because we don't have pretrained_vocab_files_map
        super().setUp()
        self.test_rust_tokenizer = True

        self.tokenizers_list = [(PreTrainedTokenizerFast, "robot-test/dummy-tokenizer-fast", {})]

        tokenizer = PreTrainedTokenizerFast.from_pretrained("robot-test/dummy-tokenizer-fast")
        tokenizer.save_pretrained(self.tmpdirname)

    def test_pretrained_model_lists(self):
        # We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any
        # model
        pass

    def test_prepare_for_model(self):
        # We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any
        # model
        pass

    def test_rust_tokenizer_signature(self):
        # PreTrainedTokenizerFast doesn't have tokenizer_file in its signature
        pass
