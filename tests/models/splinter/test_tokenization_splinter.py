# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
""" Testing suite for the splinter tokenizer. """


import unittest

from transformers import SplinterTokenizer, SplinterTokenizerFast
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
class SplinterTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SplinterTokenizer
    test_slow_tokenizer = True
    rust_tokenizer_class = SplinterTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    # TODO: Check in `TokenizerTesterMixin` if other attributes need to be changed
    def setUp(self):
        super().setUp()

        raise NotImplementedError("Here you have to implement the saving of a toy tokenizer in " "`self.tmpdirname`.")
