# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import importlib
import unittest

from transformers import RoFormerTokenizer, RoFormerTokenizerFast
from transformers.testing_utils import require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


def is_rjieba_available():
    return importlib.util.find_spec("rjieba") is not None


def require_rjieba(test_case):
    """
    Decorator marking a test that requires Jieba. These tests are skipped when Jieba isn't installed.
    """
    if not is_rjieba_available():
        return unittest.skip("test requires rjieba")(test_case)
    else:
        return test_case


@require_rjieba
@require_tokenizers
class RoFormerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = RoFormerTokenizer
    rust_tokenizer_class = RoFormerTokenizerFast
    space_between_special_tokens = True
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

    def get_tokenizer(self, **kwargs):
        return self.tokenizer_class.from_pretrained("junnyu/roformer_chinese_base", **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return self.rust_tokenizer_class.from_pretrained("junnyu/roformer_chinese_base", **kwargs)

    def get_chinese_input_output_texts(self):
        input_text = "永和服装饰品有限公司,今天天气非常好"
        output_text = "永和 服装 饰品 有限公司 , 今 天 天 气 非常 好"
        return input_text, output_text

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()
        input_text, output_text = self.get_chinese_input_output_texts()
        tokens = tokenizer.tokenize(input_text)

        self.assertListEqual(tokens, output_text.split())

        input_tokens = tokens + [tokenizer.unk_token]
        exp_tokens = [22943, 21332, 34431, 45904, 117, 306, 1231, 1231, 2653, 33994, 1266, 100]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), exp_tokens)

    def test_rust_tokenizer(self):
        tokenizer = self.get_rust_tokenizer()
        input_text, output_text = self.get_chinese_input_output_texts()
        tokens = tokenizer.tokenize(input_text)
        self.assertListEqual(tokens, output_text.split())
        input_tokens = tokens + [tokenizer.unk_token]
        exp_tokens = [22943, 21332, 34431, 45904, 117, 306, 1231, 1231, 2653, 33994, 1266, 100]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), exp_tokens)

    # due to custom pre_tokenize , char_to_token may be error
    def test_alignement_methods(self):
        pass
