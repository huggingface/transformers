# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team.
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

from transformers.testing_utils import custom_tokenizers, is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


if is_torch_available():
    from transformers.models.cpmant import CPMAntTokenizer, CPMAntTokenizerFast


@custom_tokenizers
class CPMAntTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    def test_pre_tokenization(self):
        tokenizer = CPMAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
        texts = "今天天气真好！"
        jieba_tokens = ["今天", "天气", "真", "好", "！"]
        tokens = tokenizer.tokenize(texts)
        self.assertListEqual(tokens, jieba_tokens)
        normalized_text = "今天天气真好！"
        input_tokens = [tokenizer.bos_token] + tokens

        input_jieba_tokens = [6, 9802, 14962, 2082, 831, 244]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_jieba_tokens)

        reconstructed_text = tokenizer.decode(input_jieba_tokens)
        self.assertEqual(reconstructed_text, normalized_text)

    def test_pre_fast_tokenization(self):
        tokenizer = CPMAntTokenizerFast.from_pretrained("openbmb/cpm-ant-10b")
        texts = "今天天气真好！"
        jieba_tokens = ["今天", "天气", "真", "好", "！"]
        tokens = tokenizer.tokenize(texts)
        self.assertListEqual(tokens, jieba_tokens)
        normalized_text = "今天天气真好！"
        input_tokens = [tokenizer.bos_token] + tokens

        input_jieba_tokens = [6, 9802, 14962, 2082, 831, 244]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_jieba_tokens)

        reconstructed_text = tokenizer.decode(input_jieba_tokens)
        self.assertEqual(reconstructed_text, normalized_text)
