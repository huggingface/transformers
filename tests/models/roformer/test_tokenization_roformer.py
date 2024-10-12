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

import tempfile
import unittest

from transformers import RoFormerTokenizer, RoFormerTokenizerFast
from transformers.testing_utils import require_rjieba, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_rjieba
@require_tokenizers
class RoFormerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "junnyu/roformer_chinese_small"
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

    def test_rust_tokenizer(self):  # noqa: F811
        tokenizer = self.get_rust_tokenizer()
        input_text, output_text = self.get_chinese_input_output_texts()
        tokens = tokenizer.tokenize(input_text)
        self.assertListEqual(tokens, output_text.split())
        input_tokens = tokens + [tokenizer.unk_token]
        exp_tokens = [22943, 21332, 34431, 45904, 117, 306, 1231, 1231, 2653, 33994, 1266, 100]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), exp_tokens)

    @unittest.skip(reason="Cannot train new tokenizer via Tokenizers lib")
    def test_training_new_tokenizer(self):
        pass

    @unittest.skip(reason="Cannot train new tokenizer via Tokenizers lib")
    def test_training_new_tokenizer_with_special_tokens_change(self):
        pass

    def test_save_slow_from_fast_and_reload_fast(self):
        for cls in [RoFormerTokenizer, RoFormerTokenizerFast]:
            original = cls.from_pretrained("alchemab/antiberta2")
            self.assertEqual(original.encode("生活的真谛是"), [1, 4, 4, 4, 4, 4, 4, 2])

            with tempfile.TemporaryDirectory() as tmp_dir:
                original.save_pretrained(tmp_dir)
                new = cls.from_pretrained(tmp_dir)
            self.assertEqual(new.encode("生活的真谛是"), [1, 4, 4, 4, 4, 4, 4, 2])
