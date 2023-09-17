# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers import NougatTokenizerFast
from transformers.models.nougat.tokenization_nougat_fast import markdown_compatible
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class NougatTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    slow_tokenizer_class = None
    rust_tokenizer_class = NougatTokenizerFast
    tokenizer_class = NougatTokenizerFast
    test_rust_tokenizer = True
    test_slow_tokenizer = False
    from_pretrained_vocab_key = "tokenizer_file"
    special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}

    def setUp(self):
        super().setUp()
        tokenizer = NougatTokenizerFast.from_pretrained("nielsr/nougat")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return NougatTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def test_padding(self, max_length=6):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                # Simple input
                sentence1 = "This is a simple input"
                sentence2 = ["This is a simple input 1", "This is a simple input 2"]
                pair1 = ("This is a simple input", "This is a pair")
                pair2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                try:
                    tokenizer_r.encode(sentence1, max_length=max_length)
                    tokenizer_r.encode_plus(sentence1, max_length=max_length)

                    tokenizer_r.batch_encode_plus(sentence2, max_length=max_length)
                    tokenizer_r.encode(pair1, max_length=max_length)
                    tokenizer_r.batch_encode_plus(pair2, max_length=max_length)
                except ValueError:
                    self.fail("Nougat Tokenizer should be able to deal with padding")

                tokenizer_r.pad_token = None  # Hotfixing padding = None
                self.assertRaises(
                    ValueError, tokenizer_r.encode, sentence1, max_length=max_length, padding="max_length"
                )

                # Simple input
                self.assertRaises(
                    ValueError, tokenizer_r.encode_plus, sentence1, max_length=max_length, padding="max_length"
                )

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    sentence2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, pair1, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError, tokenizer_r.encode_plus, pair1, max_length=max_length, padding="max_length"
                )

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    pair2,
                    max_length=max_length,
                    padding="max_length",
                )

    @unittest.skip("NougatTokenizerFast does not have tokenizer_file in its signature")
    def test_rust_tokenizer_signature(self):
        pass

    @unittest.skip("NougatTokenizerFast does not support pretokenized inputs")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip("NougatTokenizerFast directly inherits from PreTrainedTokenizerFast")
    def test_prepare_for_model(self):
        pass


class MarkdownCompatibleTest(unittest.TestCase):
    def test_bold_formatting(self):
        input_text = r"This is \bm{bold} text."
        expected_output = r"This is \mathbf{bold} text."
        self.assertEqual(markdown_compatible(input_text), expected_output)

    def test_url_conversion(self):
        input_text = "Visit my website at https://www.example.com"
        expected_output = "Visit my website at [https://www.example.com](https://www.example.com)"
        self.assertEqual(markdown_compatible(input_text), expected_output)

    def test_algorithm_code_block(self):
        input_text = "```python\nprint('Hello, world!')\n```"
        expected_output = "```\npython\nprint('Hello, world!')\n```"
        self.assertEqual(markdown_compatible(input_text), expected_output)