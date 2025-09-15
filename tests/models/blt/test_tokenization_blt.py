# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers import BltTokenizer, BltTokenizerFast
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class BltTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = []
    tokenizer_class = BltTokenizer
    rust_tokenizer_class = BltTokenizerFast

    test_rust_tokenizer = True
    test_sentencepiece = False
    test_slow_tokenizer = True
    from_pretrained_kwargs = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create a simple Blt tokenizer for testing
        tokenizer = BltTokenizer()
        tokenizer.save_pretrained(cls.tmpdirname)

    def get_tokenizers(self, **kwargs):
        kwargs.update({"add_bos_token": True, "add_eos_token": False})
        return super().get_tokenizers(**kwargs)

    def test_unicode_handling(self):
        tokenizer = BltTokenizer(add_bos_token=False, add_eos_token=False)

        # Test Unicode character (é)
        text = "café"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        # "café" in UTF-8 bytes: [99, 97, 102, 195, 169] (é = 195, 169)
        expected = [byte_val + tokenizer.offset for byte_val in [99, 97, 102, 195, 169]]
        self.assertEqual(encoded, expected)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

        # Test emoji
        text = "Hello 👋"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        # "Hello 👋" in UTF-8 bytes: [72, 101, 108, 108, 111, 32, 240, 159, 145, 139] (👋 = 240, 159, 145, 139)
        expected = [byte_val + tokenizer.offset for byte_val in [72, 101, 108, 108, 111, 32, 240, 159, 145, 139]]
        self.assertEqual(encoded, expected)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_special_characters_and_unicode(self):
        tokenizer = BltTokenizer(add_bos_token=False, add_eos_token=False)

        # Test special characters with unicode
        text = "Hello, 世界! 🌍"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        expected = [
            byte_val + tokenizer.offset
            for byte_val in [72, 101, 108, 108, 111, 44, 32, 228, 184, 150, 231, 149, 140, 33, 32, 240, 159, 140, 141]
        ]
        self.assertEqual(encoded, expected)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

        # Test mixed special characters, numbers, and unicode
        text = "Price: $100.50 €75.25 🎉"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        expected = [
            byte_val + tokenizer.offset
            for byte_val in [
                80,
                114,
                105,
                99,
                101,
                58,
                32,
                36,
                49,
                48,
                48,
                46,
                53,
                48,
                32,
                226,
                130,
                172,
                55,
                53,
                46,
                50,
                53,
                32,
                240,
                159,
                142,
                137,
            ]
        ]
        self.assertEqual(encoded, expected)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

        # Test control characters with unicode
        text = "Line1\nLine2\tTabbed 中文"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        expected = [
            byte_val + tokenizer.offset
            for byte_val in [
                76,
                105,
                110,
                101,
                49,
                10,
                76,
                105,
                110,
                101,
                50,
                9,
                84,
                97,
                98,
                98,
                101,
                100,
                32,
                228,
                184,
                173,
                230,
                150,
                135,
            ]
        ]
        self.assertEqual(encoded, expected)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_empty_and_whitespace(self):
        tokenizer = BltTokenizer(add_bos_token=False, add_eos_token=False)

        encoded = tokenizer.encode("", add_special_tokens=False)
        self.assertEqual(encoded, [])
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "")

        encoded = tokenizer.encode(" ", add_special_tokens=False)
        self.assertEqual(encoded, [32 + tokenizer.offset])  # space + offset
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, " ")

    @unittest.skip("Blt byte-level tokenization doesn't handle pretokenized inputs the same way")
    def test_pretokenized_inputs(self):
        pass


if __name__ == "__main__":
    unittest.main()
