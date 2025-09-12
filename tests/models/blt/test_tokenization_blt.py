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

from transformers import BltTokenizer
from transformers.testing_utils import require_tokenizers
from transformers.tokenization_utils import AddedToken

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class BltTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = []
    tokenizer_class = BltTokenizer
    rust_tokenizer_class = None

    test_rust_tokenizer = False
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

    def test_simple_encode_decode(self):
        tokenizer = BltTokenizer(add_bos_token=False, add_eos_token=False)

        text = "Hello"
        encoded = tokenizer.encode(text, add_special_tokens=False)

        # "Hello" in UTF-8 bytes: [72, 101, 108, 108, 111]
        # With offset +4: [76, 105, 112, 112, 115]
        expected = [76, 105, 112, 112, 115]
        self.assertEqual(encoded, expected)

        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_special_tokens_encoding(self):
        tokenizer = BltTokenizer(add_bos_token=True, add_eos_token=True)

        text = "Hi"
        encoded = tokenizer.encode(text, add_special_tokens=True)

        # "Hi" in UTF-8 bytes: [72, 105] -> with offset: [76, 109]
        # With BOS (1) and EOS (2): [1, 76, 109, 2]
        expected = [1, 76, 109, 2]
        self.assertEqual(encoded, expected)

    def test_tokenize_method(self):
        tokenizer = BltTokenizer()

        text = "ABC"
        tokens = tokenizer._tokenize(text)

        # "ABC" in UTF-8 bytes: [65, 66, 67]
        expected = ["65", "66", "67"]
        self.assertEqual(tokens, expected)

    def test_token_conversion(self):
        """Test token to ID and ID to token conversion"""
        tokenizer = BltTokenizer()

        # Test byte token conversion
        token = "65"  # Byte value for 'A'
        token_id = tokenizer._convert_token_to_id(token)
        self.assertEqual(token_id, 69)  # 65 + 4 offset

        converted_token = tokenizer._convert_id_to_token(token_id)
        self.assertEqual(converted_token, token)

        bos_id = tokenizer._convert_token_to_id(str(tokenizer.bos_token))
        self.assertEqual(bos_id, 1)

        bos_token = tokenizer._convert_id_to_token(1)
        self.assertEqual(bos_token, str(tokenizer.bos_token))

    def test_convert_tokens_to_string(self):
        tokenizer = BltTokenizer()

        tokens = ["72", "101", "108", "108", "111"]  # "Hello" in bytes
        result = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(result, "Hello")

        # Test with special tokens mixed in (should be ignored)
        tokens_with_special = [str(tokenizer.bos_token), "72", "105", str(tokenizer.eos_token)]
        result = tokenizer.convert_tokens_to_string(tokens_with_special)
        self.assertEqual(result, "Hi")

    def test_unicode_handling(self):
        tokenizer = BltTokenizer(add_bos_token=False, add_eos_token=False)

        # Test Unicode character (é)
        text = "café"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

        # Test emoji
        text = "Hello 👋"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_empty_and_whitespace(self):
        tokenizer = BltTokenizer(add_bos_token=False, add_eos_token=False)

        # Test empty string
        encoded = tokenizer.encode("", add_special_tokens=False)
        self.assertEqual(encoded, [])
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "")

        # Test single space
        encoded = tokenizer.encode(" ", add_special_tokens=False)
        self.assertEqual(encoded, [36])  # 32 (space) + 4 offset
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, " ")

    def test_build_inputs_with_special_tokens(self):
        tokenizer = BltTokenizer(add_bos_token=True, add_eos_token=True)

        # Single sequence
        token_ids = [76, 109]  # "Hi" encoded (H=72+4=76, i=105+4=109)
        result = tokenizer.build_inputs_with_special_tokens(token_ids)
        expected = [1, 76, 109, 2]  # BOS + tokens + EOS
        self.assertEqual(result, expected)

        # Pair of sequences
        token_ids_1 = [76, 109]  # "Hi"
        token_ids_2 = [66, 121, 101]  # "Bye"
        result = tokenizer.build_inputs_with_special_tokens(token_ids_1, token_ids_2)
        expected = [1, 76, 109, 2, 66, 121, 101, 2]  # BOS + seq1 + EOS + seq2 + EOS
        self.assertEqual(result, expected)

    def test_special_tokens_mask(self):
        tokenizer = BltTokenizer(add_bos_token=True, add_eos_token=True)

        token_ids = [76, 109]  # "Hi" encoded (H=72+4=76, i=105+4=109)
        mask = tokenizer.get_special_tokens_mask(token_ids)
        expected = [1, 0, 0, 1]  # BOS=1, content=0, content=0, EOS=1
        self.assertEqual(mask, expected)

    def test_add_special_tokens_flags(self):
        tokenizer1 = BltTokenizer(add_bos_token=True, add_eos_token=True)
        encoded1 = tokenizer1.encode("Hi", add_special_tokens=True)
        self.assertEqual(encoded1[0], 1)  # BOS
        self.assertEqual(encoded1[-1], 2)  # EOS

        tokenizer2 = BltTokenizer(add_bos_token=False, add_eos_token=False)
        encoded2 = tokenizer2.encode("Hi", add_special_tokens=True)
        self.assertNotEqual(encoded2[0], 1)  # No BOS
        self.assertNotEqual(encoded2[-1], 2)  # No EOS

        # Test with only BOS
        tokenizer3 = BltTokenizer(add_bos_token=True, add_eos_token=False)
        encoded3 = tokenizer3.encode("Hi", add_special_tokens=True)
        self.assertEqual(encoded3[0], 1)  # BOS
        self.assertNotEqual(encoded3[-1], 2)  # No EOS

    def test_added_tokens(self):
        tokenizer = BltTokenizer()

        custom_token = AddedToken("<custom>", normalized=False, special=True)
        tokenizer.add_tokens([custom_token])

        self.assertIn("<custom>", tokenizer.get_vocab())

        token_id = tokenizer._convert_token_to_id("<custom>")
        self.assertIsInstance(token_id, int)

        back_token = tokenizer._convert_id_to_token(token_id)
        self.assertEqual(back_token, "<custom>")

    @unittest.skip("Blt is byte-level, special tokens are encoded as bytes")
    def test_add_special_tokens(self):
        pass

    @unittest.skip("Blt byte-level tokenization doesn't handle pretokenized inputs the same way")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip("Blt encodes added tokens as bytes, not single tokens")
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip("Blt tokenizer serialization needs additional work for added tokens")
    def test_save_and_load_tokenizer(self):
        pass


if __name__ == "__main__":
    unittest.main()
