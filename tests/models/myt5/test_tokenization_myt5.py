# coding=utf-8
# Copyright 2024
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
import binascii
import unittest

from transformers import MyT5Tokenizer
from transformers.utils import is_tf_available, is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


def bytes_to_hex(bline: bytes, sep: str = " ") -> str:
    return str(binascii.hexlify(bline, sep), "utf-8")


def str_to_hex(line: str, sep: str = " ") -> str:
    return bytes_to_hex(bytes(line, "utf-8"), sep)


class TestByteRewriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = MyT5Tokenizer.from_pretrained("Tomlim/myt5-base")

    def test_simple_decompose(self):
        decompose_rewriter = self.tokenizer.decompose_rewriter

        # test rewriting
        in_str = "Hello WorlD"
        out_str = "hAello wAorldA"

        in_hex = str_to_hex(in_str).split(" ")
        out_hex = str_to_hex(out_str).split(" ")

        self.assertEqual(decompose_rewriter.rewrite_bytes(in_hex), out_hex)

    def test_simple_decompose_reversible(self):
        decompose_rewriter = self.tokenizer.decompose_rewriter

        in_str = "Hello WorlD"
        out_str = "Hello WorlD"

        in_hex = str_to_hex(in_str).split(" ")
        out_hex = str_to_hex(out_str).split(" ")

        self.assertEqual(
            decompose_rewriter.rewrite_bytes(decompose_rewriter.rewrite_bytes(in_hex), reverse=True), out_hex
        )

    def test_simple_decompose_non_latin(self):
        decompose_rewriter = self.tokenizer.decompose_rewriter

        in_str = "你好世界 Hello WorlD"
        out_str = "你好世界 hAello wAorldA"

        in_hex = str_to_hex(in_str).split(" ")
        out_hex = str_to_hex(out_str).split(" ")

        self.assertEqual(decompose_rewriter.rewrite_bytes(in_hex), out_hex)

    def test_unrecognized_byte(self):
        decompose_rewriter = self.tokenizer.decompose_rewriter

        in_hex = ["00", "01", "xx", "03", "61"]
        out_hex = ["00", "01", "xx", "03", "61"]

        self.assertEqual(decompose_rewriter.rewrite_bytes(in_hex), out_hex)


class MyT5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MyT5Tokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

    def get_tokenizer(self, **kwargs) -> MyT5Tokenizer:
        return self.tokenizer_class.from_pretrained("Tomlim/myt5-base", **kwargs)

    @unittest.skip(reason="inputs cannot be pretokenized as ids depend on whole input string")
    def test_pretokenized_inputs(self):
        pass

    def test_convert_tokens_to_string_format(self):
        tokenizer = self.get_tokenizer()
        with self.subTest(f"{tokenizer.__class__.__name__}"):
            tokens = ["52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90", "</s>"]
            string = tokenizer.convert_tokens_to_string(tokens)

            self.assertIsInstance(string, str)

    def test_simple_tokenize(self):
        tokenizer = self.get_tokenizer()

        in_str = "Hello World"
        out_tokens = ["52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90"]

        self.assertEqual(tokenizer.tokenize(in_str), out_tokens)

        in_pl_str = "Witaj świecie"
        out_tokens = ["77", "41", "69", "74", "61", "6a", "20", "4b", "a5", "97", "63", "69", "65"]

        self.assertEqual(tokenizer.tokenize(in_pl_str), out_tokens)

        in_jp_str = "こんにちは世界"
        out_tokens = ["58", "80", "91", "a1", "e4", "b8", "96", "e7", "95", "8c"]

        self.assertEqual(tokenizer.tokenize(in_jp_str), out_tokens)

    def test_batch_tokenize(self):
        tokenizer = self.get_tokenizer()

        in_batch = ["Hello World", "Witaj świecie", "こんにちは世界"]

        out_tokens = [
            ["52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90", "</s>"],
            ["77", "41", "69", "74", "61", "6a", "20", "4b", "a5", "97", "63", "69", "65", "</s>"],
            ["58", "80", "91", "a1", "e4", "b8", "96", "e7", "95", "8c", "</s>"],
        ]

        self.assertListEqual(
            [tokenizer.convert_ids_to_tokens(ids) for ids in tokenizer(in_batch)["input_ids"]], out_tokens
        )

    def test_special_bytes(self):
        tokenizer = self.get_tokenizer()

        in_str_special = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
        out_tokens = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]

        self.assertEqual(tokenizer.tokenize(in_str_special), out_tokens)

        in_str_mixed = "\x00Hello\x01 World\x02"
        out_tokens = ["00", "52", "85", "91", "9f", "6f", "01", "20", "52", "85", "9f", "90", "02"]

        self.assertEqual(tokenizer.tokenize(in_str_mixed), out_tokens)

    def test_special_tokens(self):
        tokenizer = self.get_tokenizer()

        in_str_special = "<unk></s><pad>"
        out_tokens = ["<unk>", "</s>", "<pad>"]

        self.assertEqual(tokenizer.tokenize(in_str_special), out_tokens)

        in_str_not_special = "<s>"
        out_tokens = ["3c", "73", "3e"]

        self.assertEqual(tokenizer.tokenize(in_str_not_special), out_tokens)

        in_str_mixed = "<s>Hello World</s>"
        out_tokens = ["3c", "73", "3e", "52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90", "</s>"]

        self.assertEqual(tokenizer.tokenize(in_str_mixed), out_tokens)

    def test_token_ids_conversion(self):
        tokenizer = self.get_tokenizer()

        tokens_range = [f"{x:02x}" for x in range(256)]
        indices_range = list(range(3, 256 + 3))

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens_range), indices_range)
        self.assertListEqual(tokenizer.convert_ids_to_tokens(indices_range), tokens_range)

        special_tokens = ["<pad>", "</s>", "<unk>"]
        special_indices = [0, 1, 2]

        self.assertListEqual(tokenizer.convert_tokens_to_ids(special_tokens), special_indices)
        self.assertListEqual(tokenizer.convert_ids_to_tokens(special_indices), special_tokens)
