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


def bytes_to_hex(bline: bytes, sep: str = " ") -> str:
    return str(binascii.hexlify(bline, sep), "utf-8")


def str_to_hex(line: str, sep: str = " ") -> str:
    return bytes_to_hex(bytes(line, "utf-8"), sep)


class TestByteRewriter(unittest.TestCase):
    tokenizer_path = "Tomlim/myt5-base"
    tokenizer = MyT5Tokenizer.from_pretrained(tokenizer_path)
    decompose_rewriter = tokenizer.decompose_rewriter

    def test_simple_decompose(self):
        # test rewriting
        in_str = "Hello WorlD"
        out_str = "hAello wAorldA"

        in_hex = str_to_hex(in_str).split(" ")
        out_hex = str_to_hex(out_str).split(" ")

        self.assertEqual(self.decompose_rewriter.rewrite_bytes(in_hex), out_hex)

    def test_simple_decompose_reversible(self):
        in_str = "Hello WorlD"
        out_str = "Hello WorlD"

        in_hex = str_to_hex(in_str).split(" ")
        out_hex = str_to_hex(out_str).split(" ")

        self.assertEqual(
            self.decompose_rewriter.rewrite_bytes(self.decompose_rewriter.rewrite_bytes(in_hex), reverse=True), out_hex
        )

    def test_simple_decompose_non_latin(self):
        in_str = "你好世界 Hello WorlD"
        out_str = "你好世界 hAello wAorldA"

        in_hex = str_to_hex(in_str).split(" ")
        out_hex = str_to_hex(out_str).split(" ")

        self.assertEqual(self.decompose_rewriter.rewrite_bytes(in_hex), out_hex)

    def test_unrecognized_byte(self):
        in_hex = ["00", "01", "xx", "03", "61"]
        out_hex = ["00", "01", "xx", "03", "61"]

        self.assertEqual(self.decompose_rewriter.rewrite_bytes(in_hex), out_hex)


class MyT5TokenizationTest(unittest.TestCase):
    tokenizer_path = "Tomlim/myt5-base"
    tokenizer = MyT5Tokenizer.from_pretrained(tokenizer_path)

    def test_simple_tokenize(self):
        in_str = "Hello World"
        out_tokens = ["52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90"]

        self.assertEqual(self.tokenizer.tokenize(in_str), out_tokens)

        in_pl_str = "Witaj świecie"
        out_tokens = ["77", "41", "69", "74", "61", "6a", "20", "4b", "a5", "97", "63", "69", "65"]

        self.assertEqual(self.tokenizer.tokenize(in_pl_str), out_tokens)

        in_jp_str = "こんにちは世界"
        out_tokens = ["58", "80", "91", "a1", "e4", "b8", "96", "e7", "95", "8c"]

        self.assertEqual(self.tokenizer.tokenize(in_jp_str), out_tokens)

    def test_batch_tokenize(self):
        in_batch = ["Hello World", "Witaj świecie", "こんにちは世界"]

        out_tokens = [
            ["52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90", "</s>"],
            ["77", "41", "69", "74", "61", "6a", "20", "4b", "a5", "97", "63", "69", "65", "</s>"],
            ["58", "80", "91", "a1", "e4", "b8", "96", "e7", "95", "8c", "</s>"],
        ]

        self.assertListEqual(
            [self.tokenizer.convert_ids_to_tokens(ids) for ids in self.tokenizer(in_batch)["input_ids"]], out_tokens
        )

    def test_special_bytes(self):
        in_str_special = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
        out_tokens = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]

        self.assertEqual(self.tokenizer.tokenize(in_str_special), out_tokens)

        in_str_mixed = "\x00Hello\x01 World\x02"
        out_tokens = ["00", "52", "85", "91", "9f", "6f", "01", "20", "52", "85", "9f", "90", "02"]

        self.assertEqual(self.tokenizer.tokenize(in_str_mixed), out_tokens)

    def test_special_tokens(self):
        in_str_special = "<unk></s><pad>"
        out_tokens = ["<unk>", "</s>", "<pad>"]

        self.assertEqual(self.tokenizer.tokenize(in_str_special), out_tokens)

        in_str_not_special = "<s>"
        out_tokens = ["3c", "73", "3e"]

        self.assertEqual(self.tokenizer.tokenize(in_str_not_special), out_tokens)

        in_str_mixed = "<s>Hello World</s>"
        out_tokens = ["3c", "73", "3e", "52", "85", "91", "9f", "6f", "20", "52", "85", "9f", "90", "</s>"]

        self.assertEqual(self.tokenizer.tokenize(in_str_mixed), out_tokens)

    def test_token_ids_conversion(self):
        tokens_range = [f"{x:02x}" for x in range(256)]
        indices_range = list(range(3, 256 + 3))

        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(tokens_range), indices_range)
        self.assertListEqual(self.tokenizer.convert_ids_to_tokens(indices_range), tokens_range)

        special_tokens = ["<pad>", "</s>", "<unk>"]
        special_indices = [0, 1, 2]

        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(special_tokens), special_indices)
        self.assertListEqual(self.tokenizer.convert_ids_to_tokens(special_indices), special_tokens)
