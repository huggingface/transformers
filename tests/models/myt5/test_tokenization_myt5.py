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
import unittest

from transformers import MyT5Tokenizer


class MyT5TokenizationTest(unittest.TestCase):

    tokenizer_path = "Tomlim/myt5-base"
    tokenizer = MyT5Tokenizer.from_pretrained(tokenizer_path)

    def test_simple_tokenize(self):
        in_str = "Hello World"
        out_tokens = ['52', '85', '91', '9f', '6f', '20', '52', '85', '9f', '90']

        self.assertEqual(self.tokenizer.tokenize(in_str), out_tokens)

        in_pl_str = "Witaj świecie"
        out_tokens = ['77', '41', '69', '74', '61', '6a', '20', '4b', 'a5', '97', '63', '69', '65']

        self.assertEqual(self.tokenizer.tokenize(in_pl_str), out_tokens)

        in_jp_str = "こんにちは世界"
        out_tokens = ['58', '80', '91', 'a1', 'e4', 'b8', '96', 'e7', '95', '8c']

        self.assertEqual(self.tokenizer.tokenize(in_jp_str), out_tokens)

    def test_batch_tokenize(self):
        in_batch = ["Hello World", "Witaj świecie", "こんにちは世界"]

        out_tokens = [['52', '85', '91', '9f', '6f', '20', '52', '85', '9f', '90', '</s>'],
                      ['77', '41', '69', '74', '61', '6a', '20', '4b', 'a5', '97', '63', '69', '65', '</s>'],
                      ['58', '80', '91', 'a1', 'e4', 'b8', '96', 'e7', '95', '8c', '</s>']]

        self.assertListEqual(
            [self.tokenizer.convert_ids_to_tokens(ids) for ids in self.tokenizer(in_batch)["input_ids"]], out_tokens)