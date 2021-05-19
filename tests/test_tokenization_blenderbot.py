#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the;
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
# LICENSE file in the root directory of this source tree.
"""Tests for Blenderbot Tokenizers, including common tests for BlenderbotSmallTokenizer."""
import json
import os
import unittest

from transformers.file_utils import cached_property
from transformers.tokenization_blenderbot import VOCAB_FILES_NAMES, BlenderbotSmallTokenizer, BlenderbotTokenizer

from .test_tokenization_common import TokenizerTesterMixin


class BlenderbotSmallTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BlenderbotSmallTokenizer

    def setUp(self):
        super().setUp()

        vocab = ["__start__", "adapt", "act", "ap@@", "te", "__end__", "__unk__"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        merges = ["#version: 0.2", "a p", "t e</w>", "ap t</w>", "a d", "ad apt</w>", "a c", "ac t</w>", ""]
        self.special_tokens_map = {"unk_token": "__unk__", "bos_token": "__start__", "eos_token": "__end__"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BlenderbotSmallTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "adapt act apte"
        output_text = "adapt act apte"
        return input_text, output_text

    def test_full_blenderbot_small_tokenizer(self):
        tokenizer = BlenderbotSmallTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "adapt act apte"
        bpe_tokens = ["adapt", "act", "ap@@", "te"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]

        input_bpe_tokens = [0, 1, 2, 3, 4, 5]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_special_tokens_small_tok(self):
        tok = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot-90M")
        assert tok("sam").input_ids == [1384]
        src_text = "I am a small frog."
        encoded = tok([src_text], padding=False, truncation=False)["input_ids"]
        decoded = tok.batch_decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        assert src_text != decoded  # I wish it did!
        assert decoded == "i am a small frog ."


class Blenderbot3BTokenizerTests(unittest.TestCase):
    @cached_property
    def tokenizer_3b(self):
        return BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")

    def test_encode_decode_cycle(self):
        tok = self.tokenizer_3b
        src_text = " I am a small frog."
        encoded = tok([src_text], padding=False, truncation=False)["input_ids"]
        decoded = tok.batch_decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        assert src_text == decoded

    def test_3B_tokenization_same_as_parlai(self):
        assert self.tokenizer_3b.add_prefix_space
        assert self.tokenizer_3b([" Sam", "Sam"]).input_ids == [[5502, 2], [5502, 2]]
