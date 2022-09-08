#!/usr/bin/env python3
# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Tests for Blenderbot Tokenizers, including common tests for BlenderbotSmallTokenizer."""
import unittest

from transformers import BlenderbotTokenizer, BlenderbotTokenizerFast
from transformers.utils import cached_property


class Blenderbot3BTokenizerTests(unittest.TestCase):
    @cached_property
    def tokenizer_3b(self):
        return BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")

    @cached_property
    def rust_tokenizer_3b(self):
        return BlenderbotTokenizerFast.from_pretrained("facebook/blenderbot-3B")

    def test_encode_decode_cycle(self):
        tok = self.tokenizer_3b
        src_text = " I am a small frog."
        encoded = tok([src_text], padding=False, truncation=False)["input_ids"]
        decoded = tok.batch_decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        assert src_text == decoded

    def test_encode_decode_cycle_rust_tokenizer(self):
        tok = self.rust_tokenizer_3b
        src_text = " I am a small frog."
        encoded = tok([src_text], padding=False, truncation=False)["input_ids"]
        decoded = tok.batch_decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        assert src_text == decoded

    def test_3B_tokenization_same_as_parlai(self):
        assert self.tokenizer_3b.add_prefix_space
        assert self.tokenizer_3b([" Sam", "Sam"]).input_ids == [[5502, 2], [5502, 2]]

    def test_3B_tokenization_same_as_parlai_rust_tokenizer(self):
        assert self.rust_tokenizer_3b.add_prefix_space
        assert self.rust_tokenizer_3b([" Sam", "Sam"]).input_ids == [[5502, 2], [5502, 2]]
