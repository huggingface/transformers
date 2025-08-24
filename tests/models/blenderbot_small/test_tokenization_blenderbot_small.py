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
"""Tests for the Blenderbot small tokenizer."""

import json
import os
import unittest
from functools import lru_cache

from transformers.models.blenderbot_small.tokenization_blenderbot_small import (
    VOCAB_FILES_NAMES,
    BlenderbotSmallTokenizer,
)

from ...test_tokenization_common import TokenizerTesterMixin, use_cache_if_possible


class BlenderbotSmallTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/blenderbot_small-90M"
    tokenizer_class = BlenderbotSmallTokenizer
    test_rust_tokenizer = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        vocab = ["__start__", "adapt", "act", "ap@@", "te", "__end__", "__unk__"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        merges = ["#version: 0.2", "a p", "t e</w>", "ap t</w>", "a d", "ad apt</w>", "a c", "ac t</w>", ""]
        cls.special_tokens_map = {"unk_token": "__unk__", "bos_token": "__start__", "eos_token": "__end__"}

        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        cls.merges_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(cls.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    @classmethod
    @use_cache_if_possible
    @lru_cache(maxsize=64)
    def get_tokenizer(cls, pretrained_name=None, **kwargs):
        kwargs.update(cls.special_tokens_map)
        pretrained_name = pretrained_name or cls.tmpdirname
        return BlenderbotSmallTokenizer.from_pretrained(pretrained_name, **kwargs)

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

    def test_empty_word_small_tok(self):
        tok = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot-90M")
        src_text = "I am a small frog ."
        src_text_dot = "."
        encoded = tok(src_text)["input_ids"]
        encoded_dot = tok(src_text_dot)["input_ids"]

        assert encoded[-1] == encoded_dot[0]
