# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
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


import json
import os
import unittest
from functools import lru_cache

from transformers.models.ctrl.tokenization_ctrl import VOCAB_FILES_NAMES, CTRLTokenizer

from ...test_tokenization_common import TokenizerTesterMixin, use_cache_if_possible


class CTRLTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "Salesforce/ctrl"
    tokenizer_class = CTRLTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["adapt", "re@@", "a@@", "apt", "c@@", "t", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "a p", "ap t</w>", "r e", "a d", "ad apt</w>", ""]
        cls.special_tokens_map = {"unk_token": "<unk>"}

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
        return CTRLTokenizer.from_pretrained(pretrained_name, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "adapt react readapt apt"
        output_text = "adapt react readapt apt"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = CTRLTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "adapt react readapt apt"
        bpe_tokens = "adapt re@@ a@@ c@@ t re@@ adapt apt".split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [0, 1, 2, 4, 5, 1, 0, 3, 6]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
