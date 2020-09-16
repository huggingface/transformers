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

from transformers.tokenization_phobert import VOCAB_FILES_NAMES, PhobertTokenizer

from .test_tokenization_common import TokenizerTesterMixin


class PhobertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PhobertTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["Tôi", "là", "sinh_viên", "trường", "đại_học", "Công_nghệ", "."]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "n h</w>", "n g</w>", "p h", "t r", "i _", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return PhobertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "Tôi là sinh_viên trường đại_học Công_nghệ ."
        output_text = "Tôi là sinh_viên trường đại_học Công_nghệ ."
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = PhobertTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "Tôi là sinh_viên trường đại_học Công_nghệ ."
        bpe_tokens = "<s> Tôi là sinh_viên trường đại_học Công_nghệ . </s>".split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [0, 218, 8, 649, 212, 956, 2413, 5, 2, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
