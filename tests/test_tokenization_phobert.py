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


import os
import unittest

from transformers.models.phobert.tokenization_phobert import VOCAB_FILES_NAMES, PhobertTokenizer

from .test_tokenization_common import TokenizerTesterMixin


class PhobertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PhobertTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["T@@", "i", "I", "R@@", "r", "e@@"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l à</w>"]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            for token in vocab_tokens:
                fp.write("{} {}".format(token, vocab_tokens[token]) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return PhobertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "Tôi là VinAI Research"
        output_text = "T<unk> i <unk> <unk> <unk> <unk> <unk> <unk> I Re<unk> e<unk> <unk> <unk> <unk>"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = PhobertTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "Tôi là VinAI Research"
        bpe_tokens = "T@@ ô@@ i l@@ à V@@ i@@ n@@ A@@ I R@@ e@@ s@@ e@@ a@@ r@@ c@@ h".split()
        tokens = tokenizer.tokenize(text)
        print(tokens)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [4, 3, 5, 3, 3, 3, 3, 3, 3, 6, 7, 9, 3, 9, 3, 3, 3, 3, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
