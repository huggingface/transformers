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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
import json
from io import open

from transformers.tokenization_ctrl import CTRLTokenizer, VOCAB_FILES_NAMES

from .tokenization_tests_commons import CommonTestCases

class CTRLTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = CTRLTokenizer

    def setUp(self):
        super(CTRLTokenizationTest, self).setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ['adapt', 're@@', 'a@@', 'apt', 'c@@', 't', '<unk>']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", 'a p', 'ap t</w>', 'r e', 'a d', 'ad apt</w>', '']
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return CTRLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"adapt react readapt apt"
        output_text = u"adapt react readapt apt"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = CTRLTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "adapt react readapt apt"
        bpe_tokens = 'adapt re@@ a@@ c@@ t re@@ adapt apt'.split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [0, 1, 2, 4, 5, 1, 0, 3, 6]
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)


if __name__ == '__main__':
    unittest.main()
