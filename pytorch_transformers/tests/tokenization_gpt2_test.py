# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
import json
from io import open

from pytorch_transformers.tokenization_gpt2 import GPT2Tokenizer, VOCAB_FILES_NAMES

from .tokenization_tests_commons import CommonTestCases

class GPT2TokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = GPT2Tokenizer

    def setUp(self):
        super(GPT2TokenizationTest, self).setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "\u0120", "\u0120l", "\u0120n",
                 "\u0120lo", "\u0120low", "er",
                 "\u0120lowest", "\u0120newer", "\u0120wider", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return GPT2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"lower newer"
        output_text = u" lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = GPT2Tokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)


if __name__ == '__main__':
    unittest.main()
