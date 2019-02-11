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

from pytorch_pretrained_bert.tokenization_openai import OpenAIGPTTokenizer


class OpenAIGPTTokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        """ Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt """
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "w</w>", "r</w>", "t</w>",
                 "lo", "low", "er</w>",
                 "low</w>", "lowest</w>", "newer</w>", "wider</w>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w", "e r</w>", ""]
        with open("/tmp/openai_tokenizer_vocab_test.json", "w") as fp:
            json.dump(vocab_tokens, fp)
            vocab_file = fp.name
        with open("/tmp/openai_tokenizer_merges_test.txt", "w") as fp:
            fp.write("\n".join(merges))
            merges_file = fp.name

        tokenizer = OpenAIGPTTokenizer(vocab_file, merges_file, special_tokens=["<unk>"])
        os.remove(vocab_file)
        os.remove(merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

if __name__ == '__main__':
    unittest.main()
