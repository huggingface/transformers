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

from pytorch_transformers.tokenization_gpt2 import GPT2Tokenizer, VOCAB_FILES_NAMES

from .tokenization_tests_commons import create_and_check_tokenizer_commons, TemporaryDirectory

class GPT2TokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        """ Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt """
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "lo", "low", "er",
                 "low", "lowest", "newer", "wider", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w", "e r", ""]
        special_tokens_map = {"unk_token": "<unk>"}

        with TemporaryDirectory() as tmpdirname:
            vocab_file = os.path.join(tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
            merges_file = os.path.join(tmpdirname, VOCAB_FILES_NAMES['merges_file'])
            with open(vocab_file, "w") as fp:
                fp.write(json.dumps(vocab_tokens))
            with open(merges_file, "w") as fp:
                fp.write("\n".join(merges))

            input_text = u"lower newer"
            output_text = u"lower<unk>newer"

            create_and_check_tokenizer_commons(self, input_text, output_text, GPT2Tokenizer, tmpdirname, **special_tokens_map)

            tokenizer = GPT2Tokenizer(vocab_file, merges_file, **special_tokens_map)
            text = "lower"
            bpe_tokens = ["low", "er"]
            tokens = tokenizer.tokenize(text)
            self.assertListEqual(tokens, bpe_tokens)

            input_tokens = tokens + [tokenizer.unk_token]
            input_bpe_tokens = [13, 12, 17]
            self.assertListEqual(
                tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)


if __name__ == '__main__':
    unittest.main()
