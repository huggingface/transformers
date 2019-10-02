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

from transformers.tokenization_xlm import XLMTokenizer, VOCAB_FILES_NAMES

from .tokenization_tests_commons import CommonTestCases

class XLMTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = XLMTokenizer

    def setUp(self):
        super(XLMTokenizationTest, self).setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "w</w>", "r</w>", "t</w>",
                 "lo", "low", "er</w>",
                 "low</w>", "lowest</w>", "newer</w>", "wider</w>", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        return XLMTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"lower newer"
        output_text = u"lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        """ Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt """
        tokenizer = XLMTokenizer(self.vocab_file, self.merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_sequence_builders(self):
        tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.add_special_tokens_single_sequence(text)
        encoded_pair = tokenizer.add_special_tokens_sequence_pair(text, text_2)

        assert encoded_sentence == [1] + text + [1]
        assert encoded_pair == [1] + text + [1] + text_2 + [1]

if __name__ == '__main__':
    unittest.main()
