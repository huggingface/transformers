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
import json
import unittest

from pytorch_transformers.tokenization_roberta import RobertaTokenizer, VOCAB_FILES_NAMES
from .tokenization_tests_commons import CommonTestCases


class RobertaTokenizationTest(CommonTestCases.CommonTokenizerTester):
    tokenizer_class = RobertaTokenizer

    def setUp(self):
        super(RobertaTokenizationTest, self).setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "lo", "low", "er",
                 "low", "lowest", "newer", "wider", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.tmpdirname, **self.special_tokens_map)

    def get_input_output_texts(self):
        input_text = u"lower newer"
        output_text = u"lower<unk>newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = RobertaTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower"
        bpe_tokens = ["low", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [13, 12, 17]
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def roberta_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(
            tokenizer.encode('Hello world!'),
            [0, 31414, 232, 328, 2]
        )
        self.assertListEqual(
            tokenizer.encode('Hello world! cécé herlolip 418'),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]
        )

    def test_sequence_builders(self):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_text_from_decode = tokenizer.encode("sequence builders", add_special_tokens=True)
        encoded_pair_from_decode = tokenizer.encode("sequence builders", "multi-sequence build", add_special_tokens=True)

        encoded_sentence = tokenizer.add_special_tokens_single_sentence(text)
        encoded_pair = tokenizer.add_special_tokens_sentences_pair(text, text_2)

        assert encoded_sentence == encoded_text_from_decode
        assert encoded_pair == encoded_pair_from_decode


if __name__ == '__main__':
    unittest.main()
