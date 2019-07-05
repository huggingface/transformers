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
import shutil
import pytest

from pytorch_pretrained_bert.tokenization_xlm import XLMTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP

from.tokenization_tests_commons import create_and_check_tokenizer_commons

class XLMTokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        """ Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt """
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "w</w>", "r</w>", "t</w>",
                 "lo", "low", "er</w>",
                 "low</w>", "lowest</w>", "newer</w>", "wider</w>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]
        with open("/tmp/openai_tokenizer_vocab_test.json", "w") as fp:
            fp.write(json.dumps(vocab_tokens))
            vocab_file = fp.name
        with open("/tmp/openai_tokenizer_merges_test.txt", "w") as fp:
            fp.write("\n".join(merges))
            merges_file = fp.name

        create_and_check_tokenizer_commons(self, XLMTokenizer, vocab_file, merges_file, special_tokens=["<unk>", "<pad>"])

        tokenizer = XLMTokenizer(vocab_file, merges_file, special_tokens=["<unk>", "<pad>"])
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

    @pytest.mark.slow
    def test_tokenizer_from_pretrained(self):
        cache_dir = "/tmp/pytorch_pretrained_bert_test/"
        for model_name in list(PRETRAINED_VOCAB_ARCHIVE_MAP.keys())[:1]:
            tokenizer = XLMTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(tokenizer)


if __name__ == '__main__':
    unittest.main()
