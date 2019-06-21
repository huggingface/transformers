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
from io import open
import shutil
import pytest

from pytorch_pretrained_bert.tokenization_xlnet import (XLNetTokenizer,
                                                        PRETRAINED_VOCAB_ARCHIVE_MAP,
                                                        SPIECE_UNDERLINE)

SAMPLE_VOCAB = os.path.join(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))),
                    'samples/test_sentencepiece.model')

class XLNetTokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB)

        tokens = tokenizer.tokenize('This is a test')
        self.assertListEqual(tokens, ['▁This', '▁is', '▁a', '▁t', 'est'])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

        vocab_path = "/tmp/"
        vocab_file, special_tokens_file = tokenizer.save_vocabulary(vocab_path)
        tokenizer = tokenizer.from_pretrained(vocab_path,
                                              keep_accents=True)
        os.remove(vocab_file)
        os.remove(special_tokens_file)

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + 'I', SPIECE_UNDERLINE + 'was', SPIECE_UNDERLINE + 'b', 'or', 'n', SPIECE_UNDERLINE + 'in', SPIECE_UNDERLINE + '',
                                      '9', '2', '0', '0', '0', ',', SPIECE_UNDERLINE + 'and', SPIECE_UNDERLINE + 'this',
                                      SPIECE_UNDERLINE + 'is', SPIECE_UNDERLINE + 'f', 'al', 's', 'é', '.'])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids, [8, 21, 84, 55, 24, 19, 7, 0,
                            602, 347, 347, 347, 3, 12, 66,
                            46, 72, 80, 6, 0, 4])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, [SPIECE_UNDERLINE + 'I', SPIECE_UNDERLINE + 'was', SPIECE_UNDERLINE + 'b', 'or', 'n', SPIECE_UNDERLINE + 'in',
                                           SPIECE_UNDERLINE + '', '<unk>', '2', '0', '0', '0', ',',
                                           SPIECE_UNDERLINE + 'and', SPIECE_UNDERLINE + 'this', SPIECE_UNDERLINE + 'is', SPIECE_UNDERLINE + 'f', 'al', 's',
                                           '<unk>', '.'])

    @pytest.mark.slow
    def test_tokenizer_from_pretrained(self):
        cache_dir = "/tmp/pytorch_pretrained_bert_test/"
        for model_name in list(PRETRAINED_VOCAB_ARCHIVE_MAP.keys())[:1]:
            tokenizer = XLNetTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(tokenizer)

    def test_tokenizer_lower(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, do_lower_case=True)
        tokens = tokenizer.tokenize(u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + '', 'i', SPIECE_UNDERLINE + 'was', SPIECE_UNDERLINE + 'b', 'or', 'n', SPIECE_UNDERLINE + 'in', SPIECE_UNDERLINE + '',
                                      '9', '2', '0', '0', '0', ',', SPIECE_UNDERLINE + 'and', SPIECE_UNDERLINE + 'this',
                                      SPIECE_UNDERLINE + 'is', SPIECE_UNDERLINE + 'f', 'al', 'se', '.'])
        self.assertListEqual(tokenizer.tokenize(u"H\u00E9llo"), ["▁he", "ll", "o"])

    def test_tokenizer_no_lower(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, do_lower_case=False)
        tokens = tokenizer.tokenize(u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + 'I', SPIECE_UNDERLINE + 'was', SPIECE_UNDERLINE + 'b', 'or', 'n', SPIECE_UNDERLINE + 'in', SPIECE_UNDERLINE + '',
                                      '9', '2', '0', '0', '0', ',', SPIECE_UNDERLINE + 'and', SPIECE_UNDERLINE + 'this',
                                      SPIECE_UNDERLINE + 'is', SPIECE_UNDERLINE + 'f', 'al', 'se', '.'])


if __name__ == '__main__':
    unittest.main()
