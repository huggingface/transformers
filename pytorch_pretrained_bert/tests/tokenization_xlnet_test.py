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
import sys
import unittest
from io import open
import shutil
import pytest

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from pytorch_pretrained_bert.tokenization_xlnet import (XLNetTokenizer,
                                                        PRETRAINED_VOCAB_ARCHIVE_MAP,
                                                        SPIECE_UNDERLINE)

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fixtures/test_sentencepiece.model')

class XLNetTokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB)

        tokens = tokenizer.tokenize(u'This is a test')
        self.assertListEqual(tokens, [u'▁This', u'▁is', u'▁a', u'▁t', u'est'])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

        vocab_path = u"/tmp/"
        vocab_file, special_tokens_file = tokenizer.save_vocabulary(vocab_path)
        tokenizer = tokenizer.from_pretrained(vocab_path,
                                              keep_accents=True)

        tokens = tokenizer.tokenize(u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b',
                                      u'or', u'n', SPIECE_UNDERLINE + u'in', SPIECE_UNDERLINE + u'',
                                      u'9', u'2', u'0', u'0', u'0', u',', SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this',
                                      SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u's', u'é', u'.'])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids, [8, 21, 84, 55, 24, 19, 7, 0,
                            602, 347, 347, 347, 3, 12, 66,
                            46, 72, 80, 6, 0, 4])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b',
                                           u'or', u'n', SPIECE_UNDERLINE + u'in',
                                           SPIECE_UNDERLINE + u'', u'<unk>', u'2', u'0', u'0', u'0', u',',
                                           SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this',
                                           SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u's',
                                           u'<unk>', u'.'])

        text = "Munich and Berlin are nice cities"
        filename = u"/tmp/tokenizer.bin"

        subwords = tokenizer.tokenize(text)

        pickle.dump(tokenizer, open(filename, "wb"))

        tokenizer_new = pickle.load(open(filename, "rb"))
        subwords_loaded = tokenizer_new.tokenize(text)

        self.assertListEqual(subwords, subwords_loaded)

        os.remove(filename)
        os.remove(vocab_file)
        os.remove(special_tokens_file)

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
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'', u'i', SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b',
                                      u'or', u'n', SPIECE_UNDERLINE + u'in', SPIECE_UNDERLINE + u'',
                                      u'9', u'2', u'0', u'0', u'0', u',', SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this',
                                      SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u'se', u'.'])
        self.assertListEqual(tokenizer.tokenize(u"H\u00E9llo"), [u"▁he", u"ll", u"o"])

    def test_tokenizer_no_lower(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, do_lower_case=False)
        tokens = tokenizer.tokenize(u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b', u'or',
                                      u'n', SPIECE_UNDERLINE + u'in', SPIECE_UNDERLINE + u'',
                                      u'9', u'2', u'0', u'0', u'0', u',', SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this',
                                      SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u'se', u'.'])


if __name__ == '__main__':
    unittest.main()
