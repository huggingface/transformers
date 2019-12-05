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

from transformers.tokenization_xlnet import (XLNetTokenizer, SPIECE_UNDERLINE)

from .tokenization_tests_commons import CommonTestCases
from .utils import slow

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fixtures/test_sentencepiece.model')

class XLNetTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = XLNetTokenizer

    def setUp(self):
        super(XLNetTokenizationTest, self).setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return XLNetTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"This is a test"
        output_text = u"This is a test"
        return input_text, output_text


    def test_full_tokenizer(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize(u'This is a test')
        self.assertListEqual(tokens, [u'▁This', u'▁is', u'▁a', u'▁t', u'est'])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

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

    @slow
    def test_sequence_builders(self):
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == text + [4, 3]
        assert encoded_pair == text + [4] + text_2 + [4, 3]


if __name__ == '__main__':
    unittest.main()
