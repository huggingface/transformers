# coding=utf-8
# Copyright 2019 Hugging Face inc.
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

from transformers.tokenization_albert import (AlbertTokenizer, SPIECE_UNDERLINE)

from .tokenization_tests_commons import CommonTestCases

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fixtures/spiece.model')

class AlbertTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = AlbertTokenizer

    def setUp(self):
        super(AlbertTokenizationTest, self).setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AlbertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"this is a test"
        output_text = u"this is a test"
        return input_text, output_text


    def test_full_tokenizer(self):
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize(u'This is a test')
        self.assertListEqual(tokens, [u'▁this', u'▁is', u'▁a', u'▁test'])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [48, 25, 21, 1289])

        tokens = tokenizer.tokenize(u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [u'▁i', u'▁was', u'▁born', u'▁in', u'▁9', u'2000', u',', u'▁and', u'▁this', u'▁is', u'▁fal', u's', u'é', u'.'])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [31, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, ['▁i', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', '<unk>', '.'])

    def test_sequence_builders(self):
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [tokenizer.sep_token_id]


if __name__ == '__main__':
    unittest.main()
