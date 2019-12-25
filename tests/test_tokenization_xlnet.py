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


import os
import unittest

from transformers.tokenization_xlnet import SPIECE_UNDERLINE, XLNetTokenizer

from .test_tokenization_common import TokenizerTesterMixin
from .utils import slow


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")


class XLNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = XLNetTokenizer

    def setUp(self):
        super(XLNetTokenizationTest, self).setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return XLNetTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = "This is a test"
        output_text = "This is a test"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, keep_accents=True)

        text = "This is a test"
        tokens = tokenizer.tokenize(text)
        tokens_wo, offsets = tokenizer.tokenize_with_offsets(text)
        self.assertEqual(len(tokens_wo), len(offsets))
        self.assertListEqual(tokens, tokens_wo)
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])
        self.assertListEqual(offsets, [0, 5, 8, 10, 11])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

        text = "I was born in 92000, and this is falsé."
        tokens = tokenizer.tokenize(text)
        tokens_wo, offsets = tokenizer.tokenize_with_offsets(text)
        self.assertEqual(len(tokens_wo), len(offsets))
        self.assertListEqual(tokens, tokens_wo)
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        self.assertListEqual(offsets, [0, 2, 6, 7, 9, 11, 14, 14, 15, 16, 17, 18, 19, 21, 25, 30, 33, 34, 36, 37, 38])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    def test_tokenizer_lower(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, do_lower_case=True)
        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "",
                "i",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "se",
                ".",
            ],
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["▁he", "ll", "o"])

    def test_tokenizer_no_lower(self):
        tokenizer = XLNetTokenizer(SAMPLE_VOCAB, do_lower_case=False)
        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "se",
                ".",
            ],
        )

    @slow
    def test_sequence_builders(self):
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == text + [4, 3]
        assert encoded_pair == text + [4] + text_2 + [4, 3]

    @slow
    def test_tokenize_with_offsets(self):
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

        """ For words that have a wordpiece tokenization that
            doesn't contain the tokenization of its prefixes.
        """
        sentence = "1603"
        expected_tokens = ["▁16", "03"]
        tokens, offsets = tokenizer.tokenize_with_offsets(sentence)
        assert tokens == expected_tokens
        assert offsets == [0, 2]

        """ For cases in which the current token won't be produced
            without an additional character that is only part of the
            text that corresponds to the next tokens.
            Example for XLNet:
            text = "How many points did the buccaneers need to tie in the first?"
            tokens = [..., '▁the', '▁', 'bu', 'cca', 'ne', 'ers', ...]
            target_tokens = ['▁']
            comparison_tokens = ['▁', 'b']
            prev_comparison_tokens = ['']
        """
        sentence = "How many points did the buccaneers need to tie in the first?"
        expected_tokens = [
            "▁How",
            "▁many",
            "▁points",
            "▁did",
            "▁the",
            "▁",
            "bu",
            "cca",
            "ne",
            "ers",
            "▁need",
            "▁to",
            "▁tie",
            "▁in",
            "▁the",
            "▁first",
            "?",
        ]
        tokens, offsets = tokenizer.tokenize_with_offsets(sentence)
        assert tokens == expected_tokens
        assert offsets == [0, 4, 9, 16, 20, 24, 24, 26, 29, 31, 35, 40, 43, 47, 50, 54, 59]
