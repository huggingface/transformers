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


import os
import unittest

from transformers.tokenization_albert import AlbertTokenizer

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/spiece.model")


class AlbertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AlbertTokenizer

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁this", "▁is", "▁a", "▁test"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [48, 25, 21, 1289])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens, ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "é", "."]
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [31, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "."],
        )

    def test_sequence_builders(self):
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]
