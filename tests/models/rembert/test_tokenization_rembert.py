# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
""" Testing suite for the RemBert tokenizer. """


import unittest

from transformers import RemBertTokenizer, RemBertTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class RemBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RemBertTokenizer
    rust_tokenizer_class = RemBertTokenizerFast
    space_between_special_tokens = True
    test_rust_tokenizer = True
    test_sentencepiece_ignore_case = True
    pre_trained_model_path = "google/rembert"

    def setUp(self):
        super().setUp()

        tokenizer = RemBertTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""

        # self.assertEqual(1000, self.sp_.GetPieceSize())
        self.assertEqual(0, self.get_tokenizer()._convert_token_to_id("<unk>"))
        self.assertEqual(1, self.get_tokenizer()._convert_token_to_id("<s>"))
        self.assertEqual(2, self.get_tokenizer()._convert_token_to_id("</s>"))
        self.assertEqual("<unk>", self.get_tokenizer()._convert_id_to_token(0))
        self.assertEqual("<s>", self.get_tokenizer()._convert_id_to_token(1))
        self.assertEqual("</s>", self.get_tokenizer()._convert_id_to_token(2))

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())
        self.assertEqual(vocab_keys[0], "<unk>")
        self.assertEqual(vocab_keys[1], "<s>")

        self.assertEqual(vocab_keys[5], "▁the")
        self.assertEqual(vocab_keys[2], "</s>")

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I was born in 92000, and this is falsé."

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_full_tokenizer(self):
        tokenizer = RemBertTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

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
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4])

    def test_encode_decode_round_trip(self):
        tokenizer = RemBertTokenizer(SAMPLE_VOCAB, keep_accents=True)
        text = "清水寺は京都にある。"
        ids = tokenizer._tokenize(text)
        text_decode = tokenizer.convert_tokens_to_string(ids)
        self.assertEquals(text_decode, text)

        text = "In the sky up above"
        encode_text = tokenizer._tokenize(text)
        decode_text = tokenizer.convert_tokens_to_string(encode_text)
        self.assertEqual(text, decode_text)

    def test_sequence_builders(self):
        tokenizer = RemBertTokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    def test_added_tokens_serialization(self):
        pass
