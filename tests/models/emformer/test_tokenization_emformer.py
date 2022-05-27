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

import unittest

from transformers import EmformerTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_torch, slow
from transformers.utils import cached_property

from ...test_tokenization_common import TokenizerTesterMixin


SPIECE_UNDERLINE = "▁"

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
class EmformerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = EmformerTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    def setUp(self):
        super().setUp()

        tokenizer = EmformerTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<unk>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<unk>")
        self.assertEqual(vocab_keys[1], "<s>")
        self.assertEqual(vocab_keys[-1], "<pad>")
        self.assertEqual(len(vocab_keys), 1_001)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

    def test_pretrained_model_lists(self):
        # EmformerModel has no max model length => no testing
        pass

    def test_full_tokenizer(self):
        tokenizer = EmformerTokenizer(SAMPLE_VOCAB, keep_accents=True)

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
        self.assertListEqual(
            ids,
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
        )

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

    @cached_property
    def big_tokenizer(self):
        return EmformerTokenizer.from_pretrained("anton-l/emformer-base-librispeech")

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "hello world"
        original_tokenizer_encodings = [34, 39, 4072, 628]

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenization_base_hard_symbols(self):
        symbols = 'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : -'
        original_tokenizer_encodings = (
            [4068, 3, 4075, 29, 95, 6, 271, 368, 305, 833, 86, 6, 2700, 27, 80, 874, 1375, 735, 3, 378, 76, 3]
            + [4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068, 3, 4068]
            + [3, 4068, 3]
        )

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenizer_integration(self):
        # the LibriSpeech ASR tokenizer handles only lowercase letters
        sequences = [
            "transformers formerly known as pytorchtransformers and pytorchpretrainedbert provides "
            "generalpurpose architectures bert gpt roberta xlm distilbert xlnet for natural language "
            "understanding nlu and natural language generation nlg with over pretrained models in languages "
            "and deep interoperability between jax pytorch and tensorflow",
            "bert is designed to pretrain deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers",
            "the quick brown fox jumps over the lazy dog",
        ]

        # fmt: off
        expected_encoding = {'input_ids': [[1542, 1811, 184, 2135, 61, 1200, 76, 33, 4086, 4070, 35, 117, 4070, 4077, 427, 1811, 184, 24, 33, 4086, 4070, 35, 117, 4087, 651, 4077, 672, 1310, 1398, 1104, 1193, 4087, 89, 4087, 274, 2828, 288, 119, 878, 14, 354, 45, 315, 232, 1310, 4071, 4068, 4092, 4079, 4081, 581, 124, 1310, 4068, 4092, 4079, 4073, 77, 78, 1776, 2985, 3982, 44, 4079, 4080, 24, 1776, 2985, 760, 174, 44, 4079, 4085, 86, 314, 1265, 4077, 672, 2636, 1755, 38, 2272, 1264, 24, 1116, 835, 278, 16, 3628, 804, 147, 2328, 33, 4086, 4070, 35, 117, 24, 4, 539, 35, 4084, 4079, 54], [14, 354, 95, 3277, 23, 26, 1265, 4077, 114, 1116, 14, 63, 350, 840, 83, 3227, 621, 150, 236, 4079, 3539, 23, 305, 833, 134, 493, 456, 61, 2257, 31, 66, 793, 606, 24, 556, 141, 369, 833, 38, 130, 908, 184, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 1205, 1942, 3474, 147, 213, 749, 314, 7, 537, 3252, 1832, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
        # fmt: on

        self.tokenizer_integration_test_util(
            sequences=sequences,
            expected_encoding=expected_encoding,
            model_name="anton-l/emformer-base-librispeech",
            revision="9fd1bb6b9fc7df4cb982de1f4a4b063b41d3cf66",
        )
