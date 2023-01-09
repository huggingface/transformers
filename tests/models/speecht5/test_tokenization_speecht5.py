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
"""Tests for the SpeechT5 tokenizer."""

import unittest

from transformers import SPIECE_UNDERLINE#, is_sentencepiece_available
from transformers.models.speecht5 import SpeechT5Tokenizer
# from transformers.models.speecht5.tokenization_speecht5 import VOCAB_FILES_NAMES
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe_char.model")

# if is_sentencepiece_available():
#     import sentencepiece as sp


@require_sentencepiece
@require_tokenizers
class SpeechT5TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SpeechT5Tokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = SpeechT5Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-2], "œ")
        self.assertEqual(len(vocab_keys), 79)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 79)

    def test_add_tokens_tokenizer(self):
        pass

    def test_internal_consistency(self):
        pass

    def test_maximum_encoding_length_pair_input(self):
        pass

    def test_maximum_encoding_length_single_input(self):
        pass

    def test_pickle_subword_regularization_tokenizer(self):
        pass

    def test_pretokenized_inputs(self):
        pass

    def test_subword_regularization_tokenizer(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = SpeechT5Tokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("This is a test")
        # fmt: off
        self.assertListEqual(tokens, [SPIECE_UNDERLINE, 'T', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'a', SPIECE_UNDERLINE, 't', 'e', 's', 't'])
        # fmt: on

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [4, 32, 11, 10, 12, 4, 10, 12, 4, 7, 4, 6, 5, 12, 6],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            # fmt: off
            [SPIECE_UNDERLINE, 'I', SPIECE_UNDERLINE, 'w', 'a', 's', SPIECE_UNDERLINE, 'b', 'o', 'r', 'n', SPIECE_UNDERLINE, 'i', 'n', SPIECE_UNDERLINE, '92000', ',', SPIECE_UNDERLINE, 'a', 'n', 'd', SPIECE_UNDERLINE, 't', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'f', 'a', 'l', 's', 'é', '.']
            # fmt: on
        )

        ids = tokenizer.convert_tokens_to_ids(tokens)
        # fmt: off
        self.assertListEqual(ids, [4, 30, 4, 20, 7, 12, 4, 25, 8, 13, 9, 4, 10, 9, 4, 3, 23, 4, 7, 9, 14, 4, 6, 11, 10, 12, 4, 10, 12, 4, 19, 7, 15, 12, 73, 26])
        # fmt: on

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            # fmt: off
            [SPIECE_UNDERLINE, 'I', SPIECE_UNDERLINE, 'w', 'a', 's', SPIECE_UNDERLINE, 'b', 'o', 'r', 'n', SPIECE_UNDERLINE, 'i', 'n', SPIECE_UNDERLINE, '<unk>', ',', SPIECE_UNDERLINE, 'a', 'n', 'd', SPIECE_UNDERLINE, 't', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'f', 'a', 'l', 's', 'é', '.']
            # fmt: on
        )

    # @slow
    # def test_tokenizer_integration(self):
    #     # fmt: off
    #     expected_encoding = {'input_ids': [[3791, 797, 31, 11, 64, 797, 31, 2429, 433, 12, 1176, 12, 20, 786, 915, 142, 2413, 240, 37, 3238, 797, 31, 11, 35, 93, 915, 142, 2413, 240, 37, 5540, 567, 1276, 93, 37, 610, 40, 62, 455, 657, 1042, 123, 780, 177, 37, 309, 241, 1298, 514, 20, 292, 2737, 114, 2469, 241, 85, 64, 302, 548, 528, 423, 4, 509, 406, 423, 37, 601, 4, 777, 302, 548, 528, 423, 284, 4, 3388, 511, 459, 4, 3555, 40, 321, 302, 705, 4, 3388, 511, 583, 326, 5, 5, 5, 62, 3310, 560, 177, 2680, 217, 1508, 32, 31, 853, 418, 64, 583, 511, 1605, 62, 35, 93, 560, 177, 2680, 217, 1508, 1521, 64, 583, 511, 519, 62, 20, 1515, 764, 20, 149, 261, 5625, 7972, 20, 5540, 567, 1276, 93, 3925, 1675, 11, 15, 802, 7972, 576, 217, 1508, 11, 35, 93, 1253, 2441, 15, 289, 652, 31, 416, 321, 3842, 115, 40, 911, 8, 476, 619, 4, 380, 142, 423, 335, 240, 35, 93, 264, 8, 11, 335, 569, 420, 163, 5, 2], [260, 548, 528, 423, 20, 451, 20, 2681, 1153, 3434, 20, 5540, 37, 567, 126, 1253, 2441, 3376, 449, 210, 431, 1563, 177, 767, 5540, 11, 1203, 472, 11, 2953, 685, 285, 364, 706, 1153, 20, 6799, 20, 2869, 20, 4464, 126, 40, 2429, 20, 1040, 866, 2664, 418, 20, 318, 20, 1726, 186, 20, 265, 522, 35, 93, 2191, 4634, 20, 1040, 12, 6799, 15, 228, 2356, 142, 31, 11, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2575, 2666, 684, 1582, 1176, 12, 627, 149, 619, 20, 4902, 563, 11, 20, 149, 261, 3420, 2356, 174, 142, 4714, 131, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
    #     # fmt: on

    #     self.tokenizer_integration_test_util(
    #         expected_encoding=expected_encoding,
    #         model_name="Matthijs/speecht5_asr",
    #         revision="63f2ee29f0d4653c691069ed190536fba977678b",
    #     )
