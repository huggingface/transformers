# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team, The Microsoft Research team.
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

from transformers.models.xlm_prophetnet.tokenization_xlm_prophetnet import SPIECE_UNDERLINE, XLMProphetNetTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, slow
from transformers.utils import cached_property

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
class XLMProphetNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = XLMProphetNetTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = XLMProphetNetTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "[PAD]"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "[PAD]")
        self.assertEqual(vocab_keys[1], "[CLS]")
        self.assertEqual(vocab_keys[-1], "j")
        self.assertEqual(len(vocab_keys), 1_012)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_012)

    def test_full_tokenizer(self):
        tokenizer = XLMProphetNetTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
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
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, -9, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, -9, 4]
            ],
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
                "[UNK]",
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
                "[UNK]",
                ".",
            ],
        )

    @cached_property
    def big_tokenizer(self):
        return XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [35389, 6672, 49, 2]
        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[11073, 82783, 18, 26, 82783, 549, 51540, 248, 17209, 1301, 217, 20, 215186, 1325, 147, 17209, 1301, 217, 20, 56370, 53, 122020, 20, 16477, 27, 87355, 4548, 20, 4728, 78392, 17, 159969, 18, 26, 24491, 629, 15, 538, 22704, 5439, 15, 2788, 24491, 9885, 15, 43534, 605, 15, 814, 18403, 33200, 29, 15, 43534, 24458, 12410, 111, 24966, 83669, 9637, 144068, 26, 850, 22346, 27, 147, 24966, 83669, 83490, 26, 39113, 735, 27, 689, 656, 2800, 1339, 4600, 53, 122020, 115785, 34, 816, 1339, 46887, 18, 147, 53905, 1951, 42238, 41170, 17732, 834, 436, 15, 27523, 98733, 217, 147, 5542, 4981, 930, 17347, 16, 2], [20091, 629, 94, 82786, 58, 490, 20, 1528, 84, 53905, 344, 80592, 110128, 18822, 5267, 1306, 62, 152537, 308, 7997, 401, 124427, 549, 35442, 225, 109, 15055, 25748, 147, 7119, 43712, 34, 767, 135366, 18, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [592, 63784, 119466, 17, 147808, 88214, 18, 656, 81, 32, 3296, 10280, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="microsoft/xprophetnet-large-wiki100-cased",
            revision="1acad1643ddd54a44df6a1b797ada8373685d90e",
        )
