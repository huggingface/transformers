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
"""Tests for the SpeechT5 tokenizers."""

import unittest

from transformers import SPIECE_UNDERLINE
from transformers.models.speecht5 import SpeechT5Tokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers.tokenization_utils import AddedToken

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe_char.model")


@require_sentencepiece
@require_tokenizers
class SpeechT5TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "microsoft/speecht5_asr"
    tokenizer_class = SpeechT5Tokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = SpeechT5Tokenizer(SAMPLE_VOCAB)

        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
        tokenizer.mask_token = mask_token
        tokenizer.add_special_tokens({"mask_token": mask_token})
        tokenizer.add_tokens(["<ctc_blank>"])

        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def get_numeric_input_output_texts(self):
        input_text = "I have $123.45 and owe €59.78. My balance is -₴876.90 and have 73% stocks in my company which equals to ₦72649201"
        output_text = "I have one hundred and twenty three point four five dollars and owe fifty nine point seven eight euros. My balance is minus eight hundred and seventy six point nine zero ukrainian hryvnia and have seventy three percent stocks in my company which equals to seventy two million six hundred and forty nine thousand two hundred and one nigerian naira"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        return text, ids

    def test_tokenizer_normalization(self):
        tokenizer = self.get_tokenizer(normalize=True)
        input_text, expected_text = self.get_numeric_input_output_texts()
        input_ids = tokenizer.encode(input_text)
        output_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        self.assertEqual(output_text, expected_text)

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
        self.assertEqual(vocab_keys[-4], "œ")
        self.assertEqual(vocab_keys[-2], "<mask>")
        self.assertEqual(vocab_keys[-1], "<ctc_blank>")
        self.assertEqual(len(vocab_keys), 81)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 79)

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokens[-4])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-3], tokenizer.pad_token_id)

    @unittest.skip
    def test_pickle_subword_regularization_tokenizer(self):
        pass

    @unittest.skip
    def test_subword_regularization_tokenizer(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer(normalize=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE, 'T', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'a', SPIECE_UNDERLINE, 't', 'e', 's', 't'])  # fmt: skip

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [4, 32, 11, 10, 12, 4, 10, 12, 4, 7, 4, 6, 5, 12, 6],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens,[SPIECE_UNDERLINE, 'I', SPIECE_UNDERLINE, 'w', 'a', 's', SPIECE_UNDERLINE, 'b', 'o', 'r', 'n', SPIECE_UNDERLINE, 'i', 'n', SPIECE_UNDERLINE, 'n', 'i', 'n', 'e', 't', 'y', SPIECE_UNDERLINE, 't', 'w', 'o', SPIECE_UNDERLINE, 't', 'h', 'o', 'u', 's', 'a', 'n', 'd', ',', SPIECE_UNDERLINE, 'a', 'n', 'd', SPIECE_UNDERLINE, 't', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'f', 'a', 'l', 's', 'é', '.'])  # fmt: skip

        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [4, 30, 4, 20, 7, 12, 4, 25, 8, 13, 9, 4, 10, 9, 4, 9, 10, 9, 5, 6, 22, 4, 6, 20, 8, 4, 6, 11, 8, 16, 12, 7, 9, 14, 23, 4, 7, 9, 14, 4, 6, 11, 10, 12, 4, 10, 12, 4, 19, 7, 15, 12, 73, 26])  # fmt: skip

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens,[SPIECE_UNDERLINE, 'I', SPIECE_UNDERLINE, 'w', 'a', 's', SPIECE_UNDERLINE, 'b', 'o', 'r', 'n', SPIECE_UNDERLINE, 'i', 'n', SPIECE_UNDERLINE, 'n', 'i', 'n', 'e', 't', 'y', SPIECE_UNDERLINE, 't', 'w', 'o', SPIECE_UNDERLINE, 't', 'h', 'o', 'u', 's', 'a', 'n', 'd', ',', SPIECE_UNDERLINE, 'a', 'n', 'd', SPIECE_UNDERLINE, 't', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'f', 'a', 'l', 's', 'é', '.'])  # fmt: skip

    @slow
    def test_tokenizer_integration(self):
        # Use custom sequence because this tokenizer does not handle numbers.
        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, GPT, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over thirty-two pretrained "
            "models in one hundred plus languages and deep interoperability between Jax, PyTorch and TensorFlow.",
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # fmt: off
        expected_encoding = {
            'input_ids': [
                [4, 32, 13, 7, 9, 12, 19, 8, 13, 18, 5, 13, 12, 4, 64, 19, 8, 13, 18, 5, 13, 15, 22, 4, 28, 9, 8, 20, 9, 4, 7, 12, 4, 24, 22, 6, 8, 13, 17, 11, 39, 6, 13, 7, 9, 12, 19, 8, 13, 18, 5, 13, 12, 4, 7, 9, 14, 4, 24, 22, 6, 8, 13, 17, 11, 39, 24, 13, 5, 6, 13, 7, 10, 9, 5, 14, 39, 25, 5, 13, 6, 63, 4, 24, 13, 8, 27, 10, 14, 5, 12, 4, 21, 5, 9, 5, 13, 7, 15, 39, 24, 16, 13, 24, 8, 12, 5, 4, 7, 13, 17, 11, 10, 6, 5, 17, 6, 16, 13, 5, 12, 4, 64, 40, 47, 54, 32, 23, 4, 53, 49, 32, 23, 4, 54, 8, 40, 47, 54, 32, 7, 23, 4, 69, 52, 43, 23, 4, 51, 10, 12, 6, 10, 15, 40, 5, 13, 6, 23, 4, 69, 52, 48, 5, 6, 26, 26, 26, 63, 4, 19, 8, 13, 4, 48, 7, 6, 16, 13, 7, 15, 4, 52, 7, 9, 21, 16, 7, 21, 5, 4, 61, 9, 14, 5, 13, 12, 6, 7, 9, 14, 10, 9, 21, 4, 64, 48, 52, 61, 63, 4, 7, 9, 14, 4, 48, 7, 6, 16, 13, 7, 15, 4, 52, 7, 9, 21, 16, 7, 21, 5, 4, 53, 5, 9, 5, 13, 7, 6, 10, 8, 9, 4, 64, 48, 52, 53, 63, 4, 20, 10, 6, 11, 4, 8, 27, 5, 13, 4, 6, 11, 10, 13, 6, 22, 39, 6, 20, 8, 4, 24, 13, 5, 6, 13, 7, 10, 9, 5, 14, 4, 18, 8, 14, 5, 15, 12, 4, 10, 9, 4, 8, 9, 5, 4, 11, 16, 9, 14, 13, 5, 14, 4, 24, 15, 16, 12, 4, 15, 7, 9, 21, 16, 7, 21, 5, 12, 4, 7, 9, 14, 4, 14, 5, 5, 24, 4, 10, 9, 6, 5, 13, 8, 24, 5, 13, 7, 25, 10, 15, 10, 6, 22, 4, 25, 5, 6, 20, 5, 5, 9, 4, 58, 7, 37, 23, 4, 49, 22, 32, 8, 13, 17, 11, 4, 7, 9, 14, 4, 32, 5, 9, 12, 8, 13, 55, 15, 8, 20, 26, 2],
                [4, 40, 47, 54, 32, 4, 10, 12, 4, 14, 5, 12, 10, 21, 9, 5, 14, 4, 6, 8, 4, 24, 13, 5, 39, 6, 13, 7, 10, 9, 4, 14, 5, 5, 24, 4, 25, 10, 14, 10, 13, 5, 17, 6, 10, 8, 9, 7, 15, 4, 13, 5, 24, 13, 5, 12, 5, 9, 6, 7, 6, 10, 8, 9, 12, 4, 19, 13, 8, 18, 4, 16, 9, 15, 7, 25, 5, 15, 5, 14, 4, 6, 5, 37, 6, 4, 25, 22, 4, 46, 8, 10, 9, 6, 15, 22, 4, 17, 8, 9, 14, 10, 6, 10, 8, 9, 10, 9, 21, 4, 8, 9, 4, 25, 8, 6, 11, 4, 15, 5, 19, 6, 4, 7, 9, 14, 4, 13, 10, 21, 11, 6, 4, 17, 8, 9, 6, 5, 37, 6, 4, 10, 9, 4, 7, 15, 15, 4, 15, 7, 22, 5, 13, 12, 26, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [4, 32, 11, 5, 4, 45, 16, 10, 17, 28, 4, 25, 13, 8, 20, 9, 4, 19, 8, 37, 4, 46, 16, 18, 24, 12, 4, 8, 27, 5, 13, 4, 6, 11, 5, 4, 15, 7, 57, 22, 4, 14, 8, 21, 26, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        }
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="microsoft/speecht5_asr",
            revision="c5ef64c71905caeccde0e4462ef3f9077224c524",
            sequences=sequences,
        )

    def test_encode_decode(self):
        tokenizer = SpeechT5Tokenizer.from_pretrained("microsoft/speecht5_tts")

        tokens = tokenizer.tokenize("a = b")
        self.assertEqual(tokens, ["▁", "a", "▁", "=", "▁", "b"])

        # the `'='` is unknown.
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(ids, [4, 7, 4, 3, 4, 25])

        # let's make sure decoding with the special unknown tokens preserves spaces
        ids = tokenizer.encode("a = b")
        self.assertEqual(tokenizer.decode(ids), "a <unk> b</s>")
