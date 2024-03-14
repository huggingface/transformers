# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the FastSpeech2Conformer tokenizer."""

import unittest

from transformers.models.fastspeech2_conformer import FastSpeech2ConformerTokenizer
from transformers.testing_utils import require_g2p_en, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_g2p_en
class FastSpeech2ConformerTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "espnet/fastspeech2_conformer"
    tokenizer_class = FastSpeech2ConformerTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()
        tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    # Custom `get_clean_sequence` since FastSpeech2ConformerTokenizer can't decode id -> string
    def get_clean_sequence(self, tokenizer, with_prefix_space=False, **kwargs):  # max_length=20, min_length=5
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        return output_text, ids

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<unk>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<blank>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-4], "UH0")
        self.assertEqual(vocab_keys[-2], "..")
        self.assertEqual(vocab_keys[-1], "<sos/eos>")
        self.assertEqual(len(vocab_keys), 78)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 78)

    @unittest.skip(
        "FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_added_token_are_matched_longest_first(self):
        pass

    @unittest.skip(
        "FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_added_tokens_do_lower_case(self):
        pass

    @unittest.skip(
        "FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_tokenize_special_tokens(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("This is a test")
        ids = [9, 12, 6, 12, 11, 2, 4, 15, 6, 4, 77]
        self.assertListEqual(tokens, ["DH", "IH1", "S", "IH1", "Z", "AH0", "T", "EH1", "S", "T", "<sos/eos>"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), ids)
        self.assertListEqual(tokenizer.convert_ids_to_tokens(ids), tokens)

    @slow
    def test_tokenizer_integration(self):
        # Custom test since:
        # 1) This tokenizer only decodes to tokens (phonemes cannot be converted to text with complete accuracy)
        # 2) Uses a sequence without numbers since espnet has different, custom number conversion.
        # This tokenizer can phonemize numbers, but where in espnet "32" is phonemized as "thirty two",
        # here "32" is phonemized as "thirty-two" because we haven't implemented the custom number handling.

        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, GPT, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over thirty-two pretrained "
            "models in one hundred plus languages and deep interoperability between Jax, PyTorch and TensorFlow.",
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        tokenizer = FastSpeech2ConformerTokenizer.from_pretrained(
            "espnet/fastspeech2_conformer", revision="07f9c4a2d6bbc69b277d87d2202ad1e35b05e113"
        )
        actual_encoding = tokenizer(sequences)

        # fmt: off
        expected_encoding = {
            'input_ids': [
                [4, 7, 60, 3, 6, 22, 30, 7, 14, 21, 11, 22, 30, 7, 14, 21, 8, 29, 3, 34, 3, 18, 11, 17, 12, 4, 21, 10, 4, 7, 60, 3, 6, 22, 30, 7, 14, 21, 11, 2, 3, 5, 17, 12, 4, 21, 10, 17, 7, 29, 4, 7, 31, 3, 5, 25, 38, 4, 17, 7, 2, 20, 32, 5, 11, 40, 15, 3, 21, 2, 8, 17, 38, 17, 2, 6, 24, 7, 10, 2, 4, 45, 10, 39, 21, 11, 25, 38, 4, 23, 37, 15, 4, 6, 23, 7, 2, 25, 38, 4, 2, 23, 11, 8, 15, 14, 11, 23, 5, 13, 6, 4, 12, 8, 4, 21, 25, 23, 11, 8, 15, 3, 39, 2, 8, 1, 22, 30, 7, 3, 18, 39, 21, 2, 8, 8, 18, 36, 37, 16, 2, 40, 62, 3, 5, 21, 6, 4, 18, 3, 5, 13, 36, 3, 8, 28, 2, 3, 5, 3, 18, 39, 21, 2, 8, 8, 18, 36, 37, 16, 2, 40, 40, 45, 3, 21, 31, 35, 2, 3, 15, 8, 36, 16, 12, 9, 34, 20, 21, 43, 38, 5, 29, 4, 28, 17, 7, 29, 4, 7, 31, 3, 5, 14, 24, 5, 2, 8, 11, 13, 3, 16, 19, 3, 26, 19, 3, 5, 7, 2, 5, 17, 8, 19, 6, 8, 18, 36, 37, 16, 2, 40, 2, 11, 2, 3, 5, 5, 27, 17, 49, 3, 4, 21, 2, 17, 21, 25, 12, 8, 2, 4, 29, 25, 13, 4, 16, 27, 3, 40, 18, 10, 6, 23, 17, 12, 4, 21, 10, 2, 3, 5, 4, 15, 3, 6, 21, 8, 46, 22, 33, 77],
                [25, 38, 4, 12, 11, 5, 13, 11, 32, 3, 5, 4, 28, 17, 7, 27, 4, 7, 31, 3, 5, 27, 17, 25, 51, 5, 13, 7, 15, 10, 35, 2, 3, 2, 8, 7, 45, 17, 7, 2, 11, 2, 3, 4, 31, 35, 2, 3, 11, 22, 7, 19, 14, 2, 3, 8, 31, 25, 2, 8, 5, 4, 15, 10, 6, 4, 25, 32, 40, 55, 3, 4, 8, 29, 10, 2, 3, 5, 12, 35, 2, 3, 13, 36, 24, 3, 25, 34, 43, 8, 15, 22, 4, 2, 3, 5, 7, 32, 4, 10, 24, 3, 4, 54, 10, 6, 4, 13, 3, 30, 8, 8, 31, 21, 11, 33, 77],
                [9, 2, 10, 16, 12, 10, 25, 7, 42, 3, 22, 24, 10, 6, 40, 19, 14, 17, 6, 34, 20, 21, 9, 2, 8, 31, 11, 29, 5, 30, 37, 33, 77]
            ],
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ]
        }
        # fmt: on

        actual_tokens = [tokenizer.decode(input_ids) for input_ids in expected_encoding["input_ids"]]
        expected_tokens = [
            [tokenizer.convert_ids_to_tokens(id) for id in sequence] for sequence in expected_encoding["input_ids"]
        ]

        self.assertListEqual(actual_encoding["input_ids"], expected_encoding["input_ids"])
        self.assertListEqual(actual_encoding["attention_mask"], expected_encoding["attention_mask"])
        self.assertTrue(actual_tokens == expected_tokens)

    @unittest.skip(
        reason="FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip(
        reason="FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_add_special_tokens(self):
        pass

    @unittest.skip(
        reason="FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_added_token_serializable(self):
        pass

    @unittest.skip(
        reason="FastSpeech2Conformer tokenizer does not support adding tokens as they can't be added to the g2p_en backend"
    )
    def test_save_and_load_tokenizer(self):
        pass

    @unittest.skip(reason="Phonemes cannot be reliably converted to string due to one-many mapping")
    def test_internal_consistency(self):
        pass

    @unittest.skip(reason="Phonemes cannot be reliably converted to string due to one-many mapping")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip(reason="Phonemes cannot be reliably converted to string due to one-many mapping")
    def test_convert_tokens_to_string_format(self):
        pass

    @unittest.skip("FastSpeech2Conformer tokenizer does not support pairs.")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip(
        "FastSpeech2Conformer tokenizer appends eos_token to each string it's passed, including `is_split_into_words=True`."
    )
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip(
        reason="g2p_en is slow is with large inputs and max encoding length is not a concern for FastSpeech2Conformer"
    )
    def test_maximum_encoding_length_single_input(self):
        pass
