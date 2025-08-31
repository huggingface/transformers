# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.models.dia import DiaTokenizer
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


# Special tokens
PAD = 0
S1 = 1
S2 = 2


class DiaTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = DiaTokenizer
    test_rust_tokenizer = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = DiaTokenizer()
        tokenizer.save_pretrained(cls.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "i"
        token_id = 105

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[PAD], "<pad>")
        self.assertEqual(vocab_keys[S1], "[S1]")
        self.assertEqual(vocab_keys[S2], "[S2]")
        self.assertEqual(len(vocab_keys), 256)

    def test_vocab_size(self):
        # utf-8 == 2**8 == 256
        self.assertEqual(self.get_tokenizer().vocab_size, 256)

    def test_full_tokenizer(self):
        tokenizer = DiaTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("Hello, world!")
        self.assertListEqual(tokens, ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33])
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"])

        tokens = tokenizer.tokenize("[S1] Hello [S2] Hello<pad>")
        self.assertListEqual(
            tokens,
            ["[S1]", " ", "H", "e", "l", "l", "o", " ", "[S2]", " ", "H", "e", "l", "l", "o", "<pad>"],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [S1, 32, 72, 101, 108, 108, 111, 32, S2, 32, 72, 101, 108, 108, 111, PAD])
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens, ["[S1]", " ", "H", "e", "l", "l", "o", " ", "[S2]", " ", "H", "e", "l", "l", "o", "<pad>"]
        )

    @slow
    def test_tokenizer_integration(self):
        # Overwritten as decoding will lead to all single bytes (i.e. characters) while usually the string format is expected
        expected_encoding = {'input_ids': [[84, 114, 97, 110, 115, 102, 111, 114, 109, 101, 114, 115, 32, 40, 102, 111, 114, 109, 101, 114, 108, 121, 32, 107, 110, 111, 119, 110, 32, 97, 115, 32, 112, 121, 116, 111, 114, 99, 104, 45, 116, 114, 97, 110, 115, 102, 111, 114, 109, 101, 114, 115, 32, 97, 110, 100, 32, 112, 121, 116, 111, 114, 99, 104, 45, 112, 114, 101, 116, 114, 97, 105, 110, 101, 100, 45, 98, 101, 114, 116, 41, 32, 112, 114, 111, 118, 105, 100, 101, 115, 32, 103, 101, 110, 101, 114, 97, 108, 45, 112, 117, 114, 112, 111, 115, 101, 32, 97, 114, 99, 104, 105, 116, 101, 99, 116, 117, 114, 101, 115, 32, 40, 66, 69, 82, 84, 44, 32, 71, 80, 84, 45, 50, 44, 32, 82, 111, 66, 69, 82, 84, 97, 44, 32, 88, 76, 77, 44, 32, 68, 105, 115, 116, 105, 108, 66, 101, 114, 116, 44, 32, 88, 76, 78, 101, 116, 46, 46, 46, 41, 32, 102, 111, 114, 32, 78, 97, 116, 117, 114, 97, 108, 32, 76, 97, 110, 103, 117, 97, 103, 101, 32, 85, 110, 100, 101, 114, 115, 116, 97, 110, 100, 105, 110, 103, 32, 40, 78, 76, 85, 41, 32, 97, 110, 100, 32, 78, 97, 116, 117, 114, 97, 108, 32, 76, 97, 110, 103, 117, 97, 103, 101, 32, 71, 101, 110, 101, 114, 97, 116, 105, 111, 110, 32, 40, 78, 76, 71, 41, 32, 119, 105, 116, 104, 32, 111, 118, 101, 114, 32, 51, 50, 43, 32, 112, 114, 101, 116, 114, 97, 105, 110, 101, 100, 32, 109, 111, 100, 101, 108, 115, 32, 105, 110, 32, 49, 48, 48, 43, 32, 108, 97, 110, 103, 117, 97, 103, 101, 115, 32, 97, 110, 100, 32, 100, 101, 101, 112, 32, 105, 110, 116, 101, 114, 111, 112, 101, 114, 97, 98, 105, 108, 105, 116, 121, 32, 98, 101, 116, 119, 101, 101, 110, 32, 74, 97, 120, 44, 32, 80, 121, 84, 111, 114, 99, 104, 32, 97, 110, 100, 32, 84, 101, 110, 115, 111, 114, 70, 108, 111, 119, 46], [66, 69, 82, 84, 32, 105, 115, 32, 100, 101, 115, 105, 103, 110, 101, 100, 32, 116, 111, 32, 112, 114, 101, 45, 116, 114, 97, 105, 110, 32, 100, 101, 101, 112, 32, 98, 105, 100, 105, 114, 101, 99, 116, 105, 111, 110, 97, 108, 32, 114, 101, 112, 114, 101, 115, 101, 110, 116, 97, 116, 105, 111, 110, 115, 32, 102, 114, 111, 109, 32, 117, 110, 108, 97, 98, 101, 108, 101, 100, 32, 116, 101, 120, 116, 32, 98, 121, 32, 106, 111, 105, 110, 116, 108, 121, 32, 99, 111, 110, 100, 105, 116, 105, 111, 110, 105, 110, 103, 32, 111, 110, 32, 98, 111, 116, 104, 32, 108, 101, 102, 116, 32, 97, 110, 100, 32, 114, 105, 103, 104, 116, 32, 99, 111, 110, 116, 101, 120, 116, 32, 105, 110, 32, 97, 108, 108, 32, 108, 97, 121, 101, 114, 115, 46], [84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 32, 106, 117, 109, 112, 115, 32, 111, 118, 101, 114, 32, 116, 104, 101, 32, 108, 97, 122, 121, 32, 100, 111, 103, 46]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained "
            "models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained("AntonV/Dia-1.6B")

            encoding = tokenizer(sequences)
            encoding_data = encoding.data
            self.assertDictEqual(encoding_data, expected_encoding)

            # Byte decoding leads to characters so we need to join them
            decoded_sequences = [
                "".join(tokenizer.decode(seq, skip_special_tokens=True)) for seq in encoding["input_ids"]
            ]

            for expected, decoded in zip(sequences, decoded_sequences):
                if self.test_sentencepiece_ignore_case:
                    expected = expected.lower()
                self.assertEqual(expected, decoded)

    @unittest.skip(reason="Dia relies on whole input string due to the byte-level nature.")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip
    def test_tokenizer_slow_store_full_signature(self):
        pass
