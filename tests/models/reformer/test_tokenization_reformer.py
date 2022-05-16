# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import SPIECE_UNDERLINE, ReformerTokenizer, ReformerTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, require_torch, slow
from transformers.utils import cached_property

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class ReformerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ReformerTokenizer
    rust_tokenizer_class = ReformerTokenizerFast
    test_rust_tokenizer = True
    test_seq2seq = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        tokenizer = ReformerTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<s>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<unk>")
        self.assertEqual(vocab_keys[1], "<s>")
        self.assertEqual(vocab_keys[-1], "j")
        self.assertEqual(len(vocab_keys), 1_000)

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

    def test_padding(self, max_length=15):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Simple input
                s = "This is a simple input"
                s2 = ["This is a simple input 1", "This is a simple input 2"]
                p = ("This is a simple input", "This is a pair")
                p2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    s2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    p2,
                    max_length=max_length,
                    padding="max_length",
                )

    # tokenizer has no padding token
    def test_padding_different_model_input_name(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = ReformerTokenizer(SAMPLE_VOCAB, keep_accents=True)

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
        return ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [126, 32, 262, 152, 38, 72, 287]

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenization_base_hard_symbols(self):
        symbols = (
            'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will'
            " add words that should not exsist and be tokenized to <unk>, such as saoneuhaoesuth"
        )
        original_tokenizer_encodings = [
            108,
            265,
            24,
            111,
            4,
            258,
            156,
            35,
            28,
            275,
            3,
            259,
            297,
            260,
            84,
            4,
            35,
            110,
            44,
            8,
            259,
            91,
            268,
            21,
            11,
            209,
            274,
            109,
            266,
            277,
            117,
            86,
            93,
            315,
            258,
            278,
            258,
            277,
            258,
            0,
            258,
            288,
            258,
            319,
            258,
            0,
            258,
            0,
            258,
            0,
            258,
            0,
            258,
            287,
            258,
            315,
            258,
            289,
            258,
            278,
            99,
            269,
            266,
            262,
            8,
            259,
            241,
            4,
            217,
            230,
            268,
            266,
            55,
            168,
            106,
            75,
            193,
            266,
            223,
            27,
            49,
            26,
            282,
            25,
            264,
            299,
            19,
            26,
            0,
            258,
            277,
            117,
            86,
            93,
            176,
            183,
            270,
            11,
            262,
            42,
            61,
            265,
        ]

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @require_torch
    @slow
    def test_torch_encode_plus_sent_to_model(self):
        import torch

        from transformers import ReformerConfig, ReformerModel

        # Build sequence
        first_ten_tokens = list(self.big_tokenizer.get_vocab().keys())[:10]
        sequence = " ".join(first_ten_tokens)
        encoded_sequence = self.big_tokenizer.encode_plus(sequence, return_tensors="pt")
        batch_encoded_sequence = self.big_tokenizer.batch_encode_plus([sequence, sequence], return_tensors="pt")

        config = ReformerConfig()
        # The input gets padded during training so adjust the axial position encodings from the pretrained model value of (512, 1024)
        config.axial_pos_shape = encoded_sequence["input_ids"].shape
        model = ReformerModel(config)

        # Reformer has config.vocab_size == tokenizer.vocab_size == len(tokenizer) - 1 = 320; len(tokenizer) is 321 (including a pad token with id 320)
        assert model.get_input_embeddings().weight.shape[0] >= self.big_tokenizer.vocab_size

        with torch.no_grad():
            model(**encoded_sequence)
            model(**batch_encoded_sequence)

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[108, 265, 24, 111, 4, 258, 156, 7, 51, 279, 58, 7, 76, 25, 69, 278], [140, 243, 264, 134, 17, 267, 77, 263, 22, 262, 297, 258, 304, 177, 279, 266, 14, 89, 13, 35, 261, 299, 272, 137, 275, 278]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # noqa: E501
        # fmt: on

        # This tokenizer does not know some characters like ")".
        # That is the reason why we use very simple texts here.
        # Also see https://github.com/huggingface/transformers/pull/11737#issuecomment-850769064
        sequences = [
            "This is a very simple sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/reformer-crime-and-punishment",
            revision="0e6c3decb8211d49bf881013425dc8b0448b3f5a",
            padding=False,
            sequences=sequences,
        )
