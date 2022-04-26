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

import unittest

from transformers import FNetTokenizer, FNetTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow, tooslow
from transformers.tokenization_utils import AddedToken

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class FNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = FNetTokenizer
    rust_tokenizer_class = FNetTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = FNetTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<pad>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "▁eloquent")
        self.assertEqual(len(vocab_keys), 30_000)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 30_000)

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
        tokenizer = FNetTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁", "T", "his", "▁is", "▁a", "▁test"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [13, 1, 4398, 25, 21, 1289])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            ["▁", "I", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "é", "."],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [13, 1, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                "▁",
                "<unk>",
                "▁was",
                "▁born",
                "▁in",
                "▁9",
                "2000",
                ",",
                "▁and",
                "▁this",
                "▁is",
                "▁fal",
                "s",
                "<unk>",
                ".",
            ],
        )

    def test_sequence_builders(self):
        tokenizer = FNetTokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    # Overriden Tests - loading the fast tokenizer from slow just takes too long
    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    p_output = tokenizer_p.encode("Hey this is a <special> token")

                    cr_output = tokenizer_r.encode("Hey this is a <special> token")

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    @tooslow
    def test_special_tokens_initialization_from_slow(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs, from_slow=True
                )
                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]
                tokenizer_p = self.tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )

                p_output = tokenizer_p.encode("Hey this is a <special> token")
                cr_output = tokenizer_r.encode("Hey this is a <special> token")

                self.assertEqual(p_output, cr_output)
                self.assertTrue(special_token_id in p_output)
                self.assertTrue(special_token_id in cr_output)

    # Overriden Tests
    def test_padding(self, max_length=50):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                self.assertEqual(tokenizer_p.pad_token_id, tokenizer_r.pad_token_id)
                pad_token_id = tokenizer_p.pad_token_id

                # Encode - Simple input
                input_r = tokenizer_r.encode("This is a simple input", max_length=max_length, pad_to_max_length=True)
                input_p = tokenizer_p.encode("This is a simple input", max_length=max_length, pad_to_max_length=True)
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode("This is a simple input", max_length=max_length, padding="max_length")
                input_p = tokenizer_p.encode("This is a simple input", max_length=max_length, padding="max_length")
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.encode("This is a simple input", padding="longest")
                input_p = tokenizer_p.encode("This is a simple input", padding=True)
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode - Pair input
                input_r = tokenizer_r.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode("This is a simple input", "This is a pair", padding=True)
                input_p = tokenizer_p.encode("This is a simple input", "This is a pair", padding="longest")
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode_plus - Simple input
                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                input_r = tokenizer_r.encode_plus("This is a simple input", padding="longest")
                input_p = tokenizer_p.encode_plus("This is a simple input", padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                # Encode_plus - Pair input
                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                input_r = tokenizer_r.encode_plus("This is a simple input", "This is a pair", padding="longest")
                input_p = tokenizer_p.encode_plus("This is a simple input", "This is a pair", padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                # Batch_encode_plus - Simple input
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding="longest",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"], padding="longest"
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"], padding=True
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Batch_encode_plus - Pair input
                input_r = tokenizer_r.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    padding=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    padding="longest",
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus("This is a input 1")
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.encode_plus("This is a input 1")
                input_p = tokenizer_r.pad(input_p)

                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus("This is a input 1")
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.encode_plus("This is a input 1")
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                # Using pad after tokenization
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_p = tokenizer_r.pad(input_p)

                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad after tokenization
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

    @slow
    def test_save_pretrained(self):
        super().test_save_pretrained()

    @slow
    def test_save_slow_from_fast_and_reload_fast(self):
        super().test_save_slow_from_fast_and_reload_fast()

    def assert_batch_padded_input_match(
        self,
        input_r: dict,
        input_p: dict,
        max_length: int,
        pad_token_id: int,
        model_main_input_name: str = "input_ids",
    ):
        for i_r in input_r.values():
            self.assertEqual(len(i_r), 2), self.assertEqual(len(i_r[0]), max_length), self.assertEqual(
                len(i_r[1]), max_length
            )
            self.assertEqual(len(i_r), 2), self.assertEqual(len(i_r[0]), max_length), self.assertEqual(
                len(i_r[1]), max_length
            )

        for i_r, i_p in zip(input_r[model_main_input_name], input_p[model_main_input_name]):
            self.assert_padded_input_match(i_r, i_p, max_length, pad_token_id)

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[4, 4616, 107, 163, 328, 14, 63, 1726, 106, 11954, 16659, 23, 83, 16688, 11427, 328, 107, 36, 11954, 16659, 23, 83, 16688, 6153, 82, 961, 16688, 3474, 16710, 1696, 2306, 16688, 10854, 2524, 3827, 561, 163, 3474, 16680, 62, 226, 2092, 16680, 379, 3474, 16660, 16680, 2436, 16667, 16671, 16680, 999, 87, 3474, 16680, 2436, 16667, 5208, 800, 16710, 68, 2018, 2959, 3037, 163, 16663, 11617, 16710, 36, 2018, 2959, 4737, 163, 16663, 16667, 16674, 16710, 91, 372, 5087, 16745, 2205, 82, 961, 3608, 38, 1770, 16745, 7984, 36, 2565, 751, 9017, 1204, 864, 218, 1244, 16680, 11954, 16659, 23, 83, 36, 14686, 23, 7619, 16678, 5], [4, 28, 532, 65, 1929, 33, 391, 16688, 3979, 9, 2565, 7849, 299, 225, 34, 2040, 305, 167, 289, 16667, 16078, 32, 1966, 181, 4626, 63, 10575, 71, 851, 1491, 36, 624, 4757, 38, 208, 8038, 16678, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [4, 13, 1467, 5187, 26, 2521, 4567, 16664, 372, 13, 16209, 3314, 16678, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/fnet-base",
            revision="34219a71ca20e280cc6000b89673a169c65d605c",
        )
