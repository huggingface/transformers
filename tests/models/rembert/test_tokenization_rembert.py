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


import tempfile
import unittest

from tests.test_tokenization_common import AddedToken, TokenizerTesterMixin
from transformers import RemBertTokenizer, RemBertTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers


SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class RemBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/rembert"
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

    # Copied from ReformerTokenizationTest.get_input_output_texts
    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())
        self.assertEqual(vocab_keys[0], "<unk>")
        self.assertEqual(vocab_keys[1], "<s>")

        self.assertEqual(vocab_keys[5], "▁the")
        self.assertEqual(vocab_keys[2], "</s>")

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

    def test_full_tokenizer(self):
        tokenizer = RemBertTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual( tokens,  [SPIECE_UNDERLINE + "I",SPIECE_UNDERLINE + "was",SPIECE_UNDERLINE + "b","or","n",SPIECE_UNDERLINE + "in",SPIECE_UNDERLINE + "","9","2","0","0","0",",",SPIECE_UNDERLINE + "and",SPIECE_UNDERLINE + "this",SPIECE_UNDERLINE + "is",SPIECE_UNDERLINE + "f","al","s","é",".",],)  # fmt: skip
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4])

    def test_encode_decode_round_trip(self):
        tokenizer = RemBertTokenizer(SAMPLE_VOCAB, keep_accents=True)

        text = "清水寺は京都にある。"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["▁", "清水寺は京都にある。"])
        encoded_string = tokenizer.encode(text)
        self.assertListEqual(encoded_string, [1000, 7, 0, 1001])
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(decode_text, text)

        text = "That's awesome! 🤩 #HuggingFace,  🌟 Have a great day! 🌈"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual( tokens, ['▁That', "'", 's', '▁a', 'w', 'es', 'ome', '!', '▁', '🤩', '▁', '#', 'H', 'u', 'g', 'g', 'ing', 'F', 'a', 'ce', ',', '▁', '🌟', '▁H', 'a', 've', '▁a', '▁great', '▁day', '!', '▁', '🌈'])  # fmt: skip
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(decode_text, "That's awesome! 🤩 #HuggingFace, 🌟 Have a great day! 🌈")

        text = "In the sky up above"
        tokens = tokenizer._tokenize(text)
        self.assertListEqual(tokens, ["▁In", "▁the", "▁s", "k", "y", "▁up", "▁a", "b", "o", "ve"])  # fmt: skip
        encoded_string = tokenizer.encode(text)
        self.assertListEqual(encoded_string, [1000, 388, 5, 47, 45, 30, 118, 10, 65, 20, 123, 1001])
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, decode_text)

        text = "The cat. . Sat <s>.In a room"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(
            tokens, ["▁The", "▁c", "at", ".", "▁", ".", "▁S", "at", "▁", "<", "s", ">", ".", "I", "n", "▁a", "▁room"]
        )
        encoded_string = tokenizer.encode(text)
        self.assertListEqual(
            encoded_string, [1000, 68, 69, 76, 4, 7, 4, 166, 76, 7, 0, 6, 0, 4, 100, 24, 10, 136, 1001]
        )
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, decode_text)

        text = "Invoice #12345, dated 2023-12-01, is due on 2024-01-15."
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ['▁In', 'v', 'o', 'ic', 'e', '▁', '#', '1', '2', '34', '5', ',', '▁da', 'ted', '▁', '2', '0', '2', '3', '-', '1', '2', '-', '0', '1', ',', '▁is', '▁d', 'u', 'e', '▁on', '▁', '2', '0', '2', '4', '-', '0', '1', '-', '1', '5', '.'])  # fmt: skip
        encoded_string = tokenizer.encode(text)
        self.assertListEqual(encoded_string, [1000, 388, 83, 20, 113, 15, 7, 0, 356, 602, 0, 555, 3, 417, 273, 7, 602, 347, 602, 0, 33, 356, 602, 33, 347, 356, 3, 46, 229, 51, 15, 59, 7, 602, 347, 602, 0, 33, 347, 356, 33, 356, 555, 4, 1001])  # fmt: skip
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, decode_text)

        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit..."
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens,  ['▁', 'L', 'or', 'em', '▁', 'i', 'p', 's', 'um', '▁do', 'l', 'or', '▁sit', '▁am', 'e', 't', ',', '▁con', 'se', 'c', 'te', 't', 'ur', '▁a', 'd', 'i', 'p', 'is', 'c', 'ing', '▁', 'el', 'it', '.', '.', '.'])  # fmt: skip
        encoded_string = tokenizer.encode(text)
        self.assertListEqual( encoded_string,  [1000, 7, 279, 55, 300, 7, 23, 29, 6, 155, 92, 27, 55, 615, 219, 15, 14, 3, 247, 114, 28, 181, 14, 108, 10, 16, 23, 29, 125, 28, 17, 7, 168, 137, 4, 4, 4, 1001] )  # fmt: skip
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, decode_text)

        # for multiple language in one sentence
        text = "Bonjour! Hello! こんにちは!"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["▁B", "on", "j", "o", "ur", "!", "▁He", "ll", "o", "!", "▁", "こんにちは", "!"])
        encoded_string = tokenizer.encode(text)
        self.assertListEqual(encoded_string, [1000, 295, 109, 999, 20, 108, 146, 156, 86, 20, 146, 7, 0, 146, 1001])
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, decode_text)

        text = "Extra spaces\tand\nline breaks\r\nshould be handled."
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ['▁E', 'x', 't', 'r', 'a', '▁sp', 'a', 'ce', 's', '▁and', '▁line', '▁b', 're', 'a', 'k', 's', '▁should', '▁be', '▁hand', 'led', '.'])  # fmt: skip
        encoded_string = tokenizer.encode(text)
        self.assertListEqual(
            encoded_string,
            [1000, 454, 297, 14, 35, 18, 277, 18, 133, 6, 12, 485, 84, 56, 18, 45, 6, 173, 36, 363, 338, 4, 1001],
        )
        decode_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual("Extra spaces and line breaks should be handled.", decode_text)

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
        # Utility to test the added vocab
        def _test_added_vocab_and_eos(expected, tokenizer_class, expected_eos, temp_dir):
            tokenizer = tokenizer_class.from_pretrained(temp_dir)
            self.assertTrue(str(expected_eos) not in tokenizer.additional_special_tokens)
            self.assertIn(new_eos, tokenizer.added_tokens_decoder.values())
            self.assertEqual(tokenizer.added_tokens_decoder[tokenizer.eos_token_id], new_eos)
            self.assertDictEqual(expected, tokenizer.added_tokens_decoder)
            return tokenizer

        new_eos = AddedToken("[NEW_EOS]", rstrip=False, lstrip=True, normalized=False, special=True)
        new_masked_token = AddedToken("[MASK]", lstrip=True, rstrip=False, normalized=False)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                # Load a slow tokenizer from the hub, init with the new token for fast to also include it
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, eos_token=new_eos, mask_token=new_masked_token
                )
                EXPECTED_ADDED_TOKENS_DECODER = tokenizer.added_tokens_decoder
                with self.subTest("Hub -> Slow: Test loading a slow tokenizer from the hub)"):
                    self.assertEqual(tokenizer._eos_token, new_eos)
                    self.assertIn(new_eos, list(tokenizer.added_tokens_decoder.values()))

                with tempfile.TemporaryDirectory() as tmp_dir_2:
                    tokenizer.save_pretrained(tmp_dir_2)
                    with self.subTest(
                        "Hub -> Slow -> Slow: Test saving this slow tokenizer and reloading it in the fast class"
                    ):
                        _test_added_vocab_and_eos(
                            EXPECTED_ADDED_TOKENS_DECODER, self.tokenizer_class, new_eos, tmp_dir_2
                        )

                    if self.rust_tokenizer_class is not None:
                        with self.subTest(
                            "Hub -> Slow -> Fast: Test saving this slow tokenizer and reloading it in the fast class"
                        ):
                            tokenizer_fast = _test_added_vocab_and_eos(
                                EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_2
                            )
                            with tempfile.TemporaryDirectory() as tmp_dir_3:
                                tokenizer_fast.save_pretrained(tmp_dir_3)
                                with self.subTest(
                                    "Hub -> Slow -> Fast -> Fast: Test saving this fast tokenizer and reloading it in the fast class"
                                ):
                                    _test_added_vocab_and_eos(
                                        EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_3
                                    )

                                with self.subTest(
                                    "Hub -> Slow -> Fast -> Slow: Test saving this slow tokenizer and reloading it in the slow class"
                                ):
                                    _test_added_vocab_and_eos(
                                        EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_3
                                    )

                with self.subTest("Hub -> Fast: Test loading a fast tokenizer from the hub)"):
                    if self.rust_tokenizer_class is not None:
                        tokenizer_fast = self.rust_tokenizer_class.from_pretrained(pretrained_name, eos_token=new_eos)
                        self.assertEqual(tokenizer_fast._eos_token, new_eos)
                        self.assertIn(new_eos, list(tokenizer_fast.added_tokens_decoder.values()))
                        # We can't test the following because for BC we kept the default rstrip lstrip in slow not fast. Will comment once normalization is alright
                        with self.subTest("Hub -> Fast == Hub -> Slow: make sure slow and fast tokenizer match"):
                            self.assertDictEqual(EXPECTED_ADDED_TOKENS_DECODER, tokenizer_fast.added_tokens_decoder)

                        EXPECTED_ADDED_TOKENS_DECODER = tokenizer_fast.added_tokens_decoder
                        with tempfile.TemporaryDirectory() as tmp_dir_4:
                            tokenizer_fast.save_pretrained(tmp_dir_4)
                            with self.subTest("Hub -> Fast -> Fast: saving Fast1 locally and loading"):
                                _test_added_vocab_and_eos(
                                    EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_4
                                )

                            with self.subTest("Hub -> Fast -> Slow: saving Fast1 locally and loading"):
                                _test_added_vocab_and_eos(
                                    EXPECTED_ADDED_TOKENS_DECODER, self.tokenizer_class, new_eos, tmp_dir_4
                                )
