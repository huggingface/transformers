# coding=utf-8
# Copyright 2022 Hugging Face inc.
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

from transformers import GPTSw3Tokenizer
from transformers.testing_utils import get_tests_dir, require_jinja, require_sentencepiece, require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_with_bytefallback.model")


@require_sentencepiece
@require_tokenizers
class GPTSw3TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "AI-Sweden-Models/gpt-sw3-126m"
    tokenizer_class = GPTSw3Tokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True
    test_sentencepiece_ignore_case = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = GPTSw3Tokenizer(SAMPLE_VOCAB, eos_token="<unk>", bos_token="<unk>", pad_token="<unk>")

        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "This is a test"
        output_text = "This is a test"
        return input_text, output_text

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
        self.assertEqual(len(vocab_keys), 2_000)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 2_000)

    def test_full_tokenizer(self):
        tokenizer = GPTSw3Tokenizer(SAMPLE_VOCAB)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [465, 287, 265, 631, 842])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        # fmt: off
        self.assertListEqual(
            tokens,
            ["▁I", "▁was", "▁bor", "n", "▁in", "▁", "<0x39>", "2", "0", "0", "0", ",", "▁and", "▁this", "▁is", "▁f", "al", "s", "<0xC3>", "<0xA9>", "."],
        )
        # fmt: on

        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [262, 272, 1525, 286, 271, 268, 60, 916, 633, 633, 633, 259, 266, 301, 287, 384, 367, 263, 198, 172, 260],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        # fmt: off
        self.assertListEqual(
            back_tokens,
            ["▁I", "▁was", "▁bor", "n", "▁in", "▁", "<0x39>", "2", "0", "0", "0", ",", "▁and", "▁this", "▁is", "▁f", "al", "s", "<0xC3>", "<0xA9>", "."]
        )
        # fmt: on

    def test_fast_encode_decode(self):
        tokenizer = GPTSw3Tokenizer(SAMPLE_VOCAB)
        texts = ["This is a test", "I was born in 92000, and this is falsé."]
        expected_ids_list = [
            [465, 287, 265, 631, 842],
            [262, 272, 1525, 286, 271, 268, 60, 916, 633, 633, 633, 259, 266, 301, 287, 384, 367, 263, 198, 172, 260],
        ]

        # Test that encode_fast returns the same as tokenize + convert_tokens_to_ids
        for text, expected_ids in zip(texts, expected_ids_list):
            self.assertListEqual(tokenizer.encode_fast(text), expected_ids)

        # Test that decode_fast returns the input text
        for text, token_ids in zip(texts, expected_ids_list):
            self.assertEqual(tokenizer.decode_fast(token_ids), text)

    @slow
    def test_tokenizer_integration(self):
        sequences = [
            "<|python|>def fibonacci(n)\n    if n < 0:\n        print('Incorrect input')",
            "Hey there, how are you doing this fine day?",
            "This is a text with a trailing spaces followed by a dot     .",
            "Häj sväjs lillebrör! =)",
            "Det är inget fel på Mr. Cool",
        ]

        expected_encoding = {"input_ids": [[63423, 5, 6811, 14954, 282, 816, 3821, 63466, 63425, 63462, 18, 63978, 678, 301, 1320, 63423, 63455, 63458, 18, 63982, 4246, 3940, 1901, 47789, 5547, 18994], [19630, 1100, 63446, 1342, 633, 544, 4488, 593, 5102, 2416, 63495, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1652, 428, 268, 1936, 515, 268, 58593, 22413, 9106, 546, 268, 33213, 63979, 698, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [55130, 63450, 924, 63449, 2249, 4062, 1558, 318, 63504, 21498, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [509, 377, 2827, 2559, 332, 6575, 63443, 26801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # fmt: skip
        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="AI-Sweden-Models/gpt-sw3-126m",
            sequences=sequences,
        )

    @require_jinja
    def test_tokenization_for_chat(self):
        tokenizer = GPTSw3Tokenizer(SAMPLE_VOCAB)
        # This is in English, but it's just here to make sure the chat control tokens are being added properly
        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
            [{"role": "assistant", "content": "Nice to meet you."}, {"role": "user", "content": "Hello!"}],
        ]
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        # fmt: off
        expected_tokens = [
            [2000, 1, 575, 541, 419, 530, 339, 265, 878, 708, 727, 275, 347, 541, 260, 1, 968, 263, 314, 419, 366, 354, 294, 360, 1, 575, 541, 419],
            [2000, 1, 575, 541, 419, 530, 339, 265, 878, 708, 727, 275, 347, 541, 260, 1, 968, 263, 314, 419, 366, 354, 294, 360, 1, 575, 541, 419, 984, 429, 281, 264, 1261, 291, 260, 1, 575, 541, 419],
            [2000, 1, 575, 541, 419, 984, 429, 281, 264, 1261, 291, 260, 1, 968, 263, 314, 419, 366, 354, 294, 360, 1, 575, 541, 419]
            ]
        # fmt: on
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)
