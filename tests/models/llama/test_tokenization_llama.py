# coding=utf-8
# Copyright 2018 Google LLaMa Authors and HuggingFace Inc. team.
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
import json
import os
import re
import tempfile
import unittest

from transformers import SPIECE_UNDERLINE, AddedToken, BatchEncoding, LLaMaTokenizer, LLaMaTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers.utils import cached_property, is_tf_available, is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


@require_sentencepiece
@require_tokenizers
class LLaMaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LLaMaTokenizer
    rust_tokenizer_class = LLaMaTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = LLaMaTokenizer(SAMPLE_VOCAB)
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
        # TODO @thomasw21: LLaMa tokenizer doesn't have pad token
        # self.assertEqual(vocab_keys[-1], "<pad>")
        self.assertEqual(len(vocab_keys), 1_101)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_100)

    def test_full_tokenizer(self):
        tokenizer = LLaMaTokenizer(SAMPLE_VOCAB)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

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
        self.assertListEqual(ids, [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4])

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

    def get_tokenizer(self, **kwargs) -> LLaMaTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, pad_token=None, **kwargs)

    def test_bos_token_in_text_id_considered_as_text(self):
        tokenizer = self.get_tokenizer()
        tokenized_string_bos = tokenizer(["<s>"])
        self.assertListEqual(
            tokenized_string_bos["input_ids"], [1, ]
        )

    def test_tokenized_text_always_starts_with_bos_token(self):
        tokenizer = self.get_tokenizer()
        tokenized_texts = tokenizer(["<s>", "", "Hello my name is John.", "If you want to use bos token, add \"<s>\" in your text input."])
        for tokenized_text in tokenized_texts["input_ids"]:
            self.assertGreaterEqual(len(tokenized_text), 1)
            self.assertEqual(tokenized_text[0], tokenizer.bos_token_id())
            self.assertNotIn(tokenizer.bos_token_id(), tokenized_text[1:])
