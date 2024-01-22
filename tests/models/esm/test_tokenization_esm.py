# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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


import os
import tempfile
import unittest
from typing import List

from transformers.models.esm.tokenization_esm import VOCAB_FILES_NAMES, EsmTokenizer
from transformers.testing_utils import require_tokenizers
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@require_tokenizers
class ESMTokenizationTest(unittest.TestCase):
    tokenizer_class = EsmTokenizer

    def setUp(self):
        super().setUp()
        self.tmpdirname = tempfile.mkdtemp()
        vocab_tokens: List[str] = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]  # fmt: skip
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizers(self, **kwargs) -> List[PreTrainedTokenizerBase]:
        return [self.get_tokenizer(**kwargs)]

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_tokenizer_single_example(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("LAGVS")
        self.assertListEqual(tokens, ["L", "A", "G", "V", "S"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [4, 5, 6, 7, 8])

    def test_tokenizer_encode_single(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        seq = "LAGVS"
        self.assertListEqual(tokenizer.encode(seq), [0, 4, 5, 6, 7, 8, 2])

    def test_tokenizer_call_no_pad(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        seq_batch = ["LAGVS", "WCB"]
        tokens_batch = tokenizer(seq_batch, padding=False)["input_ids"]

        self.assertListEqual(tokens_batch, [[0, 4, 5, 6, 7, 8, 2], [0, 22, 23, 25, 2]])

    def test_tokenizer_call_pad(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        seq_batch = ["LAGVS", "WCB"]
        tokens_batch = tokenizer(seq_batch, padding=True)["input_ids"]

        self.assertListEqual(tokens_batch, [[0, 4, 5, 6, 7, 8, 2], [0, 22, 23, 25, 2, 1, 1]])

    def test_tokenize_special_tokens(self):
        """Test `tokenize` with special tokens."""
        tokenizers = self.get_tokenizers(fast=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = "<unk>"
                SPECIAL_TOKEN_2 = "<mask>"

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), 1)
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
                self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    def test_add_tokens(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        vocab_size = len(tokenizer)
        self.assertEqual(tokenizer.add_tokens(""), 0)
        self.assertEqual(tokenizer.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer), vocab_size + 3)

        self.assertEqual(tokenizer.add_special_tokens({}), 0)
        self.assertEqual(tokenizer.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"}), 2)
        self.assertRaises(AssertionError, tokenizer.add_special_tokens, {"additional_special_tokens": "<testtoken1>"})
        self.assertEqual(tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        self.assertIn("<testtoken3>", tokenizer.special_tokens_map["additional_special_tokens"])
        self.assertIsInstance(tokenizer.special_tokens_map["additional_special_tokens"], list)
        self.assertGreaterEqual(len(tokenizer.special_tokens_map["additional_special_tokens"]), 2)

        self.assertEqual(len(tokenizer), vocab_size + 8)
