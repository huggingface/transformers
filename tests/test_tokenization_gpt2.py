# coding=utf-8
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


import json
import os
import unittest

from transformers import GPT2Tokenizer, GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class GPT2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPT2Tokenizer
    rust_tokenizer_class = GPT2TokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {"add_prefix_space": True}
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
            "<|endoftext|>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return GPT2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return GPT2TokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = GPT2Tokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer(add_prefix_space=True)

        sequence = "lower newer"

        # Testing tokenization
        tokens = tokenizer.tokenize(sequence, add_prefix_space=True)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        # Testing conversion to ids without special tokens
        ids = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        # Testing conversion to ids with special tokens
        rust_tokenizer = self.get_rust_tokenizer(add_prefix_space=True)
        ids = tokenizer.encode(sequence, add_prefix_space=True)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # Testing the unknown token
        input_tokens = tokens + [rust_tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(rust_tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_pretokenized_inputs(self, *args, **kwargs):
        # It's very difficult to mix/test pretokenization with byte-level
        # And get both GPT2 and Roberta to work at the same time (mostly an issue of adding a space before the string)
        pass

    def test_padding(self, max_length=15):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest("{} ({})".format(tokenizer.__class__.__name__, pretrained_name)):
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
