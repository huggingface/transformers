# coding=utf-8
# Copyright 2024 The Qwen Team and the HuggingFace Team. All rights reserved.
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

from transformers import Qwen2Tokenizer, Qwen2TokenizerFast
from transformers.models.qwen2.tokenization_qwen2 import VOCAB_FILES_NAMES, bytes_to_unicode
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class Qwen2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Qwen2Tokenizer
    rust_tokenizer_class = Qwen2TokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        vocab = list(bytes_to_unicode().values())
        vocab.extend(
            [
                "\u0120l",
                "\u0120n",
                "\u0120lo",
                "\u0120low",
                "er",
                "\u0120lowest",
                "\u0120newer",
                "\u0120wider",
                "01",
                ";}",
                ";}\u010a",
                "\u00cf\u0135",
            ]
        )

        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        merges = [
            "#version: 0.2",
            "\u0120 l",
            "\u0120l o",
            "\u0120lo w",
            "e r",
            "0 1",
            "; }",
            ";} \u010a",
            "\u00cf \u0135",
        ]

        # unk_token is needed, because this stub tokenizer is not complete at the byte level
        self.special_tokens_map = {"eos_token": "<|endoftext|>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Qwen2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Qwen2TokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        # this case should cover
        # - NFC normalization (code point U+03D3 has different normalization forms under NFC, NFD, NFKC, and NFKD)
        # - the pretokenization rules (spliting digits and merging symbols with \n\r)
        # - no space added by default to the left of the special tokens in decode
        input_text = "lower lower newer 010;}\n<|endoftext|>\u03d2\u0301"
        output_text = "lower lower newer 010;}\n<|endoftext|>\u03d3"
        return input_text, output_text

    def test_python_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        sequence, _ = self.get_input_output_texts(tokenizer)
        bpe_tokens = [
            "l",
            "o",
            "w",
            "er",
            "\u0120low",
            "er",
            "\u0120",
            "n",
            "e",
            "w",
            "er",
            "\u0120",
            "0",
            "1",
            "0",
            ";}\u010a",
            "<|endoftext|>",
            "\u00cf\u0135",
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens
        input_bpe_tokens = [75, 78, 86, 260, 259, 260, 220, 77, 68, 86, 260, 220, 15, 16, 15, 266, 268, 267]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
