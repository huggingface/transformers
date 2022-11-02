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


import os
import unittest

from transformers.models.transfo_xl.tokenization_transfo_xl import VOCAB_FILES_NAMES, TransfoXLTokenizer

from ...test_tokenization_common import TokenizerTesterMixin


class TransfoXLTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = TransfoXLTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "<unk>",
            "[CLS]",
            "[SEP]",
            "want",
            "unwanted",
            "wa",
            "un",
            "running",
            ",",
            "low",
            "l",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        kwargs["lower_case"] = True
        return TransfoXLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "<unk> UNwanted , running"
        output_text = "<unk> unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = TransfoXLTokenizer(vocab_file=self.vocab_file, lower_case=True)

        tokens = tokenizer.tokenize("<unk> UNwanted , running")
        self.assertListEqual(tokens, ["<unk>", "unwanted", ",", "running"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [0, 4, 8, 7])

    def test_full_tokenizer_lower(self):
        tokenizer = TransfoXLTokenizer(lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo ! how  \n Are yoU ?  "), ["hello", "!", "how", "are", "you", "?"]
        )

    def test_full_tokenizer_no_lower(self):
        tokenizer = TransfoXLTokenizer(lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo ! how  \n Are yoU ?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_full_tokenizer_moses_numbers(self):
        tokenizer = TransfoXLTokenizer(lower_case=False)
        text_in = "Hello (bracket) and side-scrolled [and] Henry's $5,000 with 3.34 m. What's up!?"
        tokens_out = [
            "Hello",
            "(",
            "bracket",
            ")",
            "and",
            "side",
            "@-@",
            "scrolled",
            "[",
            "and",
            "]",
            "Henry",
            "'s",
            "$",
            "5",
            "@,@",
            "000",
            "with",
            "3",
            "@.@",
            "34",
            "m",
            ".",
            "What",
            "'s",
            "up",
            "!",
            "?",
        ]

        self.assertListEqual(tokenizer.tokenize(text_in), tokens_out)

        self.assertEqual(tokenizer.convert_tokens_to_string(tokens_out), text_in)

    def test_move_added_token(self):
        tokenizer = self.get_tokenizer()
        original_len = len(tokenizer)

        tokenizer.add_tokens(["new1", "new2"])
        tokenizer.move_added_token("new1", 1)

        # Check that moved token is not copied (duplicate)
        self.assertEqual(len(tokenizer), original_len + 2)
        # Check that token is moved to specified id
        self.assertEqual(tokenizer.encode("new1"), [1])
        self.assertEqual(tokenizer.decode([1]), "new1")
