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


import json
import os
import unittest

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers.models.clip.tokenization_clip import VOCAB_FILES_NAMES
from transformers.testing_utils import require_ftfy, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class CLIPTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "openai/clip-vit-base-patch32"
    tokenizer_class = CLIPTokenizer
    rust_tokenizer_class = CLIPTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {}
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "<|startoftext|>", "<|endoftext|>"]  # fmt: skip
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w</w>", "e r</w>"]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return CLIPTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return CLIPTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = CLIPTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["lo", "w", "er</w>", "n", "e", "w", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [10, 2, 16, 9, 3, 2, 16, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @require_ftfy
    def test_check_encoding_slow_fast(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_s = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                text = "A\n'll 11p223RF☆ho!!to?'d'd''d of a cat to-$''d."
                text_tokenized_s = tokenizer_s.tokenize(text)
                text_tokenized_r = tokenizer_r.tokenize(text)

                self.assertListEqual(text_tokenized_s, text_tokenized_r)

                # Test that the tokenization is identical on an example containing a character (Latin Small Letter A
                # with Tilde) encoded in 2 different ways
                text = "xa\u0303y" + " " + "x\xe3y"
                text_tokenized_s = tokenizer_s.tokenize(text)
                text_tokenized_r = tokenizer_r.tokenize(text)

                self.assertListEqual(text_tokenized_s, text_tokenized_r)

                # Test that the tokenization is identical on unicode of space type
                spaces_unicodes = [
                    "\u0009",  # (horizontal tab, '\t')
                    "\u000B",  # (vertical tab)
                    "\u000C",  # (form feed)
                    "\u0020",  # (space, ' ')
                    "\u200E",  # (left-to-right mark):w
                    "\u200F",  # (right-to-left mark)
                ]
                for unicode_seq in spaces_unicodes:
                    text_tokenized_s = tokenizer_s.tokenize(unicode_seq)
                    text_tokenized_r = tokenizer_r.tokenize(unicode_seq)

                    self.assertListEqual(text_tokenized_s, text_tokenized_r)

                # Test that the tokenization is identical on unicode of line break type
                line_break_unicodes = [
                    "\u000A",  # (line feed, '\n')
                    "\r\n",  # (carriage return and line feed, '\r\n')
                    "\u000D",  # (carriage return, '\r')
                    "\r",  # (carriage return, '\r')
                    "\u000D",  # (carriage return, '\r')
                    "\u2028",  # (line separator)
                    "\u2029",  # (paragraph separator)
                    # "\u0085", # (next line)
                ]

                # The tokenization is not identical for the character "\u0085" (next line). The slow version using ftfy transforms
                # it into the Horizontal Ellipsis character "…" ("\u2026") while the fast version transforms it into a
                # space (and thus into an empty list).

                for unicode_seq in line_break_unicodes:
                    text_tokenized_s = tokenizer_s.tokenize(unicode_seq)
                    text_tokenized_r = tokenizer_r.tokenize(unicode_seq)

                    self.assertListEqual(text_tokenized_s, text_tokenized_r)

    def test_offsets_mapping_with_different_add_prefix_space_argument(self):
        # Test which aims to verify that the offsets are well adapted to the argument `add_prefix_space`
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                text_of_1_token = "hello"  # `hello` is a token in the vocabulary of `pretrained_name`
                text = f"{text_of_1_token} {text_of_1_token}"

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name,
                    use_fast=True,
                )
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(
                    encoding.offset_mapping[1],
                    (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)),
                )

                text = f" {text}"

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name,
                    use_fast=True,
                )
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (1, 1 + len(text_of_1_token)))
                self.assertEqual(
                    encoding.offset_mapping[1],
                    (1 + len(text_of_1_token) + 1, 1 + len(text_of_1_token) + 1 + len(text_of_1_token)),
                )

    def test_log_warning(self):
        # Test related to the breaking change introduced in transformers v4.17.0
        # We need to check that an error in raised when the user try to load a previous version of the tokenizer.
        with self.assertRaises(ValueError) as context:
            self.rust_tokenizer_class.from_pretrained("robot-test/old-clip-tokenizer")

        self.assertTrue(
            context.exception.args[0].startswith(
                "The `backend_tokenizer` provided does not match the expected format."
            )
        )

    @require_ftfy
    def test_tokenization_python_rust_equals(self):
        super().test_tokenization_python_rust_equals()

    # overwrite common test
    def test_added_tokens_do_lower_case(self):
        # CLIP always lower cases letters
        pass
