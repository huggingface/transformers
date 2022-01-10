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

from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class CLIPTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CLIPTokenizer
    rust_tokenizer_class = CLIPTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {}
    test_seq2seq = False

    def setUp(self):
        super().setUp()
        # temporary addition: to test the new slow to fast converter
        self.tokenizers_list = [(CLIPTokenizerFast, "SaulLu/clip-vit-base-patch32", {})]

        # fmt: off
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "<|startoftext|>", "<|endoftext|>"]
        # fmt: on
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

    def test_pretokenized_inputs(self, *args, **kwargs):
        # It's very difficult to mix/test pretokenization with byte-level
        # And get both CLIP and Roberta to work at the same time (mostly an issue of adding a space before the string)
        pass

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                # padding is very hacky in CLIPTokenizer, pad_token_id is always 0
                # so skip this check
                # self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    @require_ftfy
    def test_check_encoding_slow_fast(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_s = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                text = "A\n'll 11p223RF☆ho!!to?'d'd''d of a cat"
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

                # The tokenization is not identical for the character "\u0085" (next line). The slow version transforms
                # it into the Horizontal Ellipsis character "…" ("\u2026") while the fast version transforms it into a
                # space (and thus into an empty list).

                for unicode_seq in line_break_unicodes:
                    text_tokenized_s = tokenizer_s.tokenize(unicode_seq)
                    text_tokenized_r = tokenizer_r.tokenize(unicode_seq)

                    self.assertListEqual(text_tokenized_s, text_tokenized_r)

    @require_ftfy
    def test_tokenization_python_rust_equals(self):
        super().test_tokenization_python_rust_equals()

    # overwrite common test
    def test_added_tokens_do_lower_case(self):
        # CLIP always lower cases letters
        pass
