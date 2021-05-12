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

from transformers.models.character_bert.tokenization_character_bert import (
    VOCAB_FILES_NAMES,
    BasicTokenizer,
    CharacterBertTokenizer,
    CharacterMapper,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)
from transformers.testing_utils import require_tokenizers, slow

from .test_tokenization_common import TokenizerTesterMixin, filter_non_english


@require_tokenizers
class BertTokenizationTest(unittest.TestCase):

    tokenizer_class = CharacterBertTokenizer
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english

    def setUp(self):
        super().setUp()

    def test_full_tokenizer(self):
        sequence = "Hi!"

        # Test default (lowercase)
        tokenizer = self.tokenizer_class()
        expected_tokens = ['hi', '!']  # Same as BasicTokenizer
        expected_ids = [
            # Each word is represented as a sequence of characters
            # beginning with special beginning of word character,
            # ending with a special end of word character
            # then followed with as many padding characters as needed
            # to fill a `max_word_length` (default=50) sequence
            [
                CharacterMapper.beginning_of_word_character + 1,
                105, 106,
                CharacterMapper.end_of_word_character + 1,
            ] + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 4),
            [
                CharacterMapper.beginning_of_word_character + 1,
                34,
                CharacterMapper.end_of_word_character + 1,
            ] + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 3)
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, expected_tokens)
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), expected_ids)
        ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids,expected_ids)

        # Test UpperCase
        tokenizer = self.tokenizer_class.from_pretrained('helboukkouri/character-bert', do_lower_case=False)
        expected_tokens = ['Hi', '!']  # Same as BasicTokenizer
        expected_ids = [
            # Each word is represented as a sequence of characters
            # beginning with special beginning of word character,
            # ending with a special end of word character
            # then followed with as many padding characters as needed
            # to fill a `max_word_length` (default=50) sequence
            [
                CharacterMapper.beginning_of_word_character + 1,
                73, 106,
                CharacterMapper.end_of_word_character + 1,
            ] + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 4),
            [
                CharacterMapper.beginning_of_word_character + 1,
                34,
                CharacterMapper.end_of_word_character + 1,
            ] + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 3)
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, expected_tokens)
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), expected_ids)
        ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids,expected_ids)

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("helboukkouri/character-bert")
        CLS = [
            CharacterMapper.beginning_of_word_character + 1,
            257,
            CharacterMapper.end_of_word_character + 1,
        ] + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 3)
        SEP = [
            CharacterMapper.beginning_of_word_character + 1,
            258,
            CharacterMapper.end_of_word_character + 1,
        ] + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 3)

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [CLS] + text + [SEP]
        assert encoded_pair == [CLS] + text + [SEP] + text_2 + [SEP]

    def test_basic_tokenizer_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"), ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["hello", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hällo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["h\u00E9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HäLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HaLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"), ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"]
        )

    def test_is_whitespace(self):
        self.assertTrue(_is_whitespace(" "))
        self.assertTrue(_is_whitespace("\t"))
        self.assertTrue(_is_whitespace("\r"))
        self.assertTrue(_is_whitespace("\n"))
        self.assertTrue(_is_whitespace("\u00A0"))

        self.assertFalse(_is_whitespace("A"))
        self.assertFalse(_is_whitespace("-"))

    def test_is_control(self):
        self.assertTrue(_is_control("\u0005"))

        self.assertFalse(_is_control("A"))
        self.assertFalse(_is_control(" "))
        self.assertFalse(_is_control("\t"))
        self.assertFalse(_is_control("\r"))

    def test_is_punctuation(self):
        self.assertTrue(_is_punctuation("-"))
        self.assertTrue(_is_punctuation("$"))
        self.assertTrue(_is_punctuation("`"))
        self.assertTrue(_is_punctuation("."))

        self.assertFalse(_is_punctuation("A"))
        self.assertFalse(_is_punctuation(" "))
