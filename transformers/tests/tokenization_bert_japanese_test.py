# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from io import open

from transformers.tokenization_bert import WordpieceTokenizer
from transformers.tokenization_bert_japanese import (BertJapaneseTokenizer,
                                                     MecabTokenizer, CharacterTokenizer,
                                                     VOCAB_FILES_NAMES)

from .tokenization_tests_commons import CommonTestCases
from .utils import slow, custom_tokenizers


@custom_tokenizers
class BertJapaneseTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = BertJapaneseTokenizer

    def setUp(self):
        super(BertJapaneseTokenizationTest, self).setUp()

        vocab_tokens = [u"[UNK]", u"[CLS]", u"[SEP]",
            u"こんにちは", u"こん", u"にちは", u"ばんは", u"##こん", u"##にちは", u"##ばんは",
            u"世界", u"##世界", u"、", u"##、", u"。", u"##。"]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        return BertJapaneseTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"こんにちは、世界。 \nこんばんは、世界。"
        output_text = u"こんにちは 、 世界 。 こんばんは 、 世界 。"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize(u"こんにちは、世界。\nこんばんは、世界。")
        self.assertListEqual(tokens,
                             [u"こんにちは", u"、", u"世界", u"。",
                              u"こん", u"##ばんは", u"、", u"世界", "。"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [3, 12, 10, 14, 4, 9, 12, 10, 14])

    def test_mecab_tokenizer(self):
        tokenizer = MecabTokenizer()

        self.assertListEqual(
            tokenizer.tokenize(u" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
                               [u"アップルストア", u"で", u"iPhone", u"8", u"が",
                                u"発売", u"さ", u"れ", u"た", u"。"])

    def test_mecab_tokenizer_lower(self):
        tokenizer = MecabTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(u" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
                               [u"アップルストア", u"で", u"iphone", u"8", u"が",
                                u"発売", u"さ", u"れ", u"た", u"。"])

    def test_mecab_tokenizer_no_normalize(self):
        tokenizer = MecabTokenizer(normalize_text=False)

        self.assertListEqual(
            tokenizer.tokenize(u" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
                               [u"ｱｯﾌﾟﾙストア", u"で", u"iPhone", u"８", u"が",
                                u"発売", u"さ", u"れ", u"た", u"　", u"。"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [u"[UNK]", u"[CLS]", u"[SEP]",
            u"こんにちは", u"こん", u"にちは" u"ばんは", u"##こん", u"##にちは", u"##ばんは"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token=u"[UNK]")

        self.assertListEqual(tokenizer.tokenize(u""), [])

        self.assertListEqual(tokenizer.tokenize(u"こんにちは"),
                             [u"こんにちは"])

        self.assertListEqual(tokenizer.tokenize(u"こんばんは"),
                             [u"こん", u"##ばんは"])

        self.assertListEqual(tokenizer.tokenize(u"こんばんは こんばんにちは こんにちは"),
                             [u"こん", u"##ばんは", u"[UNK]", u"こんにちは"])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("bert-base-japanese")

        text = tokenizer.encode(u"ありがとう。", add_special_tokens=False)
        text_2 = tokenizer.encode(u"どういたしまして。", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        # 2 is for "[CLS]", 3 is for "[SEP]"
        assert encoded_sentence == [2] + text + [3]
        assert encoded_pair == [2] + text + [3] + text_2 + [3]


class BertJapaneseCharacterTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = BertJapaneseTokenizer

    def setUp(self):
        super(BertJapaneseCharacterTokenizationTest, self).setUp()

        vocab_tokens = [u"[UNK]", u"[CLS]", u"[SEP]",
            u"こ", u"ん", u"に", u"ち", u"は", u"ば", u"世", u"界", u"、", u"。"]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        return BertJapaneseTokenizer.from_pretrained(self.tmpdirname,
                                                     subword_tokenizer_type="character",
                                                     **kwargs)

    def get_input_output_texts(self):
        input_text = u"こんにちは、世界。 \nこんばんは、世界。"
        output_text = u"こ ん に ち は 、 世 界 。 こ ん ば ん は 、 世 界 。"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file,
                                         subword_tokenizer_type="character")

        tokens = tokenizer.tokenize(u"こんにちは、世界。 \nこんばんは、世界。")
        self.assertListEqual(tokens,
            [u"こ", u"ん", u"に", u"ち", u"は", u"、", u"世", u"界", u"。",
             u"こ", u"ん", u"ば", u"ん", u"は", u"、", u"世", u"界", u"。"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [3, 4, 5, 6, 7, 11, 9, 10, 12,
                              3, 4, 8, 4, 7, 11, 9, 10, 12])

    def test_character_tokenizer(self):
        vocab_tokens = [u"[UNK]", u"[CLS]", u"[SEP]",
            u"こ", u"ん", u"に", u"ち", u"は", u"ば", u"世", u"界"u"、", u"。"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = CharacterTokenizer(vocab=vocab, unk_token=u"[UNK]")

        self.assertListEqual(tokenizer.tokenize(u""), [])

        self.assertListEqual(tokenizer.tokenize(u"こんにちは"),
                             [u"こ", u"ん", u"に", u"ち", u"は"])

        self.assertListEqual(tokenizer.tokenize(u"こんにちほ"),
                             [u"こ", u"ん", u"に", u"ち", u"[UNK]"])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("bert-base-japanese-char")

        text = tokenizer.encode(u"ありがとう。", add_special_tokens=False)
        text_2 = tokenizer.encode(u"どういたしまして。", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        # 2 is for "[CLS]", 3 is for "[SEP]"
        assert encoded_sentence == [2] + text + [3]
        assert encoded_pair == [2] + text + [3] + text_2 + [3]



