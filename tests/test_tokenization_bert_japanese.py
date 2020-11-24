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


import os
import pickle
import unittest

from transformers import AutoTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    VOCAB_FILES_NAMES,
    BertJapaneseTokenizer,
    CharacterTokenizer,
    MecabTokenizer,
    WordpieceTokenizer,
)
from transformers.testing_utils import custom_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


@custom_tokenizers
class BertJapaneseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BertJapaneseTokenizer
    space_between_special_tokens = True

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "こんにちは",
            "こん",
            "にちは",
            "ばんは",
            "##こん",
            "##にちは",
            "##ばんは",
            "世界",
            "##世界",
            "、",
            "##、",
            "。",
            "##。",
        ]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "こんにちは、世界。 \nこんばんは、世界。"
        output_text = "こんにちは 、 世界 。 こんばんは 、 世界 。"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        return text, ids

    def test_pretokenized_inputs(self):
        pass  # TODO add if relevant

    def test_maximum_encoding_length_pair_input(self):
        pass  # TODO add if relevant

    def test_maximum_encoding_length_single_input(self):
        pass  # TODO add if relevant

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("こんにちは、世界。\nこんばんは、世界。")
        self.assertListEqual(tokens, ["こんにちは", "、", "世界", "。", "こん", "##ばんは", "、", "世界", "。"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [3, 12, 10, 14, 4, 9, 12, 10, 14])

    def test_pickle_mecab_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, word_tokenizer_type="mecab")
        self.assertIsNotNone(tokenizer)

        text = "こんにちは、世界。\nこんばんは、世界。"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["こんにちは", "、", "世界", "。", "こん", "##ばんは", "、", "世界", "。"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [3, 12, 10, 14, 4, 9, 12, 10, 14])

        filename = os.path.join(self.tmpdirname, "tokenizer.bin")
        with open(filename, "wb") as handle:
            pickle.dump(tokenizer, handle)

        with open(filename, "rb") as handle:
            tokenizer_new = pickle.load(handle)

        tokens_loaded = tokenizer_new.tokenize(text)

        self.assertListEqual(tokens, tokens_loaded)

    def test_mecab_tokenizer_ipadic(self):
        tokenizer = MecabTokenizer(mecab_dic="ipadic")

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["アップルストア", "で", "iPhone", "8", "が", "発売", "さ", "れ", "た", "。"],
        )

    def test_mecab_tokenizer_unidic_lite(self):
        try:
            tokenizer = MecabTokenizer(mecab_dic="unidic_lite")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["アップル", "ストア", "で", "iPhone", "8", "が", "発売", "さ", "れ", "た", "。"],
        )

    def test_mecab_tokenizer_unidic(self):
        try:
            tokenizer = MecabTokenizer(mecab_dic="unidic")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["アップル", "ストア", "で", "iPhone", "8", "が", "発売", "さ", "れ", "た", "。"],
        )

    def test_mecab_tokenizer_lower(self):
        tokenizer = MecabTokenizer(do_lower_case=True, mecab_dic="ipadic")

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["アップルストア", "で", "iphone", "8", "が", "発売", "さ", "れ", "た", "。"],
        )

    def test_mecab_tokenizer_with_option(self):
        try:
            tokenizer = MecabTokenizer(
                do_lower_case=True, normalize_text=False, mecab_option="-d /usr/local/lib/mecab/dic/jumandic"
            )
        except RuntimeError:
            # if dict doesn't exist in the system, previous code raises this error.
            return

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["ｱｯﾌﾟﾙストア", "で", "iPhone", "８", "が", "発売", "さ", "れた", "\u3000", "。"],
        )

    def test_mecab_tokenizer_no_normalize(self):
        tokenizer = MecabTokenizer(normalize_text=False, mecab_dic="ipadic")

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["ｱｯﾌﾟﾙストア", "で", "iPhone", "８", "が", "発売", "さ", "れ", "た", "　", "。"],
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こんにちは", "こん", "にちは" "ばんは", "##こん", "##にちは", "##ばんは"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("こんにちは"), ["こんにちは"])

        self.assertListEqual(tokenizer.tokenize("こんばんは"), ["こん", "##ばんは"])

        self.assertListEqual(tokenizer.tokenize("こんばんは こんばんにちは こんにちは"), ["こん", "##ばんは", "[UNK]", "こんにちは"])

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("cl-tohoku/bert-base-japanese")

        text = tokenizer.encode("ありがとう。", add_special_tokens=False)
        text_2 = tokenizer.encode("どういたしまして。", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        # 2 is for "[CLS]", 3 is for "[SEP]"
        assert encoded_sentence == [2] + text + [3]
        assert encoded_pair == [2] + text + [3] + text_2 + [3]


@custom_tokenizers
class BertJapaneseCharacterTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BertJapaneseTokenizer

    def setUp(self):
        super().setUp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こ", "ん", "に", "ち", "は", "ば", "世", "界", "、", "。"]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        return BertJapaneseTokenizer.from_pretrained(self.tmpdirname, subword_tokenizer_type="character", **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "こんにちは、世界。 \nこんばんは、世界。"
        output_text = "こ ん に ち は 、 世 界 。 こ ん ば ん は 、 世 界 。"
        return input_text, output_text

    def test_pretokenized_inputs(self):
        pass  # TODO add if relevant

    def test_maximum_encoding_length_pair_input(self):
        pass  # TODO add if relevant

    def test_maximum_encoding_length_single_input(self):
        pass  # TODO add if relevant

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, subword_tokenizer_type="character")

        tokens = tokenizer.tokenize("こんにちは、世界。 \nこんばんは、世界。")
        self.assertListEqual(
            tokens, ["こ", "ん", "に", "ち", "は", "、", "世", "界", "。", "こ", "ん", "ば", "ん", "は", "、", "世", "界", "。"]
        )
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [3, 4, 5, 6, 7, 11, 9, 10, 12, 3, 4, 8, 4, 7, 11, 9, 10, 12]
        )

    def test_character_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こ", "ん", "に", "ち", "は", "ば", "世", "界" "、", "。"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = CharacterTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("こんにちは"), ["こ", "ん", "に", "ち", "は"])

        self.assertListEqual(tokenizer.tokenize("こんにちほ"), ["こ", "ん", "に", "ち", "[UNK]"])

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("cl-tohoku/bert-base-japanese-char")

        text = tokenizer.encode("ありがとう。", add_special_tokens=False)
        text_2 = tokenizer.encode("どういたしまして。", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        # 2 is for "[CLS]", 3 is for "[SEP]"
        assert encoded_sentence == [2] + text + [3]
        assert encoded_pair == [2] + text + [3] + text_2 + [3]


@custom_tokenizers
class AutoTokenizerCustomTest(unittest.TestCase):
    def test_tokenizer_bert_japanese(self):
        EXAMPLE_BERT_JAPANESE_ID = "cl-tohoku/bert-base-japanese"
        tokenizer = AutoTokenizer.from_pretrained(EXAMPLE_BERT_JAPANESE_ID)
        self.assertIsInstance(tokenizer, BertJapaneseTokenizer)
