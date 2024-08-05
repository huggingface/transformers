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
import pickle
import unittest

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    VOCAB_FILES_NAMES,
    BertJapaneseTokenizer,
    CharacterTokenizer,
    JumanppTokenizer,
    MecabTokenizer,
    SudachiTokenizer,
    WordpieceTokenizer,
)
from transformers.testing_utils import custom_tokenizers, require_jumanpp, require_sudachi_projection

from ...test_tokenization_common import TokenizerTesterMixin


@custom_tokenizers
class BertJapaneseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "cl-tohoku/bert-base-japanese"
    tokenizer_class = BertJapaneseTokenizer
    test_rust_tokenizer = False
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
            "アップルストア",
            "外国",
            "##人",
            "参政",
            "##権",
            "此れ",
            "は",
            "猫",
            "です",
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

    def test_mecab_full_tokenizer_with_mecab_kwargs(self):
        tokenizer = self.tokenizer_class(
            self.vocab_file, word_tokenizer_type="mecab", mecab_kwargs={"mecab_dic": "ipadic"}
        )

        text = "ｱｯﾌﾟﾙストア"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["アップルストア"])

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
            import unidic

            self.assertTrue(
                os.path.isdir(unidic.DICDIR),
                "The content of unidic was not downloaded. Run `python -m unidic download` before running this test case. Note that this requires 2.1GB on disk.",
            )
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

    @require_sudachi_projection
    def test_pickle_sudachi_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, word_tokenizer_type="sudachi")
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

    @require_sudachi_projection
    def test_sudachi_tokenizer_core(self):
        tokenizer = SudachiTokenizer(sudachi_dict_type="core")

        # fmt: off
        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            [" ",  "\t",  "アップル",  "ストア",  "で",  "iPhone",  "8",  " ",  "が",  " ",  " ",  "\n ",  "発売",  "さ",  "れ",  "た",  " ",  "。",  " ",  " "],
        )
        # fmt: on

    @require_sudachi_projection
    def test_sudachi_tokenizer_split_mode_A(self):
        tokenizer = SudachiTokenizer(sudachi_dict_type="core", sudachi_split_mode="A")

        self.assertListEqual(tokenizer.tokenize("外国人参政権"), ["外国", "人", "参政", "権"])

    @require_sudachi_projection
    def test_sudachi_tokenizer_split_mode_B(self):
        tokenizer = SudachiTokenizer(sudachi_dict_type="core", sudachi_split_mode="B")

        self.assertListEqual(tokenizer.tokenize("外国人参政権"), ["外国人", "参政権"])

    @require_sudachi_projection
    def test_sudachi_tokenizer_split_mode_C(self):
        tokenizer = SudachiTokenizer(sudachi_dict_type="core", sudachi_split_mode="C")

        self.assertListEqual(tokenizer.tokenize("外国人参政権"), ["外国人参政権"])

    @require_sudachi_projection
    def test_sudachi_full_tokenizer_with_sudachi_kwargs_split_mode_B(self):
        tokenizer = self.tokenizer_class(
            self.vocab_file, word_tokenizer_type="sudachi", sudachi_kwargs={"sudachi_split_mode": "B"}
        )

        self.assertListEqual(tokenizer.tokenize("外国人参政権"), ["外国", "##人", "参政", "##権"])

    @require_sudachi_projection
    def test_sudachi_tokenizer_projection(self):
        tokenizer = SudachiTokenizer(
            sudachi_dict_type="core", sudachi_split_mode="A", sudachi_projection="normalized_nouns"
        )

        self.assertListEqual(tokenizer.tokenize("これはねこです。"), ["此れ", "は", "猫", "です", "。"])

    @require_sudachi_projection
    def test_sudachi_full_tokenizer_with_sudachi_kwargs_sudachi_projection(self):
        tokenizer = self.tokenizer_class(
            self.vocab_file, word_tokenizer_type="sudachi", sudachi_kwargs={"sudachi_projection": "normalized_nouns"}
        )

        self.assertListEqual(tokenizer.tokenize("これはねこです。"), ["此れ", "は", "猫", "です", "。"])

    @require_sudachi_projection
    def test_sudachi_tokenizer_lower(self):
        tokenizer = SudachiTokenizer(do_lower_case=True, sudachi_dict_type="core")

        self.assertListEqual(tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),[" ", "\t", "アップル", "ストア", "で", "iphone", "8", " ", "が", " ", " ", "\n ", "発売", "さ", "れ", "た", " ", "。", " ", " "])  # fmt: skip

    @require_sudachi_projection
    def test_sudachi_tokenizer_no_normalize(self):
        tokenizer = SudachiTokenizer(normalize_text=False, sudachi_dict_type="core")

        self.assertListEqual(tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),[" ", "\t", "ｱｯﾌﾟﾙ", "ストア", "で", "iPhone", "８", " ", "が", " ", " ", "\n ", "発売", "さ", "れ", "た", "\u3000", "。", " ", " "])  # fmt: skip

    @require_sudachi_projection
    def test_sudachi_tokenizer_trim_whitespace(self):
        tokenizer = SudachiTokenizer(trim_whitespace=True, sudachi_dict_type="core")

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["アップル", "ストア", "で", "iPhone", "8", "が", "発売", "さ", "れ", "た", "。"],
        )

    @require_jumanpp
    def test_pickle_jumanpp_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, word_tokenizer_type="jumanpp")
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

    @require_jumanpp
    def test_jumanpp_tokenizer(self):
        tokenizer = JumanppTokenizer()

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),["アップル", "ストア", "で", "iPhone", "8", "\u3000", "が", "\u3000", "\u3000", "\u3000", "発売", "さ", "れた", "\u3000", "。"])  # fmt: skip

    @require_jumanpp
    def test_jumanpp_tokenizer_lower(self):
        tokenizer = JumanppTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),["アップル", "ストア", "で", "iphone", "8", "\u3000", "が", "\u3000", "\u3000", "\u3000", "発売", "さ", "れた", "\u3000", "。"],)  # fmt: skip

    @require_jumanpp
    def test_jumanpp_tokenizer_no_normalize(self):
        tokenizer = JumanppTokenizer(normalize_text=False)

        self.assertListEqual(tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),["ｱ", "ｯ", "ﾌ", "ﾟ", "ﾙ", "ストア", "で", "iPhone", "８", "\u3000", "が", "\u3000", "\u3000", "\u3000", "発売", "さ", "れた", "\u3000", "。"],)  # fmt: skip

    @require_jumanpp
    def test_jumanpp_tokenizer_trim_whitespace(self):
        tokenizer = JumanppTokenizer(trim_whitespace=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            ["アップル", "ストア", "で", "iPhone", "8", "が", "発売", "さ", "れた", "。"],
        )

    @require_jumanpp
    def test_jumanpp_full_tokenizer_with_jumanpp_kwargs_trim_whitespace(self):
        tokenizer = self.tokenizer_class(
            self.vocab_file, word_tokenizer_type="jumanpp", jumanpp_kwargs={"trim_whitespace": True}
        )

        text = "こんにちは、世界。\nこんばんは、世界。"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["こんにちは", "、", "世界", "。", "こん", "##ばんは", "、", "世界", "。"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [3, 12, 10, 14, 4, 9, 12, 10, 14])

    @require_jumanpp
    def test_jumanpp_tokenizer_ext(self):
        tokenizer = JumanppTokenizer()

        self.assertListEqual(
            tokenizer.tokenize("ありがとうございますm(_ _)ｍ見つけるのが大変です。"),
            ["ありがとう", "ございます", "m(_ _)m", "見つける", "の", "が", "大変です", "。"],
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こんにちは", "こん", "にちは", "ばんは", "##こん", "##にちは", "##ばんは"]  # fmt: skip

        vocab = {}
        for i, token in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("こんにちは"), ["こんにちは"])

        self.assertListEqual(tokenizer.tokenize("こんばんは"), ["こん", "##ばんは"])

        self.assertListEqual(tokenizer.tokenize("こんばんは こんばんにちは こんにちは"), ["こん", "##ばんは", "[UNK]", "こんにちは"])  # fmt: skip

    def test_sentencepiece_tokenizer(self):
        tokenizer = BertJapaneseTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese-with-auto-jumanpp")
        subword_tokenizer = tokenizer.subword_tokenizer

        tokens = subword_tokenizer.tokenize("国境 の 長い トンネル を 抜ける と 雪国 であった 。")
        self.assertListEqual(tokens, ["▁国境", "▁の", "▁長い", "▁トンネル", "▁を", "▁抜ける", "▁と", "▁雪", "国", "▁であった", "▁。"])  # fmt: skip

        tokens = subword_tokenizer.tokenize("こんばんは こんばん にち は こんにちは")
        self.assertListEqual(tokens, ["▁こん", "ばん", "は", "▁こん", "ばん", "▁に", "ち", "▁は", "▁こんにちは"])

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
    from_pretrained_id = "cl-tohoku/bert-base-japanese"
    tokenizer_class = BertJapaneseTokenizer
    test_rust_tokenizer = False

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
        self.assertListEqual(tokens, ["こ", "ん", "に", "ち", "は", "、", "世", "界", "。", "こ", "ん", "ば", "ん", "は", "、", "世", "界", "。"])  # fmt: skip
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [3, 4, 5, 6, 7, 11, 9, 10, 12, 3, 4, 8, 4, 7, 11, 9, 10, 12]
        )

    def test_character_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こ", "ん", "に", "ち", "は", "ば", "世", "界", "、", "。"]

        vocab = {}
        for i, token in enumerate(vocab_tokens):
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


class BertTokenizerMismatchTest(unittest.TestCase):
    def test_tokenizer_mismatch_warning(self):
        EXAMPLE_BERT_JAPANESE_ID = "cl-tohoku/bert-base-japanese"
        with self.assertLogs("transformers", level="WARNING") as cm:
            BertTokenizer.from_pretrained(EXAMPLE_BERT_JAPANESE_ID)
            self.assertTrue(
                cm.records[0].message.startswith(
                    "The tokenizer class you load from this checkpoint is not the same type as the class this function"
                    " is called from."
                )
            )
        EXAMPLE_BERT_ID = "google-bert/bert-base-cased"
        with self.assertLogs("transformers", level="WARNING") as cm:
            BertJapaneseTokenizer.from_pretrained(EXAMPLE_BERT_ID)
            self.assertTrue(
                cm.records[0].message.startswith(
                    "The tokenizer class you load from this checkpoint is not the same type as the class this function"
                    " is called from."
                )
            )
