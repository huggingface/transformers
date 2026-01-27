# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers.models.roc_bert.tokenization_roc_bert import (
    VOCAB_FILES_NAMES,
    RoCBertBasicTokenizer,
    RoCBertTokenizer,
    RoCBertWordpieceTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin, filter_non_english


@require_tokenizers
class BertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "weiweishi/roc-bert-base-zh"
    tokenizer_class = RoCBertTokenizer
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "你", "好", "是", "谁", "a", "b", "c", "d"]
        word_shape = {}
        word_pronunciation = {}
        for i, value in enumerate(vocab_tokens):
            word_shape[value] = i
            word_pronunciation[value] = i
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        cls.word_shape_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["word_shape_file"])
        cls.word_pronunciation_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["word_pronunciation_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        with open(cls.word_shape_file, "w", encoding="utf-8") as word_shape_writer:
            json.dump(word_shape, word_shape_writer, ensure_ascii=False)
        with open(cls.word_pronunciation_file, "w", encoding="utf-8") as word_pronunciation_writer:
            json.dump(word_pronunciation, word_pronunciation_writer, ensure_ascii=False)

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, self.word_shape_file, self.word_pronunciation_file)

        tokens = tokenizer.tokenize("你好[SEP]你是谁")
        self.assertListEqual(tokens, ["你", "好", "[SEP]", "你", "是", "谁"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [5, 6, 2, 5, 7, 8])
        self.assertListEqual(tokenizer.convert_tokens_to_shape_ids(tokens), [5, 6, 2, 5, 7, 8])
        self.assertListEqual(tokenizer.convert_tokens_to_pronunciation_ids(tokens), [5, 6, 2, 5, 7, 8])

    def test_chinese(self):
        tokenizer = RoCBertBasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535a\u63a8zz"), ["ah", "\u535a", "\u63a8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["hello", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hällo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["h\u00e9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=True, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HäLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HaLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = RoCBertBasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"), ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"]
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]

        vocab = {}
        for i, token in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = RoCBertWordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("unwanted running"), ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_is_whitespace(self):
        self.assertTrue(_is_whitespace(" "))
        self.assertTrue(_is_whitespace("\t"))
        self.assertTrue(_is_whitespace("\r"))
        self.assertTrue(_is_whitespace("\n"))
        self.assertTrue(_is_whitespace("\u00a0"))

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

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], [], ["[UNK]"]])

        if self.test_rust_tokenizer:
            rust_tokenizer = self.get_tokenizer()
            self.assertListEqual(
                [rust_tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], [], ["[UNK]"]]
            )

    def test_change_tokenize_chinese_chars(self):
        list_of_common_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_common_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                kwargs["tokenize_chinese_chars"] = True
                tokenizer_p = self.get_tokenizer(pretrained_name, **kwargs)
                tokenizer_r = self.get_tokenizer(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer_p.encode(text_with_chinese_char, add_special_tokens=False)
                ids_without_spe_char_r = tokenizer_r.encode(text_with_chinese_char, add_special_tokens=False)

                tokens_without_spe_char_r = tokenizer_r.convert_ids_to_tokens(ids_without_spe_char_r)
                tokens_without_spe_char_p = tokenizer_p.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p, list_of_common_chinese_char)
                self.assertListEqual(tokens_without_spe_char_r, list_of_common_chinese_char)

                kwargs["tokenize_chinese_chars"] = False
                tokenizer_r = self.get_tokenizer(pretrained_name, **kwargs)
                tokenizer_p = self.get_tokenizer(pretrained_name, **kwargs)

                ids_without_spe_char_r = tokenizer_r.encode(text_with_chinese_char, add_special_tokens=False)
                ids_without_spe_char_p = tokenizer_p.encode(text_with_chinese_char, add_special_tokens=False)

                tokens_without_spe_char_r = tokenizer_r.convert_ids_to_tokens(ids_without_spe_char_r)
                tokens_without_spe_char_p = tokenizer_p.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that only the first Chinese character is not preceded by "##".
                expected_tokens = [
                    f"##{token}" if idx != 0 else token for idx, token in enumerate(list_of_common_chinese_char)
                ]
                self.assertListEqual(tokens_without_spe_char_p, expected_tokens)
                self.assertListEqual(tokens_without_spe_char_r, expected_tokens)

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class(self.vocab_file, self.word_shape_file, self.word_pronunciation_file)

        text = tokenizer.encode("你好", add_special_tokens=False)
        text_2 = tokenizer.encode("你是谁", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [1] + text + [2]
        assert encoded_pair == [1] + text + [2] + text_2 + [2]

    def test_prepare_for_model(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                string_sequence = "你好，你是谁"
                tokens = tokenizer.tokenize(string_sequence)
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
                tokens_shape_ids = tokenizer.convert_tokens_to_shape_ids(tokens)
                tokens_proun_ids = tokenizer.convert_tokens_to_pronunciation_ids(tokens)
                prepared_input_dict = tokenizer.prepare_for_model(
                    tokens_ids, tokens_shape_ids, tokens_proun_ids, add_special_tokens=True
                )

                input_dict = tokenizer(string_sequence, add_special_tokens=True)

                self.assertEqual(input_dict, prepared_input_dict)
