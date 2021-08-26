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

from transformers import RobertaJapaneseTokenizer
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece_ja.model")


@require_sentencepiece
@require_tokenizers
class JapaneseRobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = RobertaJapaneseTokenizer
    test_sentencepiece = True
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = RobertaJapaneseTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "<mask>")
        self.assertEqual(len(vocab_keys), 3_002)

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

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 3_002)

    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        """Test ``_tokenize`` and ``convert_tokens_to_string``."""
        if not self.test_sentencepiece:
            return

        tokenizer = self.get_tokenizer()
        text = "こんにちは 、 世界 。 こんばんは 、 世界 。"

        if self.test_sentencepiece_ignore_case:
            text = text.lower()

        tokens = tokenizer.tokenize(text)

        self.assertTrue(len(tokens) > 0)

        # check if converting back to original text works
        reverse_text = tokenizer.convert_tokens_to_string(tokens)

        if self.test_sentencepiece_ignore_case:
            reverse_text = reverse_text.lower()

        self.assertEqual(reverse_text, text)

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                new_toks = ["こんばんは", "こんにちは"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("こんばんは 国 こんにちは", add_special_tokens=False)

                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-1], tokenizer.vocab_size - 2)

    def test_tokenizer_with_zenkaku_conversion(self):
        tokenizer = RobertaJapaneseTokenizer(SAMPLE_VOCAB, do_zenkaku=True)
        # zenkaku conversion: iPhone --> ｉＰｈｏｎｅ
        self.assertListEqual(
            tokenizer.tokenize(" \tｱｯﾌﾟﾙストアでiPhone８ が  \n 発売された　。  "),
            [
                "▁ア",
                "ップ",
                "ル",
                "スト",
                "ア",
                "▁で",
                "▁",
                "ｉ",
                "Ｐ",
                "ｈ",
                "ｏｎ",
                "ｅ",
                "▁８",
                "▁が",
                "▁発",
                "売",
                "▁さ",
                "▁れ",
                "▁た",
                "▁。",
            ],
        )

    def test_full_tokenizer(self):
        tokenizer = RobertaJapaneseTokenizer(SAMPLE_VOCAB, do_zenkaku=True)

        tokens = tokenizer.tokenize("こんにちは、世界。こんばんは、渡邉さん。")
        self.assertListEqual(
            tokens,
            ["▁こ", "ん", "に", "ち", "は", "▁、", "▁世界", "▁。", "▁こ", "ん", "ば", "ん", "は", "▁、", "▁渡", "邉", "▁さん", "▁。"],
        )

        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [36, 1080, 968, 1129, 974, 9, 295, 11, 36, 1080, 1160, 1080, 974, 9, 639, 2, 733, 11]
            ],  # unk is 0 in fairseq
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["▁こ", "ん", "に", "ち", "は", "▁、", "▁世界", "▁。", "▁こ", "ん", "ば", "ん", "は", "▁、", "▁渡", "<unk>", "▁さん", "▁。"],
        )
