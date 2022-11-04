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

import tempfile
import unittest

from transformers import SPIECE_UNDERLINE, BartJapaneseTokenizer, BatchEncoding
from transformers.testing_utils import (
    require_jumanpp,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_zenhan,
)

from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
@require_jumanpp
@require_zenhan
class BartJapaneseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    checkpoint_name = "Formzu/bart-base-japanese"
    tokenizer_class = BartJapaneseTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = False
    src_text = ["ｱｯﾌﾟﾙストアでiPhone８が発売された", "こんにちは、世界。こんばんは、世界。"]
    tgt_text = ["ｱｯﾌﾟﾙストアでiPhone８が発売された", "こんにちは、世界。こんばんは、世界。"]
    expected_src_tokens = [0, 11748, 8765, 12, 18453, 163, 11, 546, 23, 45, 2]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: BartJapaneseTokenizer = BartJapaneseTokenizer.from_pretrained(
            cls.checkpoint_name, src_lang=None, tgt_lang=None
        )
        return cls

    def setUp(self):
        super().setUp()

        self.tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize("これはテストです")
        self.assertListEqual(tokens, ["▁これ", "▁は", "▁テスト", "▁です"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [179, 7, 4116, 4561]],
        )

        tokens = tokenizer.tokenize("私は92000年生まれで、これは噓です。")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "私",
                SPIECE_UNDERLINE + "は",
                SPIECE_UNDERLINE + "９２",
                "０００",
                SPIECE_UNDERLINE + "年",
                SPIECE_UNDERLINE + "生まれ",
                SPIECE_UNDERLINE + "で",
                SPIECE_UNDERLINE + "、",
                SPIECE_UNDERLINE + "これ",
                SPIECE_UNDERLINE + "は",
                SPIECE_UNDERLINE,
                "噓",
                SPIECE_UNDERLINE + "です",
                SPIECE_UNDERLINE + "。",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [1328, 7, 7704, 838, 14, 1536, 11, 4, 179, 7, 28426, 2, 4561, 5]
                #                                                  unk: 2 + 1 = 3 ^
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "私",
                SPIECE_UNDERLINE + "は",
                SPIECE_UNDERLINE + "９２",
                "０００",
                SPIECE_UNDERLINE + "年",
                SPIECE_UNDERLINE + "生まれ",
                SPIECE_UNDERLINE + "で",
                SPIECE_UNDERLINE + "、",
                SPIECE_UNDERLINE + "これ",
                SPIECE_UNDERLINE + "は",
                SPIECE_UNDERLINE,
                "<unk>",
                SPIECE_UNDERLINE + "です",
                SPIECE_UNDERLINE + "。",
            ],
        )

    def get_input_output_texts(self, tokenizer):
        input_text = "こんにちは、世界。こんばんは、世界。こんにちは、世界。こんばんは、世界。こんにちは、世界。"
        output_text = "こんにちは 、 世界 。 こんばんは 、 世界 。 こんにちは 、 世界 。 こんばんは 、 世界 。 こんにちは 、 世界 。"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        toks_ids = tokenizer.encode(output_text, add_special_tokens=False)
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    def test_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_tokenizer_truncation(self):
        src_text = ["これは長文になります。" * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[-1], 2)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(["<mask>"]), [32001])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = BartJapaneseTokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids, original_special_tokens)

    @require_torch
    def test_batch_fairseq_parity(self):
        batch = self.tokenizer(self.src_text, text_target=self.tgt_text, padding=True, return_tensors="pt")

        assert batch.input_ids[1][-1:].tolist() == [2]
        assert batch.labels[1][-1] == 2
        assert batch.labels[1][-1:].tolist() == [2]

    @require_torch
    def test_tokenizer_prepare_batch(self):
        batch = self.tokenizer(
            self.src_text,
            text_target=self.tgt_text,
            padding=True,
            truncation=True,
            max_length=len(self.expected_src_tokens),
            return_tensors="pt",
        )

        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 11), batch.input_ids.shape)
        self.assertEqual((2, 11), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(2, batch.labels[0, -1])  # EOS
        self.assertEqual(0, batch.labels[0, 0])  # BOS
        # Test that special tokens are reset
        self.assertEqual(self.tokenizer.prefix_tokens, [self.tokenizer.bos_token_id])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    def test_seq2seq_max_length(self):
        batch = self.tokenizer(self.src_text, padding=True, truncation=True, max_length=3, return_tensors="pt")
        targets = self.tokenizer(
            text_target=self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors="pt"
        )

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(targets.input_ids.shape[1], 10)

    def test_pickle_tokenizer(self):
        pass  # TODO add if relevant

    def test_pretokenized_inputs(self):
        pass  # TODO add if relevant
