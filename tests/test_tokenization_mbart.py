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

from transformers import SPIECE_UNDERLINE, BatchEncoding, MBartTokenizer, MBartTokenizerFast, is_torch_available
from transformers.testing_utils import (
    _sentencepiece_available,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)

from .test_tokenization_common import TokenizerTesterMixin


if _sentencepiece_available:
    from .test_tokenization_xlm_roberta import SAMPLE_VOCAB


if is_torch_available():
    from transformers.models.bart.modeling_bart import shift_tokens_right

EN_CODE = 250004
RO_CODE = 250020


@require_sentencepiece
@require_tokenizers
class MBartTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MBartTokenizer
    rust_tokenizer_class = MBartTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = MBartTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = MBartTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
                #                                       ^ unk: 2 + 1 = 3                  unk: 2 + 1 = 3 ^
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )


@require_torch
@require_sentencepiece
@require_tokenizers
class MBartEnroIntegrationTest(unittest.TestCase):
    checkpoint_name = "facebook/mbart-large-en-ro"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        'Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.',
    ]
    expected_src_tokens = [8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2, EN_CODE]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained(cls.checkpoint_name)
        cls.pad_token_id = 1
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ar_AR"], 250001)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["en_EN"], 250004)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ro_RO"], 250020)

    def test_enro_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_enro_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [RO_CODE, 884, 9019, 96, 9, 916, 86792, 36, 18743, 15596, 5, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_enro_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer.prepare_seq2seq_batch(
            src_text,
            max_length=desired_max_length,
        ).input_ids[0]
        self.assertEqual(ids[-2], 2)
        self.assertEqual(ids[-1], EN_CODE)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(["<mask>", "ar_AR"]), [250026, 250001])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = MBartTokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids, original_special_tokens)

    # prepare_seq2seq_batch tests below

    @require_torch
    def test_batch_fairseq_parity(self):
        batch: BatchEncoding = self.tokenizer.prepare_seq2seq_batch(
            self.src_text, tgt_texts=self.tgt_text, return_tensors="pt"
        )
        batch["decoder_input_ids"] = shift_tokens_right(batch.labels, self.tokenizer.pad_token_id)
        for k in batch:
            batch[k] = batch[k].tolist()
        # batch = {k: v.tolist() for k,v in batch.items()}
        # fairseq batch: https://gist.github.com/sshleifer/cba08bc2109361a74ac3760a7e30e4f4
        # batch.decoder_inputs_ids[0][0] ==
        assert batch.input_ids[1][-2:] == [2, EN_CODE]
        assert batch.decoder_input_ids[1][0] == RO_CODE
        assert batch.decoder_input_ids[1][-1] == 2
        assert batch.labels[1][-2:] == [2, RO_CODE]

    @require_torch
    def test_enro_tokenizer_prepare_seq2seq_batch(self):
        batch = self.tokenizer.prepare_seq2seq_batch(
            self.src_text, tgt_texts=self.tgt_text, max_length=len(self.expected_src_tokens), return_tensors="pt"
        )
        batch["decoder_input_ids"] = shift_tokens_right(batch.labels, self.tokenizer.pad_token_id)
        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 14), batch.input_ids.shape)
        self.assertEqual((2, 14), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(2, batch.decoder_input_ids[0, -1])  # EOS
        # Test that special tokens are reset
        self.assertEqual(self.tokenizer.prefix_tokens, [])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id, EN_CODE])

    def test_seq2seq_max_target_length(self):
        batch = self.tokenizer.prepare_seq2seq_batch(
            self.src_text, tgt_texts=self.tgt_text, max_length=3, max_target_length=10, return_tensors="pt"
        )
        batch["decoder_input_ids"] = shift_tokens_right(batch.labels, self.tokenizer.pad_token_id)
        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)
        # max_target_length will default to max_length if not specified
        batch = self.tokenizer.prepare_seq2seq_batch(
            self.src_text, tgt_texts=self.tgt_text, max_length=3, return_tensors="pt"
        )
        batch["decoder_input_ids"] = shift_tokens_right(batch.labels, self.tokenizer.pad_token_id)
        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 3)
