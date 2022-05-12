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

import shutil
import tempfile
import unittest

from transformers import SPIECE_UNDERLINE, BatchEncoding, MBart50Tokenizer, MBart50TokenizerFast, is_torch_available
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

if is_torch_available():
    from transformers.models.mbart.modeling_mbart import shift_tokens_right

EN_CODE = 250004
RO_CODE = 250020


@require_sentencepiece
@require_tokenizers
class MBart50TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MBart50Tokenizer
    rust_tokenizer_class = MBart50TokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = MBart50Tokenizer(SAMPLE_VOCAB, src_lang="en_XX", tgt_lang="ro_RO", keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<s>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "<mask>")
        self.assertEqual(len(vocab_keys), 1_054)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_054)

    def test_full_tokenizer(self):
        tokenizer = MBart50Tokenizer(SAMPLE_VOCAB, src_lang="en_XX", tgt_lang="ro_RO", keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            # fmt: off
            [SPIECE_UNDERLINE + "I", SPIECE_UNDERLINE + "was", SPIECE_UNDERLINE + "b", "or", "n", SPIECE_UNDERLINE + "in", SPIECE_UNDERLINE + "", "9", "2", "0", "0", "0", ",", SPIECE_UNDERLINE + "and", SPIECE_UNDERLINE + "this", SPIECE_UNDERLINE + "is", SPIECE_UNDERLINE + "f", "al", "s", "é", "."],
            # fmt: on
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            # fmt: off
            [SPIECE_UNDERLINE + "I", SPIECE_UNDERLINE + "was", SPIECE_UNDERLINE + "b", "or", "n", SPIECE_UNDERLINE + "in", SPIECE_UNDERLINE + "", "<unk>", "2", "0", "0", "0", ",", SPIECE_UNDERLINE + "and", SPIECE_UNDERLINE + "this", SPIECE_UNDERLINE + "is", SPIECE_UNDERLINE + "f", "al", "s", "<unk>", "."],
            # fmt: on
        )

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[250004, 11062, 82772, 7, 15, 82772, 538, 51529, 237, 17198, 1290, 206, 9, 215175, 1314, 136, 17198, 1290, 206, 9, 56359, 42, 122009, 9, 16466, 16, 87344, 4537, 9, 4717, 78381, 6, 159958, 7, 15, 24480, 618, 4, 527, 22693, 5428, 4, 2777, 24480, 9874, 4, 43523, 594, 4, 803, 18392, 33189, 18, 4, 43523, 24447, 12399, 100, 24955, 83658, 9626, 144057, 15, 839, 22335, 16, 136, 24955, 83658, 83479, 15, 39102, 724, 16, 678, 645, 2789, 1328, 4589, 42, 122009, 115774, 23, 805, 1328, 46876, 7, 136, 53894, 1940, 42227, 41159, 17721, 823, 425, 4, 27512, 98722, 206, 136, 5531, 4970, 919, 17336, 5, 2], [250004, 20080, 618, 83, 82775, 47, 479, 9, 1517, 73, 53894, 333, 80581, 110117, 18811, 5256, 1295, 51, 152526, 297, 7986, 390, 124416, 538, 35431, 214, 98, 15044, 25737, 136, 7108, 43701, 23, 756, 135355, 7, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [250004, 581, 63773, 119455, 6, 147797, 88203, 7, 645, 70, 21, 3285, 10269, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="facebook/mbart-large-50",
            revision="d3913889c59cd5c9e456b269c376325eabad57e2",
        )

    # overwrite from test_tokenization_common to speed up test
    def test_save_pretrained(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        self.tokenizers_list[0] = (self.rust_tokenizer_class, "hf-internal-testing/tiny-random-mbart50", {})
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files + the tokenizer.json file for the fast one
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))
                tokenizer_r_files = tuple(f for f in tokenizer_r_files if "tokenizer.json" not in f)
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))
                    # self.assertEqual(getattr(tokenizer_rp, key), getattr(tokenizer_pp, key))
                    # self.assertEqual(getattr(tokenizer_rp, key + "_id"), getattr(tokenizer_pp, key + "_id"))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=True
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=True)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=False
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=False)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it saved the tokenizer.json file
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)


@require_torch
@require_sentencepiece
@require_tokenizers
class MBart50OneToManyIntegrationTest(unittest.TestCase):
    checkpoint_name = "facebook/mbart-large-50-one-to-many-mmt"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei"
        ' pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor'
        " face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
    ]
    expected_src_tokens = [EN_CODE, 8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: MBart50Tokenizer = MBart50Tokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="en_XX", tgt_lang="ro_RO"
        )
        cls.pad_token_id = 1
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ar_AR"], 250001)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["en_EN"], 250004)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ro_RO"], 250020)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["mr_IN"], 250038)

    def test_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [RO_CODE, 884, 9019, 96, 9, 916, 86792, 36, 18743, 15596, 5, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[0], EN_CODE)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(["<mask>", "ar_AR"]), [250053, 250001])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = MBart50Tokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids, original_special_tokens)

    @require_torch
    def test_batch_fairseq_parity(self):
        batch = self.tokenizer(self.src_text, padding=True)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(self.tgt_text, padding=True, return_tensors="pt")
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(labels, self.tokenizer.pad_token_id).tolist()
        labels = labels.tolist()

        # fairseq batch: https://gist.github.com/sshleifer/cba08bc2109361a74ac3760a7e30e4f4
        assert batch.input_ids[1][0] == EN_CODE
        assert batch.input_ids[1][-1] == 2
        assert labels[1][0] == RO_CODE
        assert labels[1][-1] == 2
        assert batch.decoder_input_ids[1][:2] == [2, RO_CODE]

    @require_torch
    def test_tokenizer_prepare_batch(self):
        batch = self.tokenizer(
            self.src_text, padding=True, truncation=True, max_length=len(self.expected_src_tokens), return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                self.tgt_text,
                padding=True,
                truncation=True,
                max_length=len(self.expected_src_tokens),
                return_tensors="pt",
            )
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(labels, self.tokenizer.pad_token_id)

        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 14), batch.input_ids.shape)
        self.assertEqual((2, 14), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(2, batch.decoder_input_ids[0, 0])  # decoder_start_token_id
        # Test that special tokens are reset
        self.assertEqual(self.tokenizer.prefix_tokens, [EN_CODE])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    def test_seq2seq_max_target_length(self):
        batch = self.tokenizer(self.src_text, padding=True, truncation=True, max_length=3, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors="pt")
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(labels, self.tokenizer.pad_token_id)

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)

    @require_torch
    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs(
            "A test", return_tensors="pt", src_lang="en_XX", tgt_lang="ar_AR"
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                # en_XX, A, test, EOS
                "input_ids": [[250004, 62, 3034, 2]],
                "attention_mask": [[1, 1, 1, 1]],
                # ar_AR
                "forced_bos_token_id": 250001,
            },
        )
