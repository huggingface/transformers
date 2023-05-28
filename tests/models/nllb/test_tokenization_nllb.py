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

import shutil
import tempfile
import unittest

from transformers import (
    SPIECE_UNDERLINE,
    AddedToken,
    BatchEncoding,
    NllbTokenizer,
    NllbTokenizerFast,
    is_torch_available,
)
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

EN_CODE = 256047
RO_CODE = 256145


@require_sentencepiece
@require_tokenizers
class NllbTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = NllbTokenizer
    rust_tokenizer_class = NllbTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = NllbTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = NllbTokenizer(SAMPLE_VOCAB, keep_accents=True)

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

    # overwrite from test_tokenization_common to speed up test
    def test_save_pretrained(self):
        self.tokenizers_list[0] = (self.rust_tokenizer_class, "hf-internal-testing/tiny-random-nllb", {})
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
    def test_prepare_seq2seq_batch(self):
        if not self.test_seq2seq:
            return

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Longer text that will definitely require truncation.
                src_text = [
                    " UN Chief Says There Is No Military Solution in Syria",
                    " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for"
                    " Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons"
                    " will only worsen the violence and misery for millions of people.",
                ]
                tgt_text = [
                    "Şeful ONU declară că nu există o soluţie militară în Siria",
                    "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al"
                    ' Rusiei pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi'
                    " că noi arme nu vor face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
                ]
                try:
                    batch = tokenizer.prepare_seq2seq_batch(
                        src_texts=src_text,
                        tgt_texts=tgt_text,
                        max_length=3,
                        max_target_length=10,
                        return_tensors="pt",
                        src_lang="eng_Latn",
                        tgt_lang="ron_Latn",
                    )
                except NotImplementedError:
                    return
                self.assertEqual(batch.input_ids.shape[1], 3)
                self.assertEqual(batch.labels.shape[1], 10)
                # max_target_length will default to max_length if not specified
                batch = tokenizer.prepare_seq2seq_batch(
                    src_text, tgt_texts=tgt_text, max_length=3, return_tensors="pt"
                )
                self.assertEqual(batch.input_ids.shape[1], 3)
                self.assertEqual(batch.labels.shape[1], 3)

                batch_encoder_only = tokenizer.prepare_seq2seq_batch(
                    src_texts=src_text, max_length=3, max_target_length=10, return_tensors="pt"
                )
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    @unittest.skip("Unfortunately way too slow to build a BPE with SentencePiece.")
    def test_save_slow_from_fast_and_reload_fast(self):
        pass

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name,
                        additional_special_tokens=added_tokens,
                        **kwargs,  # , from_slow=True <- unfortunately too slow to convert
                    )
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    p_output = tokenizer_p.encode("Hey this is a <special> token")

                    cr_output = tokenizer_cr.encode("Hey this is a <special> token")

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)


@require_torch
@require_sentencepiece
@require_tokenizers
class NllbDistilledIntegrationTest(unittest.TestCase):
    checkpoint_name = "facebook/nllb-200-distilled-600M"
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
    expected_src_tokens = [
        256047,
        16297,
        134408,
        8165,
        248066,
        14734,
        950,
        1135,
        105721,
        3573,
        83,
        27352,
        108,
        49486,
        2,
    ]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: NllbTokenizer = NllbTokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="eng_Latn", tgt_lang="ron_Latn"
        )
        cls.pad_token_id = 1
        return cls

    def test_language_codes(self):
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ace_Arab"], 256001)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ace_Latn"], 256002)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["fra_Latn"], 256057)

    def test_enro_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_enro_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        # fmt: off
        generated_ids = [RO_CODE, 4254, 98068, 112923, 39072, 3909, 713, 102767, 26, 17314, 35642, 14683, 33118, 2022, 66987, 2, 256047]
        # fmt: on

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_enro_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[-1], 2)
        self.assertEqual(ids[0], EN_CODE)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(["<mask>", "ar_AR"]), [256203, 3])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = NllbTokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids, original_special_tokens)

    @require_torch
    def test_enro_tokenizer_prepare_batch(self):
        batch = self.tokenizer(
            self.src_text,
            text_target=self.tgt_text,
            padding=True,
            truncation=True,
            max_length=len(self.expected_src_tokens),
            return_tensors="pt",
        )
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.lang_code_to_id["ron_Latn"]
        )

        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 15), batch.input_ids.shape)
        self.assertEqual((2, 15), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(RO_CODE, batch.decoder_input_ids[0, 0])  # EOS
        # Test that special tokens are reset
        self.assertEqual(self.tokenizer.prefix_tokens, [EN_CODE])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    def test_seq2seq_max_length(self):
        batch = self.tokenizer(self.src_text, padding=True, truncation=True, max_length=3, return_tensors="pt")
        targets = self.tokenizer(
            text_target=self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors="pt"
        )
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(
            labels,
            self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang],
        )

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)

    @require_torch
    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs(
            "A test", return_tensors="pt", src_lang="eng_Latn", tgt_lang="fra_Latn"
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                # A, test, EOS, en_XX
                "input_ids": [[256047, 70, 7356, 2]],
                "attention_mask": [[1, 1, 1, 1]],
                # ar_AR
                "forced_bos_token_id": 256057,
            },
        )

    @require_torch
    def test_legacy_behaviour(self):
        self.tokenizer.legacy_behaviour = True
        inputs = self.tokenizer(
            "UN Chief says there is no military solution in Syria", src_lang="eng_Latn", tgt_lang="fra_Latn"
        )
        self.assertEqual(
            inputs.input_ids, [16297, 134408, 25653, 6370, 248, 254, 103929, 94995, 108, 49486, 2, 256047]
        )

        self.tokenizer.legacy_behaviour = False
        inputs = self.tokenizer(
            "UN Chief says there is no military solution in Syria", src_lang="eng_Latn", tgt_lang="fra_Latn"
        )
        self.assertEqual(
            inputs.input_ids, [256047, 16297, 134408, 25653, 6370, 248, 254, 103929, 94995, 108, 49486, 2]
        )
