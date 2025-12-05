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

import os
import tempfile
import unittest

from transformers import (
    AddedToken,
    BatchEncoding,
    NllbTokenizer,
    is_torch_available,
)
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

EN_CODE = 256047
RO_CODE = 256145


@require_sentencepiece
@require_tokenizers
class NllbTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/nllb-200-distilled-600M"
    tokenizer_class = NllbTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<unk>', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁enc', 'od', 'ed', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [9680, 248, 9, 7356, 248059, 253515, 117, 1398, 79519, 108, 855, 45299, 248079, 540, 3423, 248, 52428, 248132, 248075, 182892, 248506, 249573, 3, 249221, 2867, 94124, 2867, 94124, 94124, 248059, 0, 435, 0, 6370, 1617, 45893, 191422, 12516, 280, 242514, 12025, 129, 76, 248144, 94124, 248075, 9062, 528, 248072, 540, 99681, 528, 248072, 34744, 27426, 11657, 2442, 1259, 34512]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<unk>', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁enc', 'od', 'ed', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真<unk>是 Hi Hello Hi Hello Hello <s> hi<s> there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    # @classmethod
    # def setUpClass(cls):
    #     super().setUpClass()

    #     # Extract vocab from SentencePiece model
    #     extractor = SentencePieceExtractor(SAMPLE_VOCAB)
    #     vocab_ids, vocab_scores, merges = extractor.extract()

    #     # Create tokenizer with extracted vocab
    #     tokenizer = NllbTokenizer(vocab=vocab_scores)
    #     tokenizer.save_pretrained(cls.tmpdirname)

    @require_torch
    def test_prepare_seq2seq_batch(self):
        if not self.test_seq2seq:
            self.skipTest(reason="test_seq2seq is set to False")

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
                    self.skipTest(reason="Encountered NotImplementedError when calling prepare_seq2seq_batch")
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

    @unittest.skip(reason="Unfortunately way too slow to build a BPE with SentencePiece.")
    def test_save_slow_from_fast_and_reload_fast(self):
        pass

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.get_tokenizer(pretrained_name, additional_special_tokens=added_tokens, **kwargs)
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

    @unittest.skip(reason="Need to fix this after #26538")
    def test_training_new_tokenizer(self):
        pass

    def test_new_language_codes(self):
        code1, code2 = "myv_Cyrl", "myv_Latn"
        new_codes = FAIRSEQ_LANGUAGE_CODES + [code1, code2]
        # here I create a tokenizer with the default behaviour
        tok1 = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        # here I enhance the model's vocabulary with two new language codes
        tok2 = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", additional_special_tokens=new_codes)

        # testing that the new codes can work
        self.assertEqual(len(tok2), len(tok1) + 2)
        tok2.tgt_lang = code1
        tok2.src_lang = code2

        self.assertEqual(tok2("šumbrat!").input_ids[0], tok2.convert_tokens_to_ids(code2))
        with tempfile.TemporaryDirectory() as tempdir:
            # testing that saving and loading the tokenizer preserves the new behaviour
            tok2.save_pretrained(tempdir)
            tok3 = NllbTokenizer.from_pretrained(tempdir)
            self.assertEqual(tok2.get_vocab(), tok3.get_vocab())
            tok3.src_lang = code2
            self.assertEqual(tok3("šumbrat!").input_ids[0], tok3.convert_tokens_to_ids(code2))

            # testing that saving and loading the tokenizer preserves the new behaviour
            tok2.save_pretrained(tempdir)
            # Use the original vocab_file from tok2, or load from saved directory
            vocab_file = tok2.vocab_file if hasattr(tok2, "vocab_file") and tok2.vocab_file else None
            if vocab_file is None or not os.path.exists(vocab_file):
                # Fallback: load from saved directory to get vocab_file
                tok_temp = NllbTokenizer.from_pretrained(tempdir)
                vocab_file = tok_temp.vocab_file if hasattr(tok_temp, "vocab_file") and tok_temp.vocab_file else None
            # Extract vocab and merges from sentencepiece model
            if vocab_file and os.path.exists(vocab_file):
                extractor = SentencePieceExtractor(vocab_file)
                vocab_ids, vocab_scores, merges = extractor.extract()
                tok3 = NllbTokenizer(
                    vocab=vocab_ids, merges=merges, vocab_file=vocab_file, additional_special_tokens=None
                )
                self.assertEqual(len(tok3), 256204)  # legacy
                tok4 = NllbTokenizer(
                    vocab=vocab_ids, merges=merges, vocab_file=vocab_file, additional_special_tokens=[]
                )
                self.assertEqual(len(tok4), 256002)
                tok5 = NllbTokenizer(
                    vocab=vocab_ids, merges=merges, vocab_file=vocab_file, additional_special_tokens=[code1, code2]
                )
                self.assertEqual(len(tok5), 256004)


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

    def test_enro_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_enro_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [RO_CODE, 4254, 98068, 112923, 39072, 3909, 713, 102767, 26, 17314, 35642, 14683, 33118, 2022, 66987, 2, 256047]  # fmt: skip

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
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.convert_tokens_to_ids("ron_Latn")
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
            decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang),
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
