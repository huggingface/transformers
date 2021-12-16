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

import os
import unittest
from pathlib import Path
from shutil import copyfile

from transformers import SPIECE_UNDERLINE, is_sentencepiece_available
from transformers.models.speech_to_text import Speech2TextTokenizer
from transformers.models.speech_to_text.tokenization_speech_to_text import VOCAB_FILES_NAMES, save_json
from transformers.testing_utils import require_sentencepiece, require_tokenizers, slow

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")

if is_sentencepiece_available():
    import sentencepiece as sp


FR_CODE = 5
ES_CODE = 10


@require_sentencepiece
@require_tokenizers
class SpeechToTextTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Speech2TextTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        spm_model = sp.SentencePieceProcessor()
        spm_model.Load(SAMPLE_SP)
        vocab = ["<s>", "<pad>", "</s>", "<unk>"]

        vocab += [spm_model.IdToPiece(id_) for id_ in range(len(spm_model))]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        save_dir = Path(self.tmpdirname)
        save_json(vocab_tokens, save_dir / VOCAB_FILES_NAMES["vocab_file"])
        if not (save_dir / VOCAB_FILES_NAMES["spm_file"]).exists():
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["spm_file"])

        tokenizer = Speech2TextTokenizer.from_pretrained(self.tmpdirname)
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
        self.assertEqual(vocab_keys[-1], "j")
        self.assertEqual(len(vocab_keys), 1_001)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_001)

    def test_full_tokenizer(self):
        tokenizer = Speech2TextTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [289, 50, 14, 174, 386],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            # fmt: off
            [SPIECE_UNDERLINE + "I", SPIECE_UNDERLINE + "was", SPIECE_UNDERLINE + "b", "or", "n", SPIECE_UNDERLINE + "in", SPIECE_UNDERLINE + "", "9", "2", "0", "0", "0", ",", SPIECE_UNDERLINE + "and", SPIECE_UNDERLINE + "this", SPIECE_UNDERLINE + "is", SPIECE_UNDERLINE + "f", "al", "s", "é", "."],
            # fmt: on
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [12, 25, 88, 59, 28, 23, 11, 4, 606, 351, 351, 351, 7, 16, 70, 50, 76, 84, 10, 4, 8])

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
        expected_encoding = {'input_ids': [[3791, 797, 31, 11, 64, 797, 31, 2429, 433, 12, 1176, 12, 20, 786, 915, 142, 2413, 240, 37, 3238, 797, 31, 11, 35, 93, 915, 142, 2413, 240, 37, 5540, 567, 1276, 93, 37, 610, 40, 62, 455, 657, 1042, 123, 780, 177, 37, 309, 241, 1298, 514, 20, 292, 2737, 114, 2469, 241, 85, 64, 302, 548, 528, 423, 4, 509, 406, 423, 37, 601, 4, 777, 302, 548, 528, 423, 284, 4, 3388, 511, 459, 4, 3555, 40, 321, 302, 705, 4, 3388, 511, 583, 326, 5, 5, 5, 62, 3310, 560, 177, 2680, 217, 1508, 32, 31, 853, 418, 64, 583, 511, 1605, 62, 35, 93, 560, 177, 2680, 217, 1508, 1521, 64, 583, 511, 519, 62, 20, 1515, 764, 20, 149, 261, 5625, 7972, 20, 5540, 567, 1276, 93, 3925, 1675, 11, 15, 802, 7972, 576, 217, 1508, 11, 35, 93, 1253, 2441, 15, 289, 652, 31, 416, 321, 3842, 115, 40, 911, 8, 476, 619, 4, 380, 142, 423, 335, 240, 35, 93, 264, 8, 11, 335, 569, 420, 163, 5, 2], [260, 548, 528, 423, 20, 451, 20, 2681, 1153, 3434, 20, 5540, 37, 567, 126, 1253, 2441, 3376, 449, 210, 431, 1563, 177, 767, 5540, 11, 1203, 472, 11, 2953, 685, 285, 364, 706, 1153, 20, 6799, 20, 2869, 20, 4464, 126, 40, 2429, 20, 1040, 866, 2664, 418, 20, 318, 20, 1726, 186, 20, 265, 522, 35, 93, 2191, 4634, 20, 1040, 12, 6799, 15, 228, 2356, 142, 31, 11, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2575, 2666, 684, 1582, 1176, 12, 627, 149, 619, 20, 4902, 563, 11, 20, 149, 261, 3420, 2356, 174, 142, 4714, 131, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="facebook/s2t-small-mustc-en-de-st",
            revision="a14f04cf0776c02f62a8cb800cf7909e15ea23ad",
        )


@require_sentencepiece
class SpeechToTextTokenizerMultilinguialTest(unittest.TestCase):
    checkpoint_name = "valhalla/s2t_mustc_multilinguial_medium"

    french_text = "C'est trop cool"
    spanish_text = "Esto es genial"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: Speech2TextTokenizer = Speech2TextTokenizer.from_pretrained(cls.checkpoint_name)
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.lang_code_to_id["pt"], 4)
        self.assertEqual(self.tokenizer.lang_code_to_id["ru"], 6)
        self.assertEqual(self.tokenizer.lang_code_to_id["it"], 9)
        self.assertEqual(self.tokenizer.lang_code_to_id["de"], 11)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 10_000)

    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(ES_CODE, self.tokenizer.all_special_ids)
        generated_ids = [ES_CODE, 4, 1601, 47, 7647, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_spanish = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_spanish)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_tokenizer_adds_special_tokens(self):
        self.tokenizer.tgt_lang = "fr"
        encoded = self.tokenizer(self.french_text).input_ids
        self.assertEqual(encoded[0], FR_CODE)
        self.assertEqual(encoded[-1], self.tokenizer.eos_token_id)

    def test_tgt_lang_setter(self):
        self.tokenizer.tgt_lang = "fr"
        self.assertListEqual(self.tokenizer.prefix_tokens, [FR_CODE])

        self.tokenizer.tgt_lang = "es"
        self.assertListEqual(self.tokenizer.prefix_tokens, [ES_CODE])
