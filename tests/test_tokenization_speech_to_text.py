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
from transformers.testing_utils import require_sentencepiece, require_tokenizers

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
