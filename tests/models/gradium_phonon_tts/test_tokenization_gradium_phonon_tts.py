# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer, GradiumPhononTTSTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/gradium_phonon_tts"


@require_tokenizers
class GradiumPhononTTSTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    """
    reproducer: the original Rust runtime's `normalize_text` / `prepare_text_prompt` (`expected_results_processing.json`,
    `gradium-ai/xn-ptts@preview-0.4.0`).
    """

    from_pretrained_id = "eustlb/gradium-phonon-tts"
    tokenizer_class = GradiumPhononTTSTokenizer
    # the common tests compare encodings with the text as written
    from_pretrained_kwargs = {"normalize": False}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fixture = json.load(open(FIXTURES_DIR / "expected_results_processing.json", encoding="utf-8"))

    def test_normalize_text(self):
        # the fixture holds the runtime's normalization before its `prepare_text_prompt`, which applies to both
        raw = self.get_tokenizer(language=None)
        for row in self.fixture["normalization"]:
            with self.subTest(language=row["lang"], text=row["input"]):
                tokenizer = self.get_tokenizer(language=row["lang"])
                self.assertEqual(tokenizer.normalize_text(row["input"]), raw.normalize_text(row["normalized"]))

    def test_prepare_text_prompt(self):
        tokenizer = self.get_tokenizer(language=None, normalize=True)
        for row in self.fixture["prepare_text_prompt"]:
            with self.subTest(text=row["input"]):
                self.assertEqual(tokenizer.normalize_text(row["input"]), row["prepared"])
                self.assertListEqual(tokenizer(row["input"])["input_ids"], row["input_ids"])

    def test_language(self):
        # English by default, saved with the tokenizer; an unknown language is an error
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.normalize_text("1+1=2 and 1500"), "1 plus 1 equals 2 and 1 thousand 500.")
        french = self.get_tokenizer(language="fr")
        self.assertEqual(french.normalize_text("1+1=2"), "1 plus 1 égal 2.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            french.save_pretrained(tmp_dir)
            reloaded = AutoTokenizer.from_pretrained(tmp_dir)
        self.assertIsInstance(reloaded, GradiumPhononTTSTokenizer)
        self.assertEqual(reloaded.language, "fr")
        with self.assertRaises(ValueError):
            self.get_tokenizer(language="it")

    def test_normalize_on_encode(self):
        tokenizer = self.get_tokenizer(normalize=True)
        expected = tokenizer("A at b.", normalize=False)["input_ids"]
        self.assertListEqual(tokenizer("a@b")["input_ids"], expected)
        # only the text between special tokens is prepared
        voice, audio_bos = tokenizer.audio_prompt_token * 2, tokenizer.audio_bos_token
        self.assertListEqual(
            tokenizer(voice + "a@b" + audio_bos)["input_ids"],
            tokenizer(voice, normalize=False)["input_ids"] + expected + tokenizer(audio_bos)["input_ids"],
        )
