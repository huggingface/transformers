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

import tempfile
import unittest

from transformers import AutoTokenizer, PocketTTSTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


# the settings of the original's French config (`pocket_tts/config/french.yaml`)
FRENCH_SETTINGS = {
    "remove_semicolons": True,
    "replace_characters": {
        '"': "",
        "“": "",
        "”": "",
        "„": "",
        "«": "",
        "»": "",
        "’": "'",
        "‘": "'",
        ":": ",",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
    },
}


@require_tokenizers
class PocketTTSTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "eustlb/pocket-tts"
    tokenizer_class = PocketTTSTokenizer
    # the common tests compare encodings with the text as written
    from_pretrained_kwargs = {"normalize": False}

    def test_normalize_text(self):
        # the original's `prepare_text_prompt` (exact values against the original in the processor's integration test)
        tokenizer = self.get_tokenizer()
        for text, expected in (
            ("hello world", "Hello world."),
            ("  Line one\nline two\r\nline three", "Line one line two line three."),
            ('She said "goodbye"', 'She said "goodbye".'),
            ("Wait —", "Wait."),
            ("Is this the end…", "Is this the end…"),
            ("It costs 3.5 dollars; really?", "It costs 3.5 dollars; really?"),
        ):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.normalize_text(text), expected)
        with self.assertRaises(ValueError):
            tokenizer.normalize_text("   ")

    def test_normalize_text_language_settings(self):
        tokenizer = self.get_tokenizer(**FRENCH_SETTINGS)
        # quotes deleted, then the comma they leave after a sentence mark; `:` -> `,` and `;` -> `,`
        self.assertEqual(tokenizer.normalize_text("« Vraiment ? », demanda-t-elle"), "Vraiment ? demanda-t-elle.")
        self.assertEqual(
            tokenizer.normalize_text("Il a dit : « bonjour » ; puis il est parti"),
            "Il a dit , bonjour , puis il est parti.",
        )
        # the settings are saved with the tokenizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            reloaded = AutoTokenizer.from_pretrained(tmp_dir)
        self.assertIsInstance(reloaded, PocketTTSTokenizer)
        self.assertEqual(reloaded.replace_characters, FRENCH_SETTINGS["replace_characters"])
        self.assertTrue(reloaded.remove_semicolons)

    def test_normalize_on_encode(self):
        tokenizer = self.get_tokenizer(normalize=True)
        expected = tokenizer("Hello world.", normalize=False)["input_ids"]
        self.assertListEqual(tokenizer("hello world")["input_ids"], expected)
        self.assertListEqual(tokenizer(["hello world"])["input_ids"], [expected])
        # per call, the text as written
        self.assertNotEqual(tokenizer("hello world", normalize=False)["input_ids"], expected)
        # only the text between special tokens is prepared
        voice, audio_bos = (
            tokenizer.audio_prompt_bos_token + tokenizer.audio_prompt_token * 2,
            tokenizer.audio_bos_token,
        )
        self.assertListEqual(
            tokenizer(voice + " hello world" + audio_bos)["input_ids"],
            tokenizer(voice, normalize=False)["input_ids"] + expected + tokenizer(audio_bos)["input_ids"],
        )
