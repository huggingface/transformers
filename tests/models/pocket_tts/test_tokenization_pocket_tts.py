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

import unittest

from transformers import PocketTTSTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class PocketTTSTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "eustlb/pocket-tts"
    tokenizer_class = PocketTTSTokenizer

    def test_prepare_text(self):
        # the normalizer of `tokenizer.json` (the original's `prepare_text_prompt`), then the capitalization of the
        # prepared text (exact values against the original in the processor's integration test)
        tokenizer = self.get_tokenizer()
        for text, expected in (
            ("hello world", "Hello world."),
            ("  line one\nline two", "Line one line two."),
            ('she said "goodbye"', 'She said "goodbye".'),
            ("wait —", "Wait."),
        ):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer(text)["input_ids"]), expected)

    def test_special_tokens_kept(self):
        tokenizer = self.get_tokenizer()
        voice, audio_bos = (
            tokenizer.audio_prompt_bos_token + tokenizer.audio_prompt_token * 2,
            tokenizer.audio_bos_token,
        )
        self.assertListEqual(
            tokenizer(voice + " hello world" + audio_bos)["input_ids"],
            tokenizer(voice)["input_ids"] + tokenizer("Hello world.")["input_ids"] + tokenizer(audio_bos)["input_ids"],
        )
        # the backend tokenizer gets its normalizer back after the call
        self.assertIsNotNone(tokenizer.backend_tokenizer.normalizer)

    @unittest.skip(reason="Every encoding is a prepared text (a capital, a final period): no token encodes to itself.")
    def test_internal_consistency(self):
        pass

    @unittest.skip(reason="Every encoding is a prepared text (a capital, a final period): no token encodes to itself.")
    def test_maximum_encoding_length_single_input(self):
        pass

    @unittest.skip(reason="Every encoding is a prepared text (a capital, a final period): no token encodes to itself.")
    def test_maximum_encoding_length_pair_input(self):
        pass
