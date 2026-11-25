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

from transformers import VoxtralTtsConfig
from transformers.testing_utils import is_torch_available, require_mistral_common, require_torch


if is_torch_available():
    import torch

    from transformers import MistralCommonBackend, VoxtralTtsProcessor


@require_mistral_common
@require_torch
class VoxtralTtsProcessorTest(unittest.TestCase):
    model_id = "mistralai/Voxtral-4B-TTS-2603"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MistralCommonBackend.from_pretrained(cls.model_id)
        generator = torch.Generator().manual_seed(0)
        # Real presets have more placeholders (e.g. 147 for `casual_male`), a few are enough here
        cls.voice_embeddings = {
            "casual_male": torch.randn(5, 8, generator=generator),
            "fr_female": torch.randn(3, 8, generator=generator),
        }

    def get_processor(self):
        return VoxtralTtsProcessor(tokenizer=self.tokenizer, voice_embeddings=self.voice_embeddings)

    def test_prompt_matches_mistral_common(self):
        from mistral_common.protocol.speech.request import SpeechRequest

        processor = self.get_processor()
        text = "Paris is a beautiful city!"
        expected = self.tokenizer.tokenizer.instruct_tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice="casual_male")
        ).tokens
        audio_token_id = self.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids.audio
        num_audio_tokens = expected.count(audio_token_id)
        self.assertEqual(processor._build_prompt(text, num_audio_tokens), expected)

    def test_call_with_preset_voices(self):
        processor = self.get_processor()
        inputs = processor(["Hello there, how are you?", "Bonjour"], voice=["casual_male", "fr_female"])
        self.assertEqual(set(inputs.keys()), {"input_ids", "attention_mask", "voice_embeddings"})
        self.assertEqual(inputs["input_ids"].shape, inputs["attention_mask"].shape)

        audio_token_id = self.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids.audio
        self.assertEqual((inputs["input_ids"] == audio_token_id).sum(-1).tolist(), [5, 3])
        torch.testing.assert_close(
            inputs["voice_embeddings"],
            torch.cat([self.voice_embeddings["casual_male"], self.voice_embeddings["fr_female"]]),
        )
        # Left padding
        lengths = inputs["attention_mask"].sum(-1)
        self.assertEqual(int(lengths.max()), inputs["input_ids"].shape[1])
        shorter = int(lengths.argmin())
        num_pad = inputs["input_ids"].shape[1] - int(lengths[shorter])
        self.assertTrue((inputs["attention_mask"][shorter, :num_pad] == 0).all())
        self.assertTrue((inputs["input_ids"][shorter, :num_pad] == self.tokenizer.pad_token_id).all())

        # A single voice is used for every text
        inputs = processor(["Hello", "World"], voice="fr_female")
        self.assertEqual(inputs["voice_embeddings"].shape, (6, 8))

    def test_call_with_custom_voice_embeddings(self):
        processor = self.get_processor()
        custom = torch.randn(4, 8)
        inputs = processor("Hello", voice_embeddings=custom)
        audio_token_id = self.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids.audio
        self.assertEqual(int((inputs["input_ids"] == audio_token_id).sum()), 4)
        torch.testing.assert_close(inputs["voice_embeddings"], custom)

    def test_invalid_voice_arguments(self):
        processor = self.get_processor()
        with self.assertRaisesRegex(ValueError, "Unknown voice"):
            processor("Hello", voice="robot")
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            processor("Hello")
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            processor("Hello", voice="casual_male", voice_embeddings=torch.randn(2, 8))
        with self.assertRaisesRegex(ValueError, "2 texts"):
            processor(["Hello", "World"], voice=["casual_male"])

    def test_save_load_keeps_voices(self):
        processor = self.get_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            # As in the published repo, the model config lets the tokenizer resolve to `MistralCommonBackend`
            VoxtralTtsConfig().save_pretrained(tmpdir)
            processor.save_pretrained(tmpdir)
            reloaded = VoxtralTtsProcessor.from_pretrained(tmpdir)
        self.assertEqual(reloaded.available_voices, ["casual_male", "fr_female"])
        for name, embedding in self.voice_embeddings.items():
            torch.testing.assert_close(reloaded.voice_embeddings[name], embedding)
        self.assertEqual(reloaded.sampling_rate, 24000)
        self.assertEqual(
            reloaded("Hello", voice="casual_male")["input_ids"].tolist(),
            processor("Hello", voice="casual_male")["input_ids"].tolist(),
        )
