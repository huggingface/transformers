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

import shutil
import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceAsrProcessor,
)
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


class VibeVoiceAsrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VibeVoiceAsrProcessor
    tiny_model_id = "hf-internal-testing/tiny-processor-vibevoice_asr"
    checkpoint = "microsoft/VibeVoice-ASR-HF"

    @classmethod
    @require_torch
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = VibeVoiceAsrProcessor.from_pretrained(cls.tiny_model_id)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = VibeVoiceAsrProcessor.from_pretrained(self.tiny_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        processor = VibeVoiceAsrProcessor.from_pretrained(self.tiny_model_id)
        feature_extractor = processor.feature_extractor

        processor = VibeVoiceAsrProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = VibeVoiceAsrProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, VibeVoiceAcousticTokenizerFeatureExtractor)

    @require_torch
    def test_apply_transcription_request_single(self):
        processor = self.get_processor()

        audio_url = "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
        helper_outputs = processor.apply_transcription_request(audio=audio_url, prompt="About VibeVoice")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "About VibeVoice"},
                    {
                        "type": "audio",
                        "path": audio_url,
                    },
                ],
            }
        ]
        manual_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )

        for key in ("input_ids", "attention_mask", "input_values", "padding_mask"):
            self.assertIn(key, helper_outputs)
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        self.skipTest("VibeVoiceAsrProcessor does not support chat templates with text-only inputs.")

    def test_apply_chat_template_assistant_mask(self):
        self.skipTest("VibeVoiceAsrProcessor does not support chat templates with text-only inputs.")

    @require_torch
    def test_decode_output_formats(self):
        import torch
        from unittest.mock import patch

        processor = self.get_processor()

        # The original test used hardcoded Qwen token IDs that decode to JSON-formatted output
        # with the full tokenizer. Here we mock batch_decode so the test focuses on the
        # processor's parsing logic (return_format="parsed" / "transcription_only") rather than
        # tokenizer vocabulary — the tiny tokenizer would decode those IDs to garbage, causing
        # json.loads() to fail. End-to-end tokenizer correctness belongs in model integration tests.
        generated_ids = torch.tensor([[0]])
        mock_decoded = [
            '<|im_start|>assistant\n[{"Start":0,"End":7.56,"Speaker":0,"Content":"Revevoices is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio."}]<|im_end|>\n<|endoftext|>'
        ]

        with patch.object(processor.tokenizer, "batch_decode", return_value=mock_decoded):
            # test parsed output
            dicts = processor.decode(generated_ids, return_format="parsed")
            self.assertIsInstance(dicts, list)
            self.assertIsInstance(dicts[0], list)
            self.assertIsInstance(dicts[0][0], dict)
            self.assertIn("Content", dicts[0][0])
            self.assertIn("Start", dicts[0][0])
            self.assertIn("End", dicts[0][0])
            self.assertIsInstance(dicts[0][0]["Start"], float)
            self.assertIsInstance(dicts[0][0]["End"], float)

            # test transcript only
            transcript = processor.decode(generated_ids, return_format="transcription_only")
            self.assertIsInstance(transcript, list)
            self.assertIsInstance(transcript[0], str)
