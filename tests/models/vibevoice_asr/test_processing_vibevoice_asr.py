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
    # Tiny processor created with make_tiny_processor.py from "microsoft/VibeVoice-ASR-HF"
    tiny_model_id = "hf-internal-testing/tiny-processor-vibevoice_asr"

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

        audio_url = (
            "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
        )
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
        from unittest.mock import patch

        import torch

        processor = self.get_processor()

        # This test is about the processor's ability to parse the model output into structured
        # dicts (return_format="parsed") or plain transcriptions (return_format="transcription_only").
        # We are NOT testing tokenizer decoding here, so it is fine to mock batch_decode.
        # The mock string below is the exact output obtained by decoding the original generated_ids
        # with the full processor (microsoft/VibeVoice-ASR-HF) prior to PR #47213, which switched
        # to a tiny tokenizer that would decode those IDs to garbage and break json.loads().
        generated_ids = torch.tensor([[0]])
        # The decode method calls tokenizer.decode (singular) with skip_special_tokens=True.
        # When called with a 2D tensor (batch), the tokenizer returns a list of strings.
        # extract_speaker_dict then returns list[list[dict]] for a list input.
        # The mock string has special tokens already stripped (skip_special_tokens=True).
        mock_decoded = [
            'assistant\n[{"Start":0,"End":7.56,"Speaker":0,"Content":"Revevoices is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio."}]\n'
        ]

        with patch.object(processor.tokenizer, "decode", return_value=mock_decoded):
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
