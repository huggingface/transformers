# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import AutoProcessor, CanaryProcessor
from transformers.testing_utils import require_torch


def _get_prompt(source: str, target: str, pnc: bool = True, timestamps: bool = False) -> str:
    return (
        "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
        f"<|{source}|><|{target}|>"
        f"{'<|pnc|>' if pnc else '<|nopnc|>'}<|noitn|>"
        f"{'<|timestamp|>' if timestamps else '<|notimestamp|>'}<|nodiarize|>"
    )


@require_torch
class CanaryProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "harshaljanjani/canary-1b-v2-hf"
        cls.tmpdirname = tempfile.mkdtemp()
        CanaryProcessor.from_pretrained(cls.checkpoint).save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.tmpdirname)

    def _audio(self, num_samples: int = 16000):
        return np.zeros(num_samples, dtype=np.float32)

    def _decode_prompt(self, processor, inputs, index: int = 0) -> str:
        return processor.tokenizer.decode(inputs["decoder_input_ids"][index], skip_special_tokens=False)

    def test_chat_template_is_loaded(self):
        self.assertIsNotNone(self.get_processor().chat_template)

    def test_apply_transcription_request_transcription(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(audio=self._audio(), source_language="en")
        self.assertIn("input_features", inputs)
        self.assertEqual(self._decode_prompt(processor, inputs), _get_prompt("en", "en"))

    def test_apply_transcription_request_translation(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(audio=self._audio(), source_language="en", target_language="de")
        self.assertEqual(self._decode_prompt(processor, inputs), _get_prompt("en", "de"))

    def test_punctuation_and_timestamps_flags(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(
            audio=self._audio(), source_language="en", punctuation=False, timestamps=True
        )
        self.assertEqual(self._decode_prompt(processor, inputs), _get_prompt("en", "en", pnc=False, timestamps=True))

    def test_batch_broadcast_and_per_sample(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(
            audio=[self._audio(), self._audio()], source_language="en", target_language=["en", "es"]
        )
        self.assertEqual(len(inputs["decoder_input_ids"]), 2)
        self.assertEqual(self._decode_prompt(processor, inputs, 0), _get_prompt("en", "en"))
        self.assertEqual(self._decode_prompt(processor, inputs, 1), _get_prompt("en", "es"))

    def test_batch_length_mismatch_raises(self):
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor.apply_transcription_request(audio=[self._audio()], source_language=["en", "de"])

    def test_call_requires_audio(self):
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor(audio=None, text="text")

    def test_call_output_labels(self):
        processor = self.get_processor()
        outputs = processor(audio=self._audio(), text="hello world", output_labels=True)
        self.assertIn("input_features", outputs)
        self.assertIn("decoder_input_ids", outputs)
        self.assertIn("labels", outputs)
