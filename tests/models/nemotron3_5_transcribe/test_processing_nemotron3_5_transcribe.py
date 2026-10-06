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

import numpy as np

from transformers import AutoProcessor, Nemotron3_5TranscribeProcessor
from transformers.testing_utils import require_torch


@require_torch
class Nemotron3_5TranscribeProcessorTest(unittest.TestCase):
    checkpoint = "nvidia/Nemotron-3.5-Transcribe"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        Nemotron3_5TranscribeProcessor.from_pretrained(cls.checkpoint).save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.tmpdirname)

    def test_transcription_request_prompt(self):
        """Same prompt as the original `nemotron3p5` prompt formatter: empty system turn and no thinking."""
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(np.zeros(16000, dtype=np.float32), prompt="Transcribe.")
        prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        audio_tokens = processor.audio_token * 13
        expected = (
            "<|im_start|>system\n<|im_end|>\n"
            f"<|im_start|>user\nTranscribe. {audio_tokens}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
        self.assertEqual(prompt, expected)

        inputs = processor.apply_transcription_request(np.zeros(16000, dtype=np.float32))
        prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        expected = expected.replace("Transcribe.", processor.default_transcription_prompt)
        self.assertEqual(prompt, expected)

    def test_num_audio_tokens(self):
        """One audio token per 8 spectrogram frames of 160 samples, the last group may be incomplete."""
        processor = self.get_processor()
        for num_samples, num_tokens in [(1280 * 10, 10), (1280 * 10 + 159, 10), (1280 * 10 + 160, 11)]:
            inputs = processor.apply_transcription_request(np.zeros(num_samples, dtype=np.float32))
            self.assertEqual((inputs["input_ids"][0] == processor.audio_token_id).sum().item(), num_tokens)
            self.assertEqual(inputs["input_features_mask"][0].sum().item(), num_samples // 160)

    def test_batch_is_left_padded(self):
        processor = self.get_processor()
        audio = [np.zeros(16000, dtype=np.float32), np.zeros(32000, dtype=np.float32)]
        inputs = processor.apply_transcription_request(audio)
        self.assertEqual(inputs["attention_mask"][0, 0].item(), 0)
        self.assertEqual(inputs["attention_mask"][0, -1].item(), 1)
        self.assertEqual(inputs["input_features"].shape[0], 2)
        self.assertEqual(inputs["input_features_mask"].sum(-1).tolist(), [100, 200])

    def test_thinking_can_be_enabled(self):
        processor = self.get_processor()
        conversation = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, enable_thinking=True)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))
