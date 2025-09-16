# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from transformers import (
    AudioFlamingo3Processor,
    AutoProcessor,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_torch, require_torchaudio

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_torchaudio
class AudioFlamingo3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AudioFlamingo3Processor

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "lashahub/audio-flamingo-3"
        cls.tmpdirname = tempfile.mkdtemp()

        processor = AudioFlamingo3Processor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_can_load_various_tokenizers(self):
        processor = AudioFlamingo3Processor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = AudioFlamingo3Processor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = AudioFlamingo3Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = AudioFlamingo3Processor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)

    def test_tokenizer_integration(self):
        # Always test fast tokenizer
        fast_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertTrue(getattr(fast_tokenizer, "is_fast", False))

        prompt = (
            "<|im_start|>system\nAnswer the questions.<|im_end|>"
            "<|im_start|>user\nWhat is it?<|im_end|>"
            "<|im_start|>assistant\n"
        )
        EXPECTED_OUTPUT = [
            "<|im_start|>",
            "system",
            "Ċ",
            "Answer",
            "Ġthe",
            "Ġquestions",
            ".",
            "<|im_end|>",
            "<|im_start|>",
            "user",
            "Ċ",
            "What",
            "Ġis",
            "Ġit",
            "?",
            "<|im_end|>",
            "<|im_start|>",
            "assistant",
            "Ċ",
        ]

        # Fast tokenizer should produce the expected pieces
        self.assertEqual(fast_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)

        # If a slow tokenizer exists, also verify parity with a fast tokenizer
        # instantiated from the slow files.
        try:
            slow_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
        except Exception:
            slow_tokenizer = None

        if slow_tokenizer is not None:
            fast_from_slow = AutoTokenizer.from_pretrained(self.checkpoint, from_slow=True, legacy=False)
            self.assertEqual(slow_tokenizer.tokenize(prompt), fast_from_slow.tokenize(prompt))

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nSay hello.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ]

        formatted = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted)
