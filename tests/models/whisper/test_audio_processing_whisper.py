# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `WhisperAudioProcessor` and `WhisperAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class WhisperAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the Whisper audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class WhisperAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = WhisperAudioProcessingTester()
        super().setUp()
