# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `SpeechToTextAudioProcessor` and `SpeechToTextAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class SpeechToTextAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the SpeechToText audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class SpeechToTextAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # Per-waveform kaldi fbank + per-utterance CMVN (with numpy ddof=0 / torch unbiased=False
    # matched) introduces small numerical drift above the strict 1e-5 floor — empirically
    # up to ~3e-5 on batched inputs.
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = SpeechToTextAudioProcessingTester()
        super().setUp()
