# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `UnivNetAudioProcessor` and `UnivNetAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class UnivNetAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the UnivNet audio processor tests."""

    sample_rate = 24000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class UnivNetAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # UnivNet's reflect-padded STFT and float64 spectrogram path keep parity within the
    # float32 noise floor but slightly above the strict 1e-5 bar — empirically up to ~3e-5.
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = UnivNetAudioProcessingTester()
        super().setUp()
