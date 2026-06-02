# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `ClvpAudioProcessor` and `ClvpAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class ClvpAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the CLVP audio processor tests."""

    sample_rate = 22050

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class ClvpAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # CLVP's float64 log + per-mel-norm division before float32 cast accumulates a small
    # cross-backend drift above the strict 1e-5 floor — empirically up to ~4e-5.
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = ClvpAudioProcessingTester()
        super().setUp()
