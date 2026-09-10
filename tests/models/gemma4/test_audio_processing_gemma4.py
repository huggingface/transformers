# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `Gemma4AudioProcessor` and `Gemma4AudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class Gemma4AudioProcessingTester:
    """Provides init kwargs and fixture parameters for the Gemma4 audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        # Dither uses `np.random.randn` in the legacy FE — not seeded for cross-backend
        # parity. Disable for the parity test (both backends still implement dither for
        # production). HTK preemphasis is off by default, so the default flow is the
        # interesting bit-exact path.
        return {"dither": 0.0}


@require_torch
class Gemma4AudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # Like Gemma3n, the unfold-based STFT with float32-window-on-float64-frame multiplication
    # produces a slightly larger cross-backend divergence than the strict float32 noise floor.
    parity_atol = 5e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = Gemma4AudioProcessingTester()
        super().setUp()
