# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `CohereAsrAudioProcessor` and `CohereAsrAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class CohereAsrAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the CohereAsr audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        # Disable dither for the parity test: the legacy FE seeds torch RNG by valid
        # sample count, which we cannot reproduce bit-exactly with numpy's MT19937. Both
        # backends implement deterministic dither in production; only this fixture turns
        # it off.
        return {"dither": 0.0}


@require_torch
class CohereAsrAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # CohereAsr has a longer numerical chain (waveform-level preemphasis with masking, then
    # log(x + 2^-24), then per-utterance mean/var on the padded batch). The float32 noise
    # floor still holds for the unbatched test; the batched test relaxes slightly.
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = CohereAsrAudioProcessingTester()
        super().setUp()
