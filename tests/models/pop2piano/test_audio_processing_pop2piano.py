# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `Pop2PianoAudioProcessor` and `Pop2PianoAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class Pop2PianoAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the Pop2Piano audio processor tests."""

    sample_rate = 22050

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class Pop2PianoAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # Pop2Piano's `n_fft=4096`, `power=2.0` magnitudes, large `n_mels=512` HTK mel filterbank,
    # and `log10` compression amplify the `np.fft.rfft` vs `torch.fft.rfft` float32 drift.
    # Empirically up to ~7e-4 on batched inputs — within the float32 noise floor for a
    # spectrogram of this size but above the strict default bar.
    parity_atol = 1e-3
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = Pop2PianoAudioProcessingTester()
        super().setUp()
