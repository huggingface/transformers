# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `SeamlessM4tAudioProcessor` and `SeamlessM4tAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

import numpy as np

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class SeamlessM4tAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the SeamlessM4t audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class SeamlessM4tAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # SeamlessM4t's `power=2.0` magnitudes (computed in float64), kaldi-exact mel filters built
    # in float32 (cast at the matmul site), and `ddof=1` per-utterance variance normalization
    # amplify the underlying float32 STFT noise. Empirically the final audio_features drift up
    # to ~1.2e-4 on batched inputs.
    parity_atol = 5e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = SeamlessM4tAudioProcessingTester()
        super().setUp()

    def test_spectrogram_preserves_batch_dimension(self):
        audio = np.zeros((2, 1_600), dtype=np.float32)
        for backend, processor_class in self.audio_processing_classes.items():
            with self.subTest(backend=backend):
                processor = processor_class()
                backend_audio = self._to_torch(audio) if processor.backend == "torch" else audio
                features = processor.spectrogram(
                    backend_audio,
                    spectrogram_config=processor.spectrogram_config,
                    dither=0.0,
                )
                self.assertEqual(features.shape[0], 2)
                self.assertEqual(features.shape[-1], 80)
