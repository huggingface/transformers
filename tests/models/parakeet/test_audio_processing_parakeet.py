# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `ParakeetAudioProcessor` and `ParakeetAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.models.auto.feature_extraction_auto import (
    FEATURE_EXTRACTOR_MAPPING_NAMES,
    feature_extractor_class_from_name,
)
from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class ParakeetAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the Parakeet audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class ParakeetAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # Parakeet's `power=2.0` magnitudes, librosa-compatible mel filters, log(x + mel_floor)
    # compression, and per-utterance mean/var normalization compound the underlying
    # `np.fft.rfft` vs `torch.fft.rfft` float32 noise (a single STFT bin already drifts ~4e-6).
    # Empirically the final audio_features drift up to ~6e-5 on batched inputs — within the
    # float32 noise floor but above the strict default bar.
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        # Parakeet is registered under `parakeet_ctc` / `parakeet_encoder` in the auto mapping
        # rather than `parakeet` (the test directory name), so the mixin's auto-discovery
        # cannot match. We override the lookup directly.
        self.audio_processor_tester = ParakeetAudioProcessingTester()
        class_names_by_backend = FEATURE_EXTRACTOR_MAPPING_NAMES["parakeet_ctc"]
        self.audio_processing_classes = {
            backend: feature_extractor_class_from_name(class_name)
            for backend, class_name in class_names_by_backend.items()
            if class_name not in self.test_classes_to_skip
        }
        self.audio_processing_classes = {b: c for b, c in self.audio_processing_classes.items() if c is not None}
