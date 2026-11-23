# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `AudioSpectrogramTransformerAudioProcessor` on the numpy and torch backends."""

from __future__ import annotations

import unittest

from transformers.models.auto.feature_extraction_auto import (
    FEATURE_EXTRACTOR_MAPPING_NAMES,
    feature_extractor_class_from_name,
)
from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class AudioSpectrogramTransformerAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the AST audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class AudioSpectrogramTransformerAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # The numpy sibling runs the native kaldi-style pipeline rather than bridging to
    # `torchaudio.compliance.kaldi`, so it no longer matches torch bit-for-bit: numpy's FFT and
    # window differ in the last float32 ulp and `log` amplifies that in near-silent mel bins
    # (empirically up to ~2.4e-4 on the quiet fixture audio, after AudioSet normalization).
    parity_atol = 5e-4
    parity_rtol = 1e-4

    def setUp(self):
        # AST is registered under `audio-spectrogram-transformer` (hyphenated) but the test
        # directory uses underscores, so the mixin's auto-discovery cannot match.
        self.audio_processor_tester = AudioSpectrogramTransformerAudioProcessingTester()
        self.audio_processing_class = feature_extractor_class_from_name(
            FEATURE_EXTRACTOR_MAPPING_NAMES["audio-spectrogram-transformer"]
        )
        self.audio_backends = ["numpy", "torch"]
