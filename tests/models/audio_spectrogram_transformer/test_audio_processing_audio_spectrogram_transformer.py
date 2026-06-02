# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `AudioSpectrogramTransformerAudioProcessor` and its NumPy sibling."""

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
    def setUp(self):
        # AST is registered under `audio-spectrogram-transformer` (hyphenated) but the test
        # directory uses underscores, so the mixin's auto-discovery cannot match.
        self.audio_processor_tester = AudioSpectrogramTransformerAudioProcessingTester()
        class_names_by_backend = FEATURE_EXTRACTOR_MAPPING_NAMES["audio-spectrogram-transformer"]
        self.audio_processing_classes = {
            backend: feature_extractor_class_from_name(class_name)
            for backend, class_name in class_names_by_backend.items()
            if class_name not in self.test_classes_to_skip
        }
        self.audio_processing_classes = {b: c for b, c in self.audio_processing_classes.items() if c is not None}
