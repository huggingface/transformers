# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `ClapAudioProcessor` and `ClapAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class ClapAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the CLAP audio processor tests."""

    sample_rate = 48000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class ClapAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # CLAP's full-batch padded mel + 48 kHz STFT accumulates a slightly larger cross-backend
    # divergence than the strict float32 noise floor — empirically up to ~8e-5 on batched inputs.
    parity_atol = 1e-4
    parity_rtol = 1e-4
    # `_pad_waveform` tiles short audio (`padding_mode="repeatpad"`) and only zero-pads the
    # remainder, so the region outside `ranges` is repeated signal rather than `padding_value`.
    pad_fills_with_padding_value = False

    def setUp(self):
        self.audio_processor_tester = ClapAudioProcessingTester()
        super().setUp()

    def test_fusion_initializes_nested_dict_config(self):
        for processor_class in self.audio_processing_classes.values():
            with self.subTest(processor_class=processor_class):
                processor = processor_class(truncation_mode="fusion")
                mel_config = processor.spectrogram_config.mel_scale_config
                self.assertEqual(mel_config.mel_scale, "htk")
                self.assertIsNone(mel_config.norm)
                self.assertEqual(processor.mel_filters.shape[-1], mel_config.n_mels)
                self.assertEqual(processor_class().spectrogram_config.mel_scale_config.norm, "slaney")
