# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `WhisperAudioProcessor` and `WhisperAudioProcessorNumpy`."""

from __future__ import annotations

import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class WhisperAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the Whisper audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class WhisperAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = WhisperAudioProcessingTester()
        super().setUp()

    def test_nested_config_defaults_and_legacy_loading(self):
        from transformers.audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig

        for processor_class in self.audio_processing_classes.values():
            with self.subTest(processor_class=processor_class):
                self.assertIsInstance(processor_class.spectrogram_config, dict)
                processor = processor_class()
                config = processor.spectrogram_config
                self.assertIsInstance(config, SpectrogramConfig)
                self.assertIsInstance(config.stft_config, StftConfig)
                self.assertIsInstance(config.mel_scale_config, MelScaleConfig)
                loaded = processor_class.from_dict({"hop_length": 200, "feature_size": 64})
                self.assertEqual(loaded.spectrogram_config.stft_config.hop_length, 200)
                self.assertEqual(loaded.spectrogram_config.mel_scale_config.n_mels, 64)
                self.assertEqual(loaded.spectrogram_config.mel_scale_config.norm, "slaney")
                self.assertEqual(loaded.spectrogram_config.post_log_scale, 0.25)
                self.assertEqual(processor_class().spectrogram_config, config)
                self.assertEqual(processor_class.from_dict(processor.to_dict()).spectrogram_config, config)

                partial = processor_class.from_dict({"spectrogram_config": {"stft_config": {"hop_length": 200}}})
                self.assertEqual(
                    partial.spectrogram_config,
                    loaded.spectrogram_config | {"mel_scale_config": config.mel_scale_config},
                )
                no_mel = processor_class.from_dict({"spectrogram_config": {"mel_scale_config": None}})
                self.assertIsNone(no_mel.spectrogram_config.mel_scale_config)
