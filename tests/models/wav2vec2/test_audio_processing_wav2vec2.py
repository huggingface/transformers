# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `Wav2Vec2AudioProcessor` and `Wav2Vec2AudioProcessorNumpy`."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from transformers.audio_processing_utils import logger as audio_processing_logger
from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class Wav2Vec2AudioProcessingTester:
    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class Wav2Vec2AudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = Wav2Vec2AudioProcessingTester()
        super().setUp()

    def test_array_sampling_rate_warning(self):
        waveform = np.zeros(160, dtype=np.float32)

        for backend, processor_class in self.audio_processing_classes.items():
            with self.subTest(backend=backend):
                processor = processor_class()
                with patch.object(audio_processing_logger, "warning_once") as warning_once:
                    processor(waveform, return_tensors="np")
                warning_once.assert_called_once()
                self.assertIn("`sampling_rate` was not provided", warning_once.call_args.args[0])

                with patch.object(audio_processing_logger, "warning_once") as warning_once:
                    processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="np")
                warning_once.assert_not_called()

    def test_unpadded_unequal_batch_has_clear_error(self):
        waveforms = [np.zeros(160, dtype=np.float32), np.zeros(80, dtype=np.float32)]

        for backend, processor_class in self.audio_processing_classes.items():
            for padding in (False, "do_not_pad"):
                with self.subTest(backend=backend, padding=padding):
                    processor = processor_class()
                    with self.assertRaisesRegex(ValueError, "different shapes.*padding is disabled"):
                        processor(waveforms, sampling_rate=processor.sampling_rate, padding=padding)
