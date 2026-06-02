# Copyright 2026 HuggingFace Inc.
# Licensed under the Apache License, Version 2.0
import unittest

from transformers.testing_utils import require_torch
from ...test_audio_processing_common import AudioProcessingTestMixin


class DacAudioProcessingTester:
    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class DacAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = DacAudioProcessingTester()
        super().setUp()
