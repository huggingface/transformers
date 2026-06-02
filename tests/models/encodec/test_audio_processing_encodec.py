# Copyright 2026 HuggingFace Inc.
# Licensed under the Apache License, Version 2.0
import unittest

from transformers.testing_utils import require_torch
from ...test_audio_processing_common import AudioProcessingTestMixin


class EncodecAudioProcessingTester:
    sample_rate = 24000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class EncodecAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = EncodecAudioProcessingTester()
        super().setUp()
