import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class PeAudioAudioProcessingTester:
    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class PeAudioAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = PeAudioAudioProcessingTester()
        super().setUp()
