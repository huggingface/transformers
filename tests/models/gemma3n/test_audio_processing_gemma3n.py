import unittest

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class Gemma3nAudioProcessingTester:
    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class Gemma3nAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # Gemma3n's unfold-based STFT with HTK preemphasis and float32-window-on-float64-frames
    # multiplication produces a slightly larger cross-backend divergence than the strict
    # float32 noise floor — empirically up to ~1.7e-4 on batched inputs.
    parity_atol = 5e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = Gemma3nAudioProcessingTester()
        super().setUp()
