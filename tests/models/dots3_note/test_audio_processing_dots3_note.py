# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np

from transformers import Dots3NoteAudioProcessor
from transformers.testing_utils import Expectations, require_torch, torch_device

from ...test_audio_processing_common import AudioProcessingTestMixin


class Dots3NoteAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the Dots3Note audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class Dots3NoteAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # numpy's float64 FFT and window bits against torch's: a few ulp in low-energy log bins, magnified by log10
    # (the same gap the parity suite budgets against the legacy extractor; measured 1.9e-5)
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = Dots3NoteAudioProcessingTester()
        super().setUp()

    def get_small_processor(self, backend):
        # 2 s chunks of 64 samples, 4-sample hop: one audio token per 32 samples
        return Dots3NoteAudioProcessor.from_dict(
            {"sampling_rate": 32, "feature_size": 8, "n_fft": 16, "hop_length": 4, "chunk_length": 2},
            backend=backend,
        )

    def test_audio_token_length_boundaries(self):
        for backend in self.audio_backends:
            processor = self.get_small_processor(backend)
            for num_samples, expected_tokens in [(1, 1), (31, 1), (32, 1), (33, 2), (64, 2), (65, 3)]:
                with self.subTest(backend=backend, num_samples=num_samples):
                    output = processor(np.zeros(num_samples, dtype=np.float32), sampling_rate=32, return_tensors="np")
                    self.assertEqual(output["num_audio_tokens"].tolist(), [expected_tokens])

    def test_chunking_and_lengths(self):
        for backend in self.audio_backends:
            with self.subTest(backend=backend):
                processor = self.get_small_processor(backend)
                ramp = np.linspace(-0.25, 0.25, 65, dtype=np.float32)
                output = processor(ramp, sampling_rate=32, return_tensors="np")
                self.assertEqual(output["audio_features"].shape, (2, 8, 16))
                self.assertEqual(output["chunk_sample_lengths"].tolist(), [64, 1])
                self.assertEqual(output["num_audio_tokens"].tolist(), [3])

                short = processor(np.zeros(1, dtype=np.float32), sampling_rate=32, return_tensors="np")
                self.assertEqual(short["chunk_sample_lengths"].tolist(), [1])
                self.assertEqual(short["audio_features"].shape, (1, 8, 8))

                batch = processor(
                    [np.zeros(64, dtype=np.float32), np.zeros(65, dtype=np.float32)],
                    sampling_rate=32,
                    return_tensors="np",
                )
                self.assertEqual(batch["num_audio_tokens"].tolist(), [2, 3])
                self.assertEqual(batch["chunk_sample_lengths"].tolist(), [64, 64, 1])

    def test_per_call_chunk_length(self):
        processor = Dots3NoteAudioProcessor()
        output = processor(np.zeros(16000 * 50, dtype=np.float32), chunk_length=30, return_tensors="np")
        self.assertEqual(output["chunk_sample_lengths"].tolist(), [480000, 320000])

    def test_sglang_log_mel_numerical_golden(self):
        import torch

        processor = Dots3NoteAudioProcessor()
        waveform = ((torch.arange(960_001, dtype=torch.int32) % 257) - 128).to(torch.float32) / 128.0
        output = processor(waveform, sampling_rate=16_000, device=torch_device, return_tensors="pt")

        self.assertEqual(output["audio_features"].shape, (2, 128, 6000))
        positions = [(0, 0, 0), (0, 0, 1), (0, 63, 123), (0, 127, 5999), (1, 0, 0), (1, 64, 10)]
        selected = torch.stack([output["audio_features"][index] for index in positions])
        expected = Expectations(
            {
                ("cpu", None): [1.4883524179, 1.3922336102, 0.7106180191, 0.5591694713, 0.1503863335, -1.5],
                ("cuda", None): [1.4883524179, 1.3922336102, 0.7106180191, 0.5591694713, 0.1503863335, -1.5],
            }
        ).get_expectation()
        torch.testing.assert_close(selected, torch.tensor(expected, device=torch_device), rtol=1e-5, atol=1e-6)
