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

from ...test_audio_processing_common import AudioProcessingTestMixin


class MossTranscribeDiarizeAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the MOSS-Transcribe-Diarize audio processor tests."""

    sample_rate = 16000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


class MossTranscribeDiarizeAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.audio_processor_tester = MossTranscribeDiarizeAudioProcessingTester()
        super().setUp()

    def test_long_audio_is_split_into_windows(self):
        """Audio longer than the `chunk_length` window is split into consecutive windows (batch axis counts
        windows), and `padding_mask` records each clip's own raw length, one row per clip."""
        rng = np.random.default_rng(0)
        for backend in self.audio_backends:
            with self.subTest(backend=backend):
                # a 1 s window: 16000 samples, 100 frames
                processor = self.audio_processing_class(backend=backend, chunk_length=1)
                short_audio = rng.standard_normal(6000).astype(np.float32)
                long_audio = rng.standard_normal(36000).astype(np.float32)

                out = processor([short_audio, long_audio], sampling_rate=16000, return_tensors="np")
                # 1 window for the short clip + 3 for the long one, each padded to the full window
                self.assertEqual(out["audio_features"].shape, (4, 80, 100))
                self.assertEqual(out["audio_features_mask"].shape, (4, 100))
                self.assertEqual(out["audio_features_mask"].sum(-1).tolist(), [38, 100, 100, 25])
                self.assertEqual(out["padding_mask"].shape, (2, 36000))
                self.assertEqual(out["padding_mask"].sum(-1).tolist(), [6000, 36000])

                # each window is the Whisper recipe on that slice of audio
                last = processor([long_audio[32000:]], sampling_rate=16000, return_tensors="np")
                np.testing.assert_array_equal(out["audio_features"][3], last["audio_features"][0])

    def test_window_ignores_padding_options(self):
        """The window is model geometry: text-oriented `padding` / `max_length` passed through a processor call
        neither shorten nor truncate it."""
        processor = self.audio_processing_class()
        audio = [np.zeros(1600, dtype=np.float32), np.zeros(500_000, dtype=np.float32)]
        out = processor(audio, sampling_rate=16000, padding="longest", max_length=201, return_tensors="np")
        self.assertEqual(out["audio_features"].shape, (3, 80, 3000))
        self.assertEqual(out["padding_mask"].sum(-1).tolist(), [1600, 500_000])
