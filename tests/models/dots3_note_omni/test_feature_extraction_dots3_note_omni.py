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
import tempfile
import unittest

import numpy as np

from transformers import Dots3NoteOmniFeatureExtractor, is_torch_available
from transformers.models.dots3_note_omni.feature_extraction_dots3_note_omni import (
    compute_audio_token_length,
)
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch


@require_torch
class Dots3NoteOmniFeatureExtractorTest(unittest.TestCase):
    def get_feature_extractor(self):
        return Dots3NoteOmniFeatureExtractor(
            feature_size=8,
            sampling_rate=32,
            n_fft=16,
            hop_length=4,
            chunk_seconds=2,
            conv_temporal_stride=8,
        )

    def test_audio_token_length_boundaries(self):
        cases = [(0, 0), (1, 1), (31, 1), (32, 1), (33, 2), (64, 2), (65, 3)]
        for num_samples, expected_tokens in cases:
            with self.subTest(num_samples=num_samples):
                self.assertEqual(
                    compute_audio_token_length(num_samples, chunk_samples=64, token_stride=32),
                    expected_tokens,
                )

    def test_first_channel_chunking_and_lengths(self):
        extractor = self.get_feature_extractor()
        channel_zero = torch.linspace(-0.25, 0.25, 65)
        channel_one = torch.ones_like(channel_zero)

        stereo_output = extractor(torch.stack((channel_zero, channel_one)), sampling_rate=32)
        mono_output = extractor(channel_zero, sampling_rate=32)

        self.assertEqual(stereo_output.input_features.shape, (2, 8, 16))
        self.assertEqual(stereo_output.chunk_sample_lengths.tolist(), [64, 1])
        self.assertEqual(stereo_output.chunk_token_lengths.tolist(), [2, 1])
        self.assertEqual(stereo_output.audio_token_lengths.tolist(), [3])
        self.assertEqual(stereo_output.audio_chunk_counts.tolist(), [2])
        torch.testing.assert_close(stereo_output.input_features, mono_output.input_features)

    def test_batch_tracks_chunk_ownership(self):
        extractor = self.get_feature_extractor()
        output = extractor([torch.zeros(64), torch.zeros(65)], sampling_rate=32)

        self.assertEqual(output.audio_token_lengths.tolist(), [2, 3])
        self.assertEqual(output.audio_chunk_counts.tolist(), [1, 2])
        self.assertEqual(output.chunk_audio_indices.tolist(), [0, 1, 1])

    def test_save_and_reload(self):
        extractor = self.get_feature_extractor()
        with tempfile.TemporaryDirectory() as tmpdirname:
            extractor.save_pretrained(tmpdirname)
            reloaded = Dots3NoteOmniFeatureExtractor.from_pretrained(tmpdirname)

        self.assertEqual(extractor.to_dict(), reloaded.to_dict())
        self.assertTrue(np.array_equal(extractor.mel_filters, reloaded.mel_filters))

    def test_rejects_wrong_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "resample"):
            self.get_feature_extractor()(torch.zeros(64), sampling_rate=16)


if __name__ == "__main__":
    unittest.main()
