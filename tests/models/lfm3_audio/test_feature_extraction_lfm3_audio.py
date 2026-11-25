# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import Lfm3AudioFeatureExtractor, is_torch_available
from transformers.testing_utils import require_torch

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


class Lfm3AudioFeatureExtractionTester:
    def __init__(self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, feature_size=16):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = 0.0
        self.sampling_rate = 16000

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "n_fft": 400,
            "hop_length": 160,
            "win_length": 400,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        # padding tests pad (num_frames, feature_size) sequences, as Whisper's tester
        rng = np.random.default_rng(0)
        if equal_length:
            lengths = [self.max_seq_length] * self.batch_size
        else:
            lengths = list(range(self.min_seq_length, self.max_seq_length, self.seq_length_diff))
        inputs = [rng.standard_normal((length, self.feature_size)).tolist() for length in lengths]
        return [np.asarray(x) for x in inputs] if numpify else inputs


@require_torch
class Lfm3AudioFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Lfm3AudioFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Lfm3AudioFeatureExtractionTester(self)
        self.extractor = Lfm3AudioFeatureExtractor()

    def test_frame_counts_follow_the_streaming_transform(self):
        # 1 frame per 160 samples after the first 400-sample window; clips padded, nothing dropped
        lengths, frames = (100, 400, 401, 559, 560, 561, 44550), [1, 1, 2, 2, 2, 3, 277]
        clips = [np.zeros(length, dtype=np.float32) for length in lengths]
        features = self.extractor(clips, sampling_rate=16000, return_tensors="pt")
        self.assertEqual(features["input_features_mask"].sum(-1).tolist(), frames)
        self.assertEqual(features["input_features"].shape, (len(lengths), max(frames), 128))

    def test_features_shape_and_normalization(self):
        rng = np.random.default_rng(0)
        clips = [
            rng.standard_normal(16000).astype(np.float32) * 0.1,
            rng.standard_normal(8000).astype(np.float32) * 0.1,
        ]
        features = self.extractor(clips, sampling_rate=16000, return_tensors="pt")
        self.assertEqual(features["input_features"].shape, (2, 99, 128))
        self.assertEqual(features["input_features_mask"].sum(-1).tolist(), [99, 49])
        # the padding frames hold `padding_value`
        self.assertTrue((features["input_features"][1, 49:] == 0.0).all())
        # the batch gives each clip's own features
        single = self.extractor(clips[1], sampling_rate=16000, return_tensors="pt")["input_features"]
        torch.testing.assert_close(features["input_features"][1:, :49], single)
        # silence sits at the log floor, scaled by the affine normalization
        silence = self.extractor(np.zeros(1600, dtype=np.float32), sampling_rate=16000, return_tensors="pt")[
            "input_features"
        ]
        self.assertTrue(np.allclose(silence.numpy(), (-12.0 + 4.0) / 4.0))

    def test_rejects_other_sampling_rates(self):
        with self.assertRaises(ValueError):
            self.extractor(np.zeros(1000, dtype=np.float32), sampling_rate=24000)

    def test_live_chunks_match_offline(self):
        # live transcription loop: chunked windows give the whole clip's frames
        window = (96 - 1) * self.extractor.hop_length + self.extractor.n_fft
        hop = 96 * self.extractor.hop_length
        for length in (window, window + 1, window + 240, window + 241, 3 * hop + 5000, 3 * hop + 241):
            wave = torch.randn(length)
            expected = self.extractor(wave, sampling_rate=16000, return_tensors="pt")["input_features"]
            pieces = [
                self.extractor(wave[start : start + window], sampling_rate=16000, return_tensors="pt")[
                    "input_features"
                ]
                for start in range(0, length - (window - hop), hop)
            ]
            torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-6, rtol=2e-6)
