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

from transformers import LiquidCodecFeatureExtractor, is_torch_available
from transformers.testing_utils import require_torch

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


class LiquidCodecFeatureExtractionTester:
    def __init__(self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, n_fft=30):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.n_fft = n_fft
        self.feature_size = 3 * (n_fft // 2 + 1)
        self.padding_value = 0.0
        self.sampling_rate = 24000

    def prepare_feat_extract_dict(self):
        return {
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "n_fft": self.n_fft,
            "hop_length": 10,
            "win_length": self.n_fft,
            "frame_stack": 6,
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
class LiquidCodecFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = LiquidCodecFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = LiquidCodecFeatureExtractionTester(self)
        self.extractor = LiquidCodecFeatureExtractor()

    def test_whole_codec_frames(self):
        # clips padded to whole codec frames: 1920 samples = 6 hops of 320
        lengths = (1, 1920, 1921, 24000)
        clips = [np.zeros(length, dtype=np.float32) for length in lengths]

        features = self.extractor(clips, sampling_rate=24000, return_tensors="pt")
        self.assertEqual(features["input_features_mask"].sum(-1).tolist(), [6, 6, 12, 78])
        self.assertEqual(features["input_features"].shape, (4, 1923, 78))

    def test_batch_matches_single(self):
        rng = np.random.default_rng(0)
        clips = [rng.standard_normal(24000).astype(np.float32), rng.standard_normal(5000).astype(np.float32)]
        features = self.extractor(clips, sampling_rate=24000, return_tensors="pt")

        for index, clip in enumerate(clips):
            single = self.extractor(clip, sampling_rate=24000, return_tensors="pt")["input_features"][0]
            torch.testing.assert_close(features["input_features"][index, :, : single.shape[-1]], single)
        self.assertTrue((features["input_features"][1, :, 18:] == 0.0).all())

    def test_padding_and_truncation(self):
        clips = [np.zeros(length, dtype=np.float32) for length in (1000, 5000)]

        # max_length in samples, rounded up to whole codec frames: 4000 -> 3 x 1920 -> 18 hops
        features = self.extractor(clips, sampling_rate=24000, padding="max_length", max_length=4000, truncation=True)
        self.assertEqual(np.asarray(features["input_features"]).shape, (2, 1923, 18))
        self.assertEqual(np.asarray(features["input_features_mask"]).sum(-1).tolist(), [6, 18])

        features = self.extractor(clips, sampling_rate=24000, padding="max_length", max_length=24000)
        self.assertEqual(np.asarray(features["input_features"]).shape, (2, 1923, 78))
        self.assertEqual(np.asarray(features["input_features_mask"]).sum(-1).tolist(), [6, 18])

        with self.assertRaisesRegex(ValueError, "without padding"):
            self.extractor(clips, sampling_rate=24000, padding=False)
        with self.assertRaisesRegex(ValueError, "max_length"):
            self.extractor(clips, sampling_rate=24000, truncation=True)

    def test_rejects_other_sampling_rates(self):
        with self.assertRaises(ValueError):
            self.extractor(np.zeros(1000, dtype=np.float32), sampling_rate=16000)
