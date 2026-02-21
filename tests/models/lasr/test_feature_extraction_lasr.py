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
"""Tests for the LASR feature extractor."""

import unittest

import numpy as np

from transformers import LasrFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_torch
class LasrFeatureExtractionTest(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = LasrFeatureExtractor()
        self.sampling_rate = self.feature_extractor.sampling_rate

    def test_call_single_audio(self):
        input_audio = np.random.randn(self.sampling_rate).astype(np.float32)

        features = self.feature_extractor(
            input_audio,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )

        expected_num_frames = (
            (len(input_audio) - self.feature_extractor.win_length) // self.feature_extractor.hop_length
        ) + 1

        self.assertEqual(
            features.input_features.shape,
            (1, expected_num_frames, self.feature_extractor.feature_size),
        )
        self.assertEqual(features.attention_mask.shape, (1, expected_num_frames))
        self.assertEqual(features.attention_mask.sum().item(), expected_num_frames)

    def test_center_kwarg_is_ignored(self):
        input_audio = np.random.randn(self.sampling_rate).astype(np.float32)

        input_features = self.feature_extractor(
            input_audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features
        input_features_with_center = self.feature_extractor(
            input_audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            center=False,
        ).input_features

        torch.testing.assert_close(input_features, input_features_with_center)
