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
"""Tests of the NemotronAsrStreaming feature extractor options."""

import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_librosa, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import NemotronAsrStreamingFeatureExtractor


@require_torch
@require_librosa
class NemotronAsrStreamingFeatureExtractionTest(unittest.TestCase):
    def test_mel_filters_dtype_rounds_the_constants(self):
        """`mel_filters_dtype` rounds the filter bank and the window to that dtype when the features are computed."""
        reference = NemotronAsrStreamingFeatureExtractor(feature_size=128)
        rounded = NemotronAsrStreamingFeatureExtractor(feature_size=128, mel_filters_dtype="bfloat16")
        self.assertEqual(reference.mel_filters_dtype, "float32")
        self.assertEqual(rounded.mel_filters_dtype, "bfloat16")
        torch.testing.assert_close(rounded.mel_filters, reference.mel_filters, rtol=0, atol=0)  # stored unchanged

        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        features = rounded(audio, sampling_rate=16000, return_tensors="pt")["input_features"]
        self.assertEqual(features.dtype, torch.float32)
        self.assertFalse(
            torch.equal(features, reference(audio, sampling_rate=16000, return_tensors="pt")["input_features"])
        )
        # the float32 pipeline with the constants rounded by hand gives the same features
        reference.mel_filters = reference.mel_filters.to(torch.bfloat16).to(torch.float32)
        reference.mel_filters_dtype = "bfloat16"  # also rounds the window
        torch.testing.assert_close(
            features, reference(audio, sampling_rate=16000, return_tensors="pt")["input_features"], rtol=0, atol=0
        )

    def test_mel_filters_dtype_round_trip(self):
        extractor = NemotronAsrStreamingFeatureExtractor(mel_filters_dtype="bfloat16")
        with tempfile.TemporaryDirectory() as tmpdirname:
            extractor.save_pretrained(tmpdirname)
            reloaded = NemotronAsrStreamingFeatureExtractor.from_pretrained(tmpdirname)
        self.assertEqual(reloaded.mel_filters_dtype, "bfloat16")

    def test_mel_filters_dtype_must_be_a_torch_dtype(self):
        with self.assertRaises(ValueError):
            NemotronAsrStreamingFeatureExtractor(mel_filters_dtype="int8x")
