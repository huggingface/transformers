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
"""Tests for the ZONOS3-Mini speaker feature extractor."""

import json
import unittest
from pathlib import Path

import numpy as np

from transformers import Zonos3MiniFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_torch
@require_torchaudio
class Zonos3MiniFeatureExtractionTest(unittest.TestCase):
    def test_padding_and_mask(self):
        feature_extractor = Zonos3MiniFeatureExtractor()
        rng = np.random.default_rng(0)
        clips = [rng.standard_normal(44100).astype(np.float32) * 0.1, rng.standard_normal(22050).astype(np.float32)]
        inputs = feature_extractor(clips, sampling_rate=44100, return_tensors="pt")
        # 1 s and 0.5 s at 16 kHz: 98 and 48 frames of 25 ms every 10 ms
        self.assertEqual(inputs["input_features"].shape, (2, 98, 80))
        self.assertEqual(inputs["attention_mask"].sum(-1).tolist(), [98, 48])
        single = feature_extractor(clips[1], sampling_rate=44100, return_tensors="pt")
        torch.testing.assert_close(inputs["input_features"][1, :48], single["input_features"][0])

    def test_stereo_is_averaged(self):
        rng = np.random.default_rng(0)
        stereo = rng.standard_normal((2, 44100)).astype(np.float32) * 0.1
        feature_extractor = Zonos3MiniFeatureExtractor()
        torch.testing.assert_close(
            feature_extractor(stereo, sampling_rate=44100, return_tensors="pt")["input_features"],
            feature_extractor(stereo.mean(0), sampling_rate=44100, return_tensors="pt")["input_features"],
        )

    def test_rejects_other_sampling_rates_and_short_clips(self):
        feature_extractor = Zonos3MiniFeatureExtractor()
        with self.assertRaises(ValueError):
            feature_extractor(np.zeros(16000, dtype=np.float32), sampling_rate=16000)
        with self.assertRaises(ValueError):
            feature_extractor(np.zeros(1000, dtype=np.float32), sampling_rate=44100)

    @slow
    def test_integration(self):
        """Reproducer: zonos3_repro/reproducer.py `fbank` (the remote code of Zyphra/ZONOS3-Mini-hf@e39d30d)."""
        from ..zac.test_modeling_zac import load_reference_clip

        fixture = Path(__file__).parent.parent.parent / "fixtures/zonos3_mini/expected_speaker_features.json"
        with open(fixture, encoding="utf-8") as f:
            expected = json.load(f)
        inputs = Zonos3MiniFeatureExtractor()(load_reference_clip().numpy(), sampling_rate=44100, return_tensors="pt")
        self.assertEqual(inputs["input_features"].shape[1], expected["num_frames"])
        torch.testing.assert_close(
            inputs["input_features"][0, :50], torch.tensor(expected["input_features"]), atol=1e-3, rtol=1e-3
        )
