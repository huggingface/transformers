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
"""Tests for the ZAC-1 feature extractor."""

import tempfile
import unittest

import numpy as np

from transformers import SeamlessM4TFeatureExtractor, ZacFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.utils import is_torch_available, is_torchaudio_available


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


SAMPLING_RATE = 44100
HOP = 2940


def tone(seconds, amplitude=0.5, channels=1):
    t = np.arange(int(seconds * SAMPLING_RATE)) / SAMPLING_RATE
    return np.repeat((amplitude * np.sin(2 * np.pi * 220 * t))[None], channels, 0).astype(np.float32)


@require_torch
@require_torchaudio
class ZacFeatureExtractionTest(unittest.TestCase):
    def test_shapes_and_padding_mask(self):
        feature_extractor = ZacFeatureExtractor()
        clips = [tone(1.0)[0], tone(0.6)[0]]
        inputs = feature_extractor(clips, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        frames = [int(np.ceil(len(clip) / HOP)) for clip in clips]
        self.assertEqual(inputs["input_values"].shape, (2, 1, max(frames) * HOP))
        self.assertEqual(inputs["padding_mask"].sum(-1).tolist(), [len(clip) for clip in clips])
        # one lane each: 300 frames of 2940 samples at 16 kHz -> 1999 fbank frames -> 999 stacked frames
        self.assertEqual(inputs["input_features"].shape, (2, 1, 999, 160))
        self.assertEqual(inputs["input_features_mask"].shape, (2, 1, 999))

    def test_lanes(self):
        feature_extractor = ZacFeatureExtractor(lane_frames=4, lane_context_frames=2)
        inputs = feature_extractor([tone(1.0)[0], tone(0.6)[0]], sampling_rate=SAMPLING_RATE, return_tensors="pt")
        # 15 frames: [0, 4) then 2 new frames per lane -> 7 lanes; 9 frames -> 4 lanes (the rest is padding)
        self.assertEqual(inputs["input_features"].shape[1], 7)
        self.assertEqual(inputs["input_features_mask"][1, 4:].sum().item(), 0)
        self.assertGreater(inputs["input_features_mask"][1, :4].sum().item(), 0)

    def test_rejects_other_sampling_rates(self):
        with self.assertRaises(ValueError):
            ZacFeatureExtractor()(tone(1.0)[0], sampling_rate=16000)

    def test_rejects_short_clips(self):
        with self.assertRaises(ValueError):
            ZacFeatureExtractor()(tone(0.4)[0], sampling_rate=SAMPLING_RATE)

    def test_peak_normalized_only_above_one(self):
        """Ported from ZAC-1 `tests/test_prepare.py::test_peak_normalized_only_above_one`."""
        feature_extractor = ZacFeatureExtractor()
        quiet = tone(1.0, amplitude=0.9)[0]
        values = feature_extractor(quiet, sampling_rate=SAMPLING_RATE, return_tensors="pt")["input_values"][0, 0]
        torch.testing.assert_close(values[: len(quiet)], torch.from_numpy(quiet))
        loud = tone(1.0, amplitude=2.0)[0]
        values = feature_extractor(loud, sampling_rate=SAMPLING_RATE, return_tensors="pt")["input_values"][0, 0]
        self.assertAlmostEqual(float(values.abs().amax()), 1.0, places=6)
        torch.testing.assert_close(values[: len(loud)], torch.from_numpy(loud / np.abs(loud).max()))

    def test_stereo_is_averaged(self):
        stereo = np.concatenate([tone(1.0, amplitude=0.2), tone(1.0, amplitude=0.6)])
        values = ZacFeatureExtractor()(stereo, sampling_rate=SAMPLING_RATE, return_tensors="pt")["input_values"]
        torch.testing.assert_close(values[0, 0, : stereo.shape[-1]], torch.from_numpy(stereo.mean(0)))

    def test_semantic_features_match_reference_preprocessing(self):
        """The teacher features equal ZAC-1's `Wav2Vec2BertFeaturePreprocessor(44100)` (a `torchaudio` resampler to
        16 kHz, then `SeamlessM4TFeatureExtractor`) on the zero-padded lane."""
        audio = np.random.default_rng(0).standard_normal(int(1.3 * SAMPLING_RATE)).astype(np.float32) * 0.1
        inputs = ZacFeatureExtractor()(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")

        lane = torch.nn.functional.pad(torch.from_numpy(audio), (0, 300 * HOP - len(audio)))
        resampled = torchaudio.transforms.Resample(SAMPLING_RATE, 16000)(lane)
        reference = SeamlessM4TFeatureExtractor(
            feature_size=80, sampling_rate=16000, num_mel_bins=80, padding_value=1, stride=2
        )(resampled.numpy(), sampling_rate=16000, return_tensors="pt")
        torch.testing.assert_close(inputs["input_features"][0, 0], reference["input_features"][0])
        torch.testing.assert_close(inputs["input_features_mask"][0, 0], reference["attention_mask"][0].long())

    def test_save_load(self):
        feature_extractor = ZacFeatureExtractor(lane_frames=40, lane_context_frames=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_extractor.save_pretrained(tmpdir)
            loaded = ZacFeatureExtractor.from_pretrained(tmpdir)
        self.assertEqual(loaded.to_dict(), feature_extractor.to_dict())
        self.assertNotIn("semantic_feature_extractor", loaded.to_dict())
