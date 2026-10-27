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
"""Testing suite for the LFM3-Audio feature extractor."""

import unittest

import numpy as np

from transformers import Lfm3AudioFeatureExtractor, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, torch_device

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
        # the padding tests pad feature sequences of shape (num_frames, feature_size), as the Whisper tester does
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

    def test_empty_clip_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one clip"):
            self.extractor([], sampling_rate=16000)

    def test_unsupported_options_are_refused(self):
        clip = np.zeros(1600, dtype=np.float32)
        for kwargs in ({"truncation": True, "max_length": 50}, {"padding": "max_length"}, {"sampling_rte": 8000}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(TypeError, "does not implement"):
                self.extractor(clip, sampling_rate=16000, **kwargs)

    @require_torch_accelerator
    def test_accelerator_tensor_clips_match_numpy(self):
        wave = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
        expected = self.extractor(wave, sampling_rate=16000)["input_features"]
        actual = self.extractor(torch.as_tensor(wave, device=torch_device), sampling_rate=16000)["input_features"]
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
        batched = self.extractor([torch.as_tensor(wave, device=torch_device)] * 2, sampling_rate=16000)
        self.assertEqual(np.asarray(batched["input_features"]).shape, (2, 128, expected.shape[-1]))

    def test_frame_counts_follow_the_streaming_transform(self):
        # one frame per 160 samples once the first 400-sample window is filled, clips padded so nothing is dropped
        for num_samples, frames in ((100, 1), (400, 1), (401, 2), (559, 2), (560, 2), (561, 3), (44550, 277)):
            self.assertEqual(self.extractor.num_frames(num_samples), frames, num_samples)
        self.assertEqual(self.extractor.num_encoder_tokens(277), 35)

    def test_features_shape_and_normalization(self):
        rng = np.random.default_rng(0)
        clips = [
            rng.standard_normal(16000).astype(np.float32) * 0.1,
            rng.standard_normal(8000).astype(np.float32) * 0.1,
        ]
        features = self.extractor(clips, sampling_rate=16000, return_tensors="pt")
        self.assertEqual(features["input_features"].shape, (2, 128, self.extractor.num_frames(16000)))
        self.assertEqual(
            features["feature_attention_mask"].sum(-1).tolist(),
            [self.extractor.num_frames(16000), self.extractor.num_frames(8000)],
        )
        # silence sits at the log floor, scaled by the affine normalization
        silence = self.extractor(np.zeros(1600, dtype=np.float32), sampling_rate=16000, return_tensors="pt")[
            "input_features"
        ]
        self.assertTrue(np.allclose(silence.numpy(), (-12.0 + 4.0) / 4.0))

    def test_rejects_other_sampling_rates(self):
        with self.assertRaises(ValueError):
            self.extractor(np.zeros(1000, dtype=np.float32), sampling_rate=24000)

    def test_streaming_failure_leaves_the_cache_unchanged(self):
        # the cache changes only once a chunk's features exist, so the chunk can be sent again
        wave = np.random.default_rng(0).standard_normal(2000).astype(np.float32)
        _, cache = self.extractor.stream(wave[:500])
        before = (cache.num_samples, cache.num_frames, cache.samples.clone())
        log_mel, self.extractor._log_mel = self.extractor._log_mel, lambda samples: 1 / 0
        try:
            with self.assertRaises(ZeroDivisionError):
                self.extractor.stream(wave[500:1500], cache)
        finally:
            self.extractor._log_mel = log_mel
        self.assertEqual((cache.num_samples, cache.num_frames), before[:2])
        torch.testing.assert_close(cache.samples, before[2])
        features, cache = self.extractor.stream(wave[500:1500], cache)
        expected = self.extractor(wave[:1500], sampling_rate=16000, return_tensors="pt")["input_features"][0]
        torch.testing.assert_close(features["input_features"][0], expected[:, before[1] : cache.num_frames])

    def test_streaming_matches_offline_for_arbitrary_chunks_and_final_flush(self):
        for length in (0, 1, 399, 400, 401, 560, 16000, 16321):
            wave = torch.randn(length)
            expected = self.extractor(wave, sampling_rate=16000, return_tensors="pt")["input_features"]
            for chunk_size in (157, 1280):
                state, pieces = None, []
                for start in range(0, length, chunk_size):
                    features, state = self.extractor.stream(
                        wave[start : start + chunk_size], state, sampling_rate=16000
                    )
                    pieces.append(features["input_features"])
                    self.assertLess(state.samples.numel(), self.extractor.n_fft)
                features, state = self.extractor.stream(wave[:0], state, is_final=True, sampling_rate=16000)
                pieces.append(features["input_features"])
                torch.testing.assert_close(torch.cat(pieces, dim=-1), expected, atol=2e-6, rtol=2e-6)
                self.assertIsNone(state.samples)
                with self.assertRaises(ValueError):
                    self.extractor.stream(wave, state)
