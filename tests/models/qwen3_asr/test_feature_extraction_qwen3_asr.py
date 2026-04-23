# Copyright 2026 HuggingFace Inc.
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

import itertools
import random
import unittest

import numpy as np

from transformers import Qwen3ASRFeatureExtractor

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None):
    rng = rng or global_rng
    values = []
    for _ in range(shape[0]):
        values.append([rng.random() * scale for _ in range(shape[1])])
    return values


class Qwen3ASRFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=10,
        hop_length=160,
        chunk_length=8,
        padding_value=0.0,
        sampling_rate=4_000,
        return_attention_mask=False,
        n_window=13,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.n_window = n_window

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "chunk_length": self.chunk_length,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "n_window": self.n_window,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            speech_inputs = [
                floats_list((x, self.feature_size))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


class Qwen3ASRFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Qwen3ASRFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Qwen3ASRFeatureExtractionTester(self)

    def test_default_feature_size_is_128(self):
        """Qwen3 ASR uses 128-bin mel filters by default."""
        fe = Qwen3ASRFeatureExtractor()
        self.assertEqual(fe.feature_size, 128)
        self.assertEqual(fe.mel_filters.shape[1], 128)

    def test_default_n_window_is_50(self):
        fe = Qwen3ASRFeatureExtractor()
        self.assertEqual(fe.n_window, 50)

    def test_mel_padding_aligns_to_chunk(self):
        """The mel time axis is right-padded to a multiple of `2 * n_window`."""
        fe = Qwen3ASRFeatureExtractor()
        # 5.85 s at 16 kHz -> 585 mel frames before padding -> 600 after (multiple of 100).
        audio = np.random.randn(int(5.85 * 16_000)).astype(np.float32)
        out = fe(
            audio,
            sampling_rate=16_000,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="np",
        )
        self.assertEqual(out["input_features"].shape, (1, 128, 600))
        self.assertEqual(out["attention_mask"].shape, (1, 600))
        self.assertEqual(int(out["attention_mask"].sum(-1)), 585)
        self.assertEqual(out["input_features"].shape[-1] % 100, 0)

    def test_n_window_kwarg_override(self):
        fe = Qwen3ASRFeatureExtractor()
        audio = np.random.randn(int(5.85 * 16_000)).astype(np.float32)
        out = fe(
            audio,
            sampling_rate=16_000,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="np",
            n_window=25,
        )
        self.assertEqual(out["input_features"].shape[-1] % 50, 0)

    def test_n_window_disabled(self):
        """`n_window=0` disables mel-axis padding."""
        fe = Qwen3ASRFeatureExtractor()
        audio = np.random.randn(int(5.85 * 16_000)).astype(np.float32)
        out = fe(
            audio,
            sampling_rate=16_000,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="np",
            n_window=0,
        )
        self.assertEqual(out["input_features"].shape[-1], 585)
        self.assertEqual(out["attention_mask"].shape[-1], 585)

    def test_batched_call_shape(self):
        fe = Qwen3ASRFeatureExtractor()
        # Two clips of different lengths; padded to the longer one (rounded up to 2 * n_window).
        audio = [
            np.random.randn(int(2.0 * 16_000)).astype(np.float32),
            np.random.randn(int(5.5 * 16_000)).astype(np.float32),
        ]
        out = fe(
            audio,
            sampling_rate=16_000,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="np",
        )
        self.assertEqual(out["input_features"].ndim, 3)
        self.assertEqual(out["input_features"].shape[0], 2)
        self.assertEqual(out["input_features"].shape[1], 128)
        self.assertEqual(out["input_features"].shape[-1] % 100, 0)
        per_sample_valid = out["attention_mask"].sum(-1).tolist()
        self.assertEqual(per_sample_valid, [200, 550])

    def test_mismatched_sampling_rate_raises(self):
        fe = Qwen3ASRFeatureExtractor(sampling_rate=16_000)
        audio = np.random.randn(16_000).astype(np.float32)
        with self.assertRaises(ValueError):
            fe(audio, sampling_rate=8_000, return_tensors="np")
