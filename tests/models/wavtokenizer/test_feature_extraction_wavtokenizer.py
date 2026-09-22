# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the WavTokenizer feature extractor."""

import itertools
import unittest

import numpy as np

from transformers import WavTokenizerFeatureExtractor
from transformers.testing_utils import require_torch

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


@require_torch
class WavTokenizerFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=24000,
        hop_length=600,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.hop_length = hop_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "hop_length": self.hop_length,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            audio_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            audio_inputs = [
                _flatten(floats_list((x, self.feature_size)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            audio_inputs = [np.asarray(x) for x in audio_inputs]

        return audio_inputs


@require_torch
class WavTokenizerFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = WavTokenizerFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = WavTokenizerFeatureExtractionTester(self)

    def test_single_input_not_padded(self):
        # single inputs stay unpadded (the model pads internally) so codes stay bit-identical to the
        # original WavTokenizer pipeline
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        hop = feat_extract.hop_length
        audio = floats_list((1, hop + 3))[0]
        processed = feat_extract(audio, sampling_rate=feat_extract.sampling_rate, return_tensors="np")
        self.assertEqual(processed.input_values.shape[-1], hop + 3)
        self.assertEqual(int(processed.padding_mask.sum()), hop + 3)

    def test_batch_padded_to_longest(self):
        """Batches pad to the longest sample by default, or truncate to `max_length` first, with `padding_mask`
        marking the valid samples and `input_values` shaped `(batch_size, 1, num_samples)`."""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        lengths = [1200, 800, 1234]
        batch = [np.arange(length, dtype=np.float32) for length in lengths]

        processed = feat_extract(batch, sampling_rate=feat_extract.sampling_rate, return_tensors="np")
        self.assertEqual(processed.input_values.shape, (len(batch), 1, max(lengths)))
        self.assertEqual(processed.padding_mask.sum(-1).tolist(), lengths)
        np.testing.assert_array_equal(processed.input_values[1, 0, : lengths[1]], batch[1])
        np.testing.assert_array_equal(processed.input_values[1, 0, lengths[1] :], 0.0)

        max_length = 1000
        processed = feat_extract(
            batch,
            truncation=True,
            max_length=max_length,
            sampling_rate=feat_extract.sampling_rate,
            return_tensors="np",
        )
        self.assertEqual(processed.input_values.shape, (len(batch), 1, max_length))
        self.assertEqual(processed.padding_mask.sum(-1).tolist(), [max_length, lengths[1], max_length])
        np.testing.assert_array_equal(processed.input_values[0, 0], batch[0][:max_length])
        np.testing.assert_array_equal(processed.input_values[1, 0, : lengths[1]], batch[1])
        np.testing.assert_array_equal(processed.input_values[1, 0, lengths[1] :], 0.0)

    def test_rejects_invalid_audio(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = feat_extract.sampling_rate
        for name, audio, kwargs in [
            ("empty", np.zeros(0, dtype=np.float32), {"sampling_rate": sampling_rate}),
            ("wrong sampling rate", floats_list((1, 800))[0], {"sampling_rate": sampling_rate + 1}),
            ("non-mono", [np.random.rand(2, 800).astype(np.float32)], {"sampling_rate": sampling_rate}),
        ]:
            with self.subTest(name), self.assertRaises(ValueError):
                feat_extract(audio, **kwargs)
