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
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


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

    def test_call(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]

        # Test not batched input
        encoded_sequences_1 = feat_extract(audio_inputs[0], return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_audio_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feat_extract(audio_inputs, padding=True, return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_audio_inputs, padding=True, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

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
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        lengths = [800, 1000, 1234]
        batch = [floats_list((1, length))[0] for length in lengths]
        processed = feat_extract(batch, sampling_rate=feat_extract.sampling_rate, return_tensors="np")
        self.assertEqual(processed.input_values.shape[-1], max(lengths))
        self.assertEqual(processed.padding_mask.sum(-1).tolist(), lengths)

    def test_rejects_empty_audio(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        with self.assertRaises(ValueError):
            feat_extract(np.zeros(0, dtype=np.float32), sampling_rate=feat_extract.sampling_rate)

    def test_get_num_audio_codes(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        hop = feat_extract.hop_length
        for num_samples, expected in [
            (1, 1),
            (hop - 1, 1),
            (hop, 1),
            (hop + 1, 2),
            (10 * hop, 10),
            (10 * hop + hop // 2, 11),
        ]:
            self.assertEqual(feat_extract.get_num_audio_codes(num_samples), expected)

    def test_rejects_wrong_sampling_rate(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = floats_list((1, 800))[0]
        with self.assertRaises(ValueError):
            feat_extract(audio, sampling_rate=feat_extract.sampling_rate + 1)

    def test_rejects_non_mono(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        stereo = np.random.rand(2, 800).astype(np.float32)
        with self.assertRaises(ValueError):
            feat_extract([stereo], sampling_rate=feat_extract.sampling_rate)

    def test_double_precision_pad(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_audio_inputs = np.random.rand(100).astype(np.float64)
        py_audio_inputs = np_audio_inputs.tolist()

        for inputs in [py_audio_inputs, np_audio_inputs]:
            np_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_values.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_values.dtype == torch.float32)
