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

import unittest

import numpy as np

from transformers import MossTranscribeDiarizeFeatureExtractor

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


class MossTranscribeDiarizeFeatureExtractionTester:
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

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "chunk_length": self.chunk_length,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                floats_list((x, self.feature_size))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


class MossTranscribeDiarizeFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = MossTranscribeDiarizeFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = MossTranscribeDiarizeFeatureExtractionTester(self)

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = feature_extractor.sampling_rate
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]

        # Non-batched input matches the same sample processed as part of a batch.
        encoded_1 = feature_extractor(np_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np")
        encoded_2 = feature_extractor(np_audio_inputs, sampling_rate=sampling_rate, return_tensors="np")
        self.assertTrue(np.allclose(encoded_1.input_features, encoded_2.input_features[0], atol=1e-3))

    def test_padding_mask_generation(self):
        """Audio longer than `chunk_length` seconds is split into consecutive windows, and `padding_mask`
        records each sample's original raw length rather than a multiple of the window size."""
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        window_size = feature_extractor.n_samples
        short_audio = np.random.randn(5_000).astype(np.float32)
        long_audio = np.random.randn(int(2.2 * window_size)).astype(np.float32)

        out = feature_extractor([short_audio, long_audio], sampling_rate=feature_extractor.sampling_rate)
        # 1 window for the short sample + 3 windows for the long one.
        self.assertEqual(out.input_features.shape[0], 4)
        self.assertEqual(out.padding_mask.shape, (2, len(long_audio)))
        self.assertEqual(out.padding_mask.sum(-1).tolist(), [len(short_audio), len(long_audio)])

    def test_sampling_rate_validation(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = np.random.randn(5_000).astype(np.float32)
        with self.assertRaises(ValueError):
            feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate + 1)
            