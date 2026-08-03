# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers import Gemma4UnifiedAudioFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

global_rng = random.Random()


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


class Gemma4UnifiedAudioFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=80,
        sampling_rate=16_000,
        padding_value=0.0,
        audio_samples_per_token=80,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.audio_samples_per_token = audio_samples_per_token

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "audio_samples_per_token": self.audio_samples_per_token,
        }

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTester.prepare_inputs_for_common
    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

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


@require_torch
class Gemma4UnifiedAudioFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Gemma4UnifiedAudioFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Gemma4UnifiedAudioFeatureExtractionTester(self)

    def test_chunking_shape(self):
        """A 1-D waveform is chunked into ceil(len / audio_samples_per_token) frames."""
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        samples_per_token = self.feat_extract_tester.audio_samples_per_token

        waveform = np.random.rand(1000).astype(np.float32)
        result = feature_extractor(waveform, return_tensors="np")

        expected_num_tokens = -(-1000 // samples_per_token)
        self.assertEqual(result.input_features.shape, (1, expected_num_tokens, samples_per_token))
        self.assertEqual(result.input_features_mask.shape, (1, expected_num_tokens))
        self.assertTrue(result.input_features_mask.all())

    def test_chunking_preserves_values(self):
        """Chunking is a pure reshape: values are preserved and the last frame is zero-padded."""
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        samples_per_token = self.feat_extract_tester.audio_samples_per_token

        length = samples_per_token * 3 + 17
        waveform = np.random.rand(length).astype(np.float32)
        result = feature_extractor(waveform, return_tensors="np")

        flattened = result.input_features[0].flatten()
        self.assertTrue(np.array_equal(flattened[:length], waveform))
        self.assertTrue((flattened[length:] == 0).all())

    def test_batch_padding_longest(self):
        """Batched waveforms of different lengths are padded to the longest, with a matching mask."""
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        samples_per_token = self.feat_extract_tester.audio_samples_per_token

        short = np.random.rand(samples_per_token * 5).astype(np.float32)
        long = np.random.rand(samples_per_token * 10).astype(np.float32)
        result = feature_extractor([short, long], padding="longest", return_tensors="np")

        self.assertEqual(result.input_features.shape, (2, 10, samples_per_token))
        self.assertEqual(result.input_features_mask.shape, (2, 10))
        self.assertTrue(result.input_features_mask[0, :5].all())
        self.assertFalse(result.input_features_mask[0, 5:].any())
        self.assertTrue(result.input_features_mask[1].all())
        self.assertTrue((result.input_features[0, 5:] == 0).all())

    def test_max_length_truncation(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        samples_per_token = self.feat_extract_tester.audio_samples_per_token

        waveform = np.random.rand(samples_per_token * 10).astype(np.float32)
        result = feature_extractor(waveform, padding="max_length", max_length=4, truncation=True, return_tensors="np")

        self.assertEqual(result.input_features.shape, (1, 4, samples_per_token))

    def test_return_tensors_pt(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        samples_per_token = self.feat_extract_tester.audio_samples_per_token

        waveform = np.random.rand(samples_per_token * 4).astype(np.float32)
        result = feature_extractor(waveform, return_tensors="pt")

        self.assertIsInstance(result.input_features, torch.Tensor)
        self.assertEqual(result.input_features.dtype, torch.float32)
        self.assertIsInstance(result.input_features_mask, torch.Tensor)
        self.assertEqual(result.input_features_mask.dtype, torch.bool)

    def test_numpy_and_pt_outputs_match(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        samples_per_token = self.feat_extract_tester.audio_samples_per_token

        waveform = np.random.rand(samples_per_token * 6 + 13).astype(np.float32)
        result_np = feature_extractor(waveform, return_tensors="np")
        result_pt = feature_extractor(waveform, return_tensors="pt")

        self.assertTrue(np.array_equal(result_np.input_features, result_pt.input_features.numpy()))
        self.assertTrue(np.array_equal(result_np.input_features_mask, result_pt.input_features_mask.numpy()))
