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

import itertools
import unittest

import numpy as np

from transformers import Qwen3TTSTokenizerFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


@require_torch
class Qwen3TTSTokenizerFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=24000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
        }

    # Copied from tests.models.encodec.test_feature_extraction_encodec.EnCodecFeatureExtractionTester.prepare_inputs_for_common
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
class Qwen3TTSTokenizerFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Qwen3TTSTokenizerFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Qwen3TTSTokenizerFeatureExtractionTester(self)

    def test_call(self):
        TOL = 1e-6

        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = feature_extractor.sampling_rate
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]
        torch_audio_inputs = [torch.tensor(audio_input) for audio_input in audio_inputs]

        # Test non-batched input
        encoded_sequences_1 = feature_extractor(torch_audio_inputs[0], sampling_rate=sampling_rate).input_values
        encoded_sequences_2 = feature_extractor(np_audio_inputs[0], sampling_rate=sampling_rate).input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=TOL))

        # Test batched input
        encoded_sequences_1 = feature_extractor(torch_audio_inputs, sampling_rate=sampling_rate).input_values
        encoded_sequences_2 = feature_extractor(np_audio_inputs, sampling_rate=sampling_rate).input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=TOL))

    def test_sampling_rate_validation(self):
        """Test that sampling rate validation works correctly."""
        feature_extractor = Qwen3TTSTokenizerFeatureExtractor(sampling_rate=24000)
        input_audio = np.random.randn(1000).astype(np.float32)

        result = feature_extractor([input_audio], sampling_rate=24000)
        self.assertIsInstance(result.input_values, torch.Tensor)

        with self.assertRaises(ValueError):
            feature_extractor([input_audio], sampling_rate=16000)

    def test_padding_mask_generation(self):
        """Test that padding masks are generated correctly."""
        feature_extractor = Qwen3TTSTokenizerFeatureExtractor()
        audio1 = np.random.randn(1000).astype(np.float32)
        audio2 = np.random.randn(1500).astype(np.float32)

        result = feature_extractor([audio1, audio2], padding=True, return_attention_mask=True)
        self.assertIn("padding_mask", result)
        # input_values is (batch_size, num_samples), the padding_mask shares that shape
        self.assertEqual(result.padding_mask.shape, result.input_values.shape)

        # First (shorter) sample should have some padding (False values at the end)
        self.assertTrue(torch.any(~result.padding_mask[0].bool()))
        # Second (longest) sample should have no padding (all True values)
        self.assertTrue(torch.all(result.padding_mask[1].bool()))
