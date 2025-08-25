# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Parakeet feature extraction."""

import itertools
import random
import tempfile
import unittest

import numpy as np

from transformers import ParakeetFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


global_rng = random.Random()


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


class ParakeetFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=128,
        hop_length=160,
        win_length=400,
        n_fft=512,
        sampling_rate=16000,
        padding_value=0.0,
        normalize="per_feature",
        return_attention_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.normalize = normalize
        self.return_attention_mask = return_attention_mask

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "normalize": self.normalize,
            "return_attention_mask": self.return_attention_mask,
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


class ParakeetFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = ParakeetFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = ParakeetFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_common_properties(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feat_extract, "feature_size"))
        self.assertTrue(hasattr(feat_extract, "sampling_rate"))
        self.assertTrue(hasattr(feat_extract, "hop_length"))

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            feat_extract_first.save_pretrained(tmpdirname)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_init_without_params(self):
        feat_extract = self.feature_extraction_class()
        self.assertIsNotNone(feat_extract)

    @require_torch
    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [list(itertools.chain(*floats_list((1, x)))) for x in range(800, 1400, 200)]

        # Test batched input with torch tensors
        if is_torch_available():
            # Convert to torch tensors
            torch_speech_inputs = [torch.tensor(speech_input, dtype=torch.float32) for speech_input in speech_inputs]

            # Test single input
            encoded_sequence = feat_extract(torch_speech_inputs[0])
            self.assertTrue("input_features" in encoded_sequence)
            self.assertTrue("attention_mask" in encoded_sequence)

            # Test batched input with list of tensors (automatic padding)
            encoded_sequences = feat_extract(torch_speech_inputs)

            self.assertTrue("input_features" in encoded_sequences)
            self.assertTrue("attention_mask" in encoded_sequences)
            self.assertEqual(encoded_sequences.input_features.shape[0], len(torch_speech_inputs))

    @require_torch
    def test_torch_extraction(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())

        # Create test audio
        speech_inputs = [list(itertools.chain(*floats_list((1, x)))) for x in range(800, 1400, 200)]
        torch_speech_inputs = [torch.tensor(speech_input, dtype=torch.float32) for speech_input in speech_inputs]

        # Test single input
        features = feat_extract(torch_speech_inputs[0])

        # Check output format
        self.assertEqual(features.input_features.dim(), 3)  # (batch, time, features)
        self.assertEqual(features.input_features.shape[2], feat_extract.feature_size)
        self.assertEqual(features.attention_mask.dim(), 2)  # (batch, time)

    @require_torch
    def test_attention_mask_computation(self):
        feat_extract = self.feature_extraction_class()

        # Create inputs of different lengths
        short_audio = torch.randn(4000)  # 0.25 seconds
        long_audio = torch.randn(12000)  # 0.75 seconds

        # Pass as list of tensors (automatic padding will be handled internally)
        features = feat_extract([short_audio, long_audio])

        # Check attention mask
        attention_mask = features.attention_mask

        # First sequence should have fewer valid frames
        valid_frames_0 = attention_mask[0].sum().item()
        valid_frames_1 = attention_mask[1].sum().item()

        self.assertLess(valid_frames_0, valid_frames_1)

    def test_invalid_normalize_parameter(self):
        """Test that invalid normalize parameter raises ValueError."""
        with self.assertRaises(ValueError):
            ParakeetFeatureExtractor(normalize="invalid_type")

    def test_feature_extractor_without_torch(self):
        """Test that appropriate error is raised when torch is not available."""
        import unittest.mock

        with unittest.mock.patch(
            "transformers.models.parakeet.feature_extraction_parakeet.is_torch_available", lambda: False
        ):
            with self.assertRaises(ImportError):
                feat_extract = ParakeetFeatureExtractor()
                audio = [1, 2, 3, 4, 5]  # Simple list of numbers
                feat_extract(audio)
