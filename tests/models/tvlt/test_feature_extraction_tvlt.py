# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
""" Testing suite for the TVLT feature extraction. """

import random
import unittest

import numpy as np

from transformers import is_speech_available
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

if is_speech_available():
    from transformers import TvltFeatureExtractor

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


@require_torch
@require_torchaudio
class TvltFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        audio_size=2048,
        num_channels=1,
        feature_size=128,
        sampling_rate=44100,
        hop_length=512,
        chunk_length=30,
    ):
        self.parent = parent
        self.batch_size = batch_size

        self.audio_size = audio_size
        self.num_channels = num_channels
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        return {
            "max_seq_length": self.max_seq_length,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = [floats_list((self.audio_size, self.feature_size)) for _ in range(self.batch_size)]
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                floats_list((x, self.feature_size))
                for x in range(0, self.audio_size, self.audio_size // self.batch_size)
            ]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


@require_torch
@require_torchaudio
class TvltFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = TvltFeatureExtractor if is_speech_available() else None

    def setUp(self):
        self.feature_extract_tester = TvltFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "audio_size"))

    def test_batch_feature(self):
        pass

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 20000)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        print(np_speech_inputs[0].shape)
        # Test not batched input
        encoded_audios = feature_extractor(np_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

        # Test batched
        encoded_audios = feature_extractor(np_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 20000)]
        torch_speech_inputs = [torch.tensor(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_audios = feature_extractor(torch_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

        # Test batched
        encoded_audios = feature_extractor(torch_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)
