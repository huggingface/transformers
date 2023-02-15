# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import itertools
import os
import random
import tempfile
import unittest

import numpy as np

from transformers import is_datasets_available, is_speech_available
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_torchaudio
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

if is_datasets_available():
    from datasets import load_dataset

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


class TvltFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        spectrogram_length=2048,
        feature_size=128,
        num_audio_channels=1,
        hop_length=512,
        chunk_length=30,
        sampling_rate=44100,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.spectrogram_length = spectrogram_length
        self.feature_size = feature_size
        self.num_audio_channels = num_audio_channels
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        return {
            "spectrogram_length": self.spectrogram_length,
            "feature_size": self.feature_size,
            "num_audio_channels": self.num_audio_channels,
            "hop_length": self.hop_length,
            "chunk_length": self.chunk_length,
            "sampling_rate": self.sampling_rate,
        }

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
@require_torchaudio
class TvltFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = TvltFeatureExtractor if is_speech_available() else None

    def setUp(self):
        self.feat_extract_tester = TvltFeatureExtractionTester(self)

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "spectrogram_length"))
        self.assertTrue(hasattr(feature_extractor, "feature_size"))
        self.assertTrue(hasattr(feature_extractor, "num_audio_channels"))
        self.assertTrue(hasattr(feature_extractor, "hop_length"))
        self.assertTrue(hasattr(feature_extractor, "chunk_length"))
        self.assertTrue(hasattr(feature_extractor, "sampling_rate"))

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = dict_first.pop("mel_filters")
        mel_2 = dict_second.pop("mel_filters")
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = dict_first.pop("mel_filters")
        mel_2 = dict_second.pop("mel_filters")
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_call(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 20000)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_audios = feature_extractor(np_speech_inputs[0], return_tensors="np", sampling_rate=44100).audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

        # Test batched
        encoded_audios = feature_extractor(np_speech_inputs, return_tensors="np", sampling_rate=44100).audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

        # Test audio masking
        encoded_audios = feature_extractor(
            np_speech_inputs, return_tensors="np", sampling_rate=44100, mask_audio=True
        ).audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_integration(self):
        input_speech = self._load_datasamples(1)
        feaure_extractor = TvltFeatureExtractor()
        audio_values = feaure_extractor(input_speech, return_tensors="pt").audio_values

        self.assertTrue(audio_values.shape, [1, 1, 192, 128])

        expected_slice = torch.tensor([[-0.3032, -0.2708], [-0.4434, -0.4007]])
        self.assertTrue(torch.allclose(audio_values[0, 0, :2, :2], expected_slice, atol=1e-4))
