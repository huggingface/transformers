# Copyright 2025 HuggingFace Inc.
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
import os
import random
import tempfile
import unittest
from typing import Optional, Sequence

import numpy as np
from parameterized import parameterized

from transformers.models.gemma3n import Gemma3nAudioFeatureExtractor
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
)
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    pass

global_rng = random.Random()

MAX_LENGTH_FOR_TESTING = 512


def floats_list(shape, scale=1.0, rng=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for _ in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


class Gemma3nAudioFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        # ignore hop_length / frame_length for now, as ms -> length conversion causes issues with serialization tests
        # frame_length_ms: float = 32.0,
        # hop_length: float = 10.0,
        min_frequency: float = 125.0,
        max_frequency: float = 7600.0,
        preemphasis: float = 0.97,
        preemphasis_htk_flavor: bool = True,
        fft_overdrive: bool = True,
        dither: float = 0.0,
        input_scale_factor: float = 1.0,
        mel_floor: float = 1e-5,
        per_bin_mean: Optional[Sequence[float]] = None,
        per_bin_stddev: Optional[Sequence[float]] = None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask
        # ignore hop_length / frame_length for now, as ms -> length conversion causes issues with serialization tests
        # self.frame_length_ms = frame_length_ms
        # self.hop_length = hop_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.fft_overdrive = fft_overdrive
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.mel_floor = mel_floor
        self.per_bin_mean = per_bin_mean
        self.per_bin_stddev = per_bin_stddev

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "return_attention_mask": self.return_attention_mask,
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "preemphasis": self.preemphasis,
            "preemphasis_htk_flavor": self.preemphasis_htk_flavor,
            "fft_overdrive": self.fft_overdrive,
            "dither": self.dither,
            "input_scale_factor": self.input_scale_factor,
            "mel_floor": self.mel_floor,
            "per_bin_mean": self.per_bin_mean,
            "per_bin_stddev": self.per_bin_stddev,
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


class Gemma3nAudioFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Gemma3nAudioFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Gemma3nAudioFeatureExtractionTester(self)

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = feat_extract_first.mel_filters
        mel_2 = feat_extract_second.mel_filters
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
        mel_1 = feat_extract_first.mel_filters
        mel_2 = feat_extract_second.mel_filters
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_from_pretrained_kwargs(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(
                tmpdirname, feature_size=2 * self.feat_extract_dict["feature_size"]
            )

        mel_1 = feat_extract_first.mel_filters
        mel_2 = feat_extract_second.mel_filters
        self.assertTrue(2 * mel_1.shape[1] == mel_2.shape[1])

    @parameterized.expand(
        [
            ([floats_list((1, x))[0] for x in range(800, 1400, 200)],),
            ([floats_list((1, x))[0] for x in (800, 800, 800)],),
            ([floats_list((1, x))[0] for x in range(200, (MAX_LENGTH_FOR_TESTING + 500), 200)], True),
        ]
    )
    def test_call(self, audio_inputs, test_truncation=False):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]

        input_features = feature_extractor(np_audio_inputs, padding="max_length", return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        # input_features.shape should be (batch, num_frames, n_mels) ~= (batch, num_frames, feature_size)
        # 480_000 is the max_length that inputs are padded to. we use that to calculate num_frames
        expected_num_frames = (480_000 - feature_extractor.frame_length) // (feature_extractor.hop_length) + 1
        self.assertTrue(
            input_features.shape[-2] == expected_num_frames,
            f"no match: {input_features.shape[-1]} vs {expected_num_frames}",
        )
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size)

        encoded_sequences_1 = feature_extractor(audio_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_audio_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        if test_truncation:
            audio_inputs_truncated = [x[:MAX_LENGTH_FOR_TESTING] for x in audio_inputs]
            np_audio_inputs_truncated = [np.asarray(audio_input) for audio_input in audio_inputs_truncated]

            encoded_sequences_1 = feature_extractor(
                audio_inputs_truncated, max_length=MAX_LENGTH_FOR_TESTING, return_tensors="np"
            ).input_features
            encoded_sequences_2 = feature_extractor(
                np_audio_inputs_truncated, max_length=MAX_LENGTH_FOR_TESTING, return_tensors="np"
            ).input_features
            for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
                self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_dither(self):
        np.random.seed(42)  # seed the dithering randn()

        # Tests that features with and without little dithering are similar, but not the same
        dict_no_dither = self.feat_extract_tester.prepare_feat_extract_dict()
        dict_no_dither["dither"] = 0.0

        dict_dither = self.feat_extract_tester.prepare_feat_extract_dict()
        dict_dither["dither"] = 0.00003  # approx. 1/32k

        feature_extractor_no_dither = self.feature_extraction_class(**dict_no_dither)
        feature_extractor_dither = self.feature_extraction_class(**dict_dither)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # compute features
        input_features_no_dither = feature_extractor_no_dither(
            np_speech_inputs, padding=True, return_tensors="np", sampling_rate=dict_no_dither["sampling_rate"]
        ).input_features
        input_features_dither = feature_extractor_dither(
            np_speech_inputs, padding=True, return_tensors="np", sampling_rate=dict_dither["sampling_rate"]
        ).input_features

        # test there is a difference between features (there's added noise to input signal)
        diff = input_features_dither - input_features_no_dither

        # features are not identical
        self.assertTrue(np.abs(diff).mean() > 1e-6)
        # features are not too different
        self.assertTrue(np.abs(diff).mean() <= 1e-4)
        self.assertTrue(np.abs(diff).max() <= 5e-3)

    @require_torch
    def test_double_precision_pad(self):
        import torch

        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)
