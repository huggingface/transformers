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


import itertools
import random
import unittest

import numpy as np

from transformers import is_speech_available
from transformers.testing_utils import require_torch

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_speech_available():
    from transformers import MCTCTFeatureExtractor

global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for _batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


@require_torch
class MCTCTFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=24,
        num_mel_bins=24,
        padding_value=0.0,
        sampling_rate=16_000,
        return_attention_mask=True,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.num_mel_bins = num_mel_bins
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "num_mel_bins": self.num_mel_bins,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
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
class MCTCTFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = MCTCTFeatureExtractor if is_speech_available() else None

    def setUp(self):
        self.feat_extract_tester = MCTCTFeatureExtractionTester(self)

    def _check_zero_mean_unit_variance(self, input_vector):
        self.assertTrue(np.all(np.mean(input_vector) < 1e-3))
        self.assertTrue(np.all(np.abs(np.var(input_vector) - 1) < 1e-3))

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 12000
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 2000)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding=True, return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_cepstral_mean_and_variance_normalization(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 2000)]

        paddings = ["longest", "max_length", "do_not_pad"]
        max_lengths = [None, 16, None]
        for max_length, padding in zip(max_lengths, paddings):
            inputs = feature_extractor(
                speech_inputs,
                padding=padding,
                max_length=max_length,
                return_attention_mask=True,
                truncation=max_length is not None,  # reference to #16419
            )
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask
            fbank_feat_lengths = [np.sum(x) for x in attention_mask]
            self._check_zero_mean_unit_variance(input_features[0][: fbank_feat_lengths[0]])
            self._check_zero_mean_unit_variance(input_features[1][: fbank_feat_lengths[1]])
            self._check_zero_mean_unit_variance(input_features[2][: fbank_feat_lengths[2]])

    def test_cepstral_mean_and_variance_normalization_np(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 2000)]

        paddings = ["longest", "max_length", "do_not_pad"]
        max_lengths = [None, 16, None]
        for max_length, padding in zip(max_lengths, paddings):
            inputs = feature_extractor(
                speech_inputs,
                max_length=max_length,
                padding=padding,
                return_tensors="np",
                return_attention_mask=True,
                truncation=max_length is not None,
            )
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask
            fbank_feat_lengths = [np.sum(x) for x in attention_mask]

            self._check_zero_mean_unit_variance(input_features[0][: fbank_feat_lengths[0]])
            self.assertTrue(input_features[0][fbank_feat_lengths[0] :].sum() < 1e-6)
            self._check_zero_mean_unit_variance(input_features[1][: fbank_feat_lengths[1]])
            self.assertTrue(input_features[0][fbank_feat_lengths[1] :].sum() < 1e-6)
            self._check_zero_mean_unit_variance(input_features[2][: fbank_feat_lengths[2]])

    def test_cepstral_mean_and_variance_normalization_trunc_max_length(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 2000)]
        inputs = feature_extractor(
            speech_inputs,
            padding="max_length",
            max_length=4,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask
        fbank_feat_lengths = np.sum(attention_mask == 1, axis=1)

        self._check_zero_mean_unit_variance(input_features[0, : fbank_feat_lengths[0]])
        self._check_zero_mean_unit_variance(input_features[1])
        self._check_zero_mean_unit_variance(input_features[2])

    def test_cepstral_mean_and_variance_normalization_trunc_longest(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 2000)]
        inputs = feature_extractor(
            speech_inputs,
            padding="longest",
            max_length=4,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask
        fbank_feat_lengths = np.sum(attention_mask == 1, axis=1)

        self._check_zero_mean_unit_variance(input_features[0, : fbank_feat_lengths[0]])
        self._check_zero_mean_unit_variance(input_features[1, : fbank_feat_lengths[1]])
        self._check_zero_mean_unit_variance(input_features[2])

        # make sure that if max_length < longest -> then pad to max_length
        self.assertEqual(input_features.shape, (3, 4, 24))

        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 2000)]
        inputs = feature_extractor(
            speech_inputs,
            padding="longest",
            max_length=16,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask
        fbank_feat_lengths = np.sum(attention_mask == 1, axis=1)

        self._check_zero_mean_unit_variance(input_features[0, : fbank_feat_lengths[0]])
        self._check_zero_mean_unit_variance(input_features[1, : fbank_feat_lengths[1]])
        self._check_zero_mean_unit_variance(input_features[2])

        # make sure that if max_length < longest -> then pad to max_length
        self.assertEqual(input_features.shape, (3, 16, 24))

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

    def test_different_window(self):
        import torch

        init_dict = self.feat_extract_tester.prepare_feat_extract_dict()
        init_dict["win_function"] = "hann_window"

        feature_extractor = self.feature_extraction_class(**init_dict)
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_integration(self):
        # fmt: off
        expected = np.array([
            [
                1.1280,  1.1319,  1.2744,  1.4369,  1.4328,  1.3671,  1.2889,  1.3046,
                1.4419,  0.8387,  0.2995,  0.0404,  0.1068,  0.0472,  0.3728,  1.3356,
                1.4491,  0.4770,  0.3997,  0.2776,  0.3184, -0.1243, -0.1170, -0.0828
            ],
            [
                1.0826,  1.0565,  1.2110,  1.3886,  1.3416,  1.2009,  1.1894,  1.2707,
                1.5153,  0.7005,  0.4916,  0.4017,  0.3743,  0.1935,  0.4228,  1.1084,
                0.9768,  0.0608,  0.2044,  0.1723,  0.0433, -0.2360, -0.2478, -0.2643
            ],
            [
                1.0590,  0.9923,  1.1185,  1.3309,  1.1971,  1.0067,  1.0080,  1.2036,
                1.5397,  1.0383,  0.7672,  0.7551,  0.4878,  0.8771,  0.7565,  0.8775,
                0.9042,  0.4595,  0.6157,  0.4954,  0.1857,  0.0307,  0.0199,  0.1033
            ],
        ])
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        input_features = feature_extractor(input_speech, sampling_rate=16000, return_tensors="pt").input_features
        self.assertTrue(np.allclose(input_features[0, 100:103], expected, atol=1e-4))
