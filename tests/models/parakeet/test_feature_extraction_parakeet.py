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
import unittest

import numpy as np

from transformers import ParakeetFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_datasets_available, is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

if is_datasets_available():
    from datasets import load_dataset

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
        feature_size=80,
        hop_length=160,
        win_length=400,
        n_fft=512,
        sampling_rate=16000,
        padding_value=0.0,
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

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
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

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @require_torch
    def test_torch_integration(self):
        """
        reproducer: https://gist.github.com/eustlb/c4a0999e54466b7e8d8b040d8e0900df
        """
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                0.60935932, 1.18187428, 1.29877627, 1.36461377, 1.09311509, 1.39821815,
                1.63753450, 1.37100816, 1.26510608, 1.70332706, 1.69067430, 1.28770995,
                1.52999651, 1.77962756, 1.71420062, 1.21944094, 1.30884087, 1.44343364,
                1.17694926, 1.42690814, 1.78877723, 1.68655288, 1.27155364, 1.66103351,
                1.75820673, 1.41575801, 1.40622294, 1.70603478, 1.63117850, 1.13353217,
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = ParakeetFeatureExtractor()
        inputs = feature_extractor(input_speech, return_tensors="pt")

        self.assertEqual(inputs.input_features.shape, (1, 586, 80))
        torch.testing.assert_close(inputs.input_features[0, 100, :30], EXPECTED_INPUT_FEATURES, atol=1e-4, rtol=1e-4)

        self.assertEqual(inputs.attention_mask.shape, (1, 586))
        # last frame should be masked
        self.assertEqual(inputs.attention_mask.sum(), 585)

    @require_torch
    def test_torch_integration_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/c4a0999e54466b7e8d8b040d8e0900df
        """
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [ 0.60935932,  1.18187428,  1.29877627,  1.36461377,  1.09311533,
                  1.39821827,  1.63753450,  1.37100816,  1.26510608,  1.70332706,
                  1.69067478,  1.28770995,  1.52999651,  1.77962780,  1.71420062,
                  1.21944094,  1.30884087,  1.44343400,  1.17694926,  1.42690814,
                  1.78877664,  1.68655288,  1.27155364,  1.66103351,  1.75820673,
                  1.41575801,  1.40622294,  1.70603478,  1.63117862,  1.13353217],
                [ 0.58339858,  0.54317272,  0.46222782,  0.34154415,  0.17806509,
                  0.32182255,  0.28909618,  0.02141305, -0.09710173, -0.35818669,
                 -0.48172510, -0.52942866, -0.58029658, -0.70519227, -0.67929971,
                 -0.54698551, -0.28611183, -0.24780270, -0.31363955, -0.41913241,
                 -0.32394424, -0.44897896, -0.68657434, -0.62047797, -0.46886450,
                 -0.65987164, -1.02435589, -0.58527517, -0.56095684, -0.73582536],
                [-0.91937613, -0.97933632, -1.06843162, -1.02642107, -0.94232899,
                 -0.83840621, -0.82306921, -0.45763230, -0.45182887, -0.75917768,
                 -0.42541453, -0.28512970, -0.39637473, -0.66478080, -0.68004298,
                 -0.49690303, -0.31799242, -0.12917191,  0.13149273,  0.10163058,
                 -0.40041649,  0.05001565,  0.23906317,  0.28816083,  0.14308788,
                 -0.29588422, -0.05428466,  0.14418560,  0.28865972, -0.12138986],
                [ 0.73217624,  0.84484011,  0.79323846,  0.66315967,  0.41556871,
                  0.88633078,  0.90718138,  0.91268104,  1.15920067,  1.26141894,
                  1.10222173,  0.92990804,  0.96352047,  0.88142169,  0.56635213,
                  0.71491158,  0.81301254,  0.67301887,  0.74780160,  0.64429688,
                  0.22885245,  0.47035533,  0.46498337,  0.17544533,  0.44458991,
                  0.79245001,  0.57207537,  0.85768145,  1.00491571,  0.93360955],
                [ 1.40496337,  1.32492661,  1.16519547,  0.98379827,  0.77614164,
                  0.95871657,  0.81910741,  1.23010278,  1.33011520,  1.16538525,
                  1.28319681,  1.45041633,  1.33421600,  0.91677380,  0.67107433,
                  0.52890682,  0.82009870,  1.15821445,  1.15343642,  1.10958862,
                  1.44962490,  1.44485891,  1.46043479,  1.90800595,  1.95863307,
                  1.63670933,  1.49021459,  1.18701911,  0.74906683,  0.84700620]
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(5)
        feature_extractor = ParakeetFeatureExtractor()
        inputs = feature_extractor(input_speech, return_tensors="pt")

        self.assertEqual(inputs.input_features.shape, (5, 2941, 80))
        torch.testing.assert_close(inputs.input_features[:, 100, :30], EXPECTED_INPUT_FEATURES, atol=1e-4, rtol=1e-4)

        self.assertEqual(inputs.attention_mask.shape, (5, 2941))
        self.assertTrue(inputs.attention_mask.sum(dim=-1).tolist(), [585, 481, 1248, 990, 2940])
