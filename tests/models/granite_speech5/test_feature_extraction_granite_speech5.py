# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the GraniteSpeech5 feature extraction."""

import itertools
import unittest

import numpy as np

from transformers import GraniteSpeech5FeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_datasets_available, is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

if is_datasets_available():
    from datasets import load_dataset


class GraniteSpeech5FeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        num_mel_bins=80,
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
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        # the front-end concatenates deltas and stacks frame pairs, so features are 4x the mel bins
        self.feature_size = 4 * num_mel_bins

    def prepare_feat_extract_dict(self):
        return {
            "num_mel_bins": self.num_mel_bins,
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


class GraniteSpeech5FeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = GraniteSpeech5FeatureExtractor

    def setUp(self):
        self.feat_extract_tester = GraniteSpeech5FeatureExtractionTester(self)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @require_torch
    def test_torch_integration(self):
        """Expected values were generated with the original `granite-speech-4.2-470m-turboctc` processor:

        ```python
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        inputs = processor(samples)
        ```
        """
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                0.46439588, 0.96522224, 1.08147550, 1.28836179, 1.26172411, 1.18971181,
                1.10675824, 1.14042044, 1.17948639, 1.09332263, 1.13654244, 1.28362978,
                1.27928782, 1.14827120, 1.30975986, 1.32958913, 1.19343317, 1.33287477,
                1.34555960, 1.12338209, 1.18367600, 1.13442707, 1.17981040, 1.22258615,
                1.00220931, 0.98556274, 0.84045565, 0.64106494, 0.57913423, 0.55850160,
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = GraniteSpeech5FeatureExtractor()
        inputs = feature_extractor(input_speech, sampling_rate=16000, return_tensors="pt")

        self.assertEqual(inputs.input_features.shape, (1, 293, 320))
        torch.testing.assert_close(inputs.input_features[0, 100, :30], EXPECTED_INPUT_FEATURES, atol=1e-4, rtol=1e-4)

        self.assertEqual(inputs.attention_mask.shape, (1, 293))
        self.assertEqual(inputs.attention_mask.sum(), 293)

    @require_torch
    def test_torch_integration_batch(self):
        """Expected values were generated with the original `granite-speech-4.2-470m-turboctc` processor:

        ```python
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        inputs = processor(samples)
        ```
        """
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [0.46439588, 0.96522224, 1.08147550, 1.28836179, 1.26172411, 1.18971181],
                [0.84482145, 1.03227293, 1.13553369, 1.41563487, 1.42796099, 1.36153793],
                [0.30556983, 0.33381593, 0.37241113, 0.38547039, 0.28719616, 0.21620500],
                [0.89753264, 1.06956792, 1.17057538, 1.32346106, 1.26157808, 1.19741154],
                [0.57288790, 0.85506272, 0.96651471, 1.14401937, 1.09785473, 1.03110588],
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(5)
        feature_extractor = GraniteSpeech5FeatureExtractor()
        inputs = feature_extractor(input_speech, sampling_rate=16000, return_tensors="pt")

        self.assertEqual(inputs.input_features.shape, (5, 1470, 320))
        torch.testing.assert_close(inputs.input_features[:, 100, :6], EXPECTED_INPUT_FEATURES, atol=1e-4, rtol=1e-4)

        self.assertEqual(inputs.attention_mask.shape, (5, 1470))
        self.assertEqual(inputs.attention_mask.sum(dim=-1).tolist(), [293, 241, 624, 495, 1470])
