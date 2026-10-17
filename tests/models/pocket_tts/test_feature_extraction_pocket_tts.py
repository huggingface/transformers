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

from transformers import PocketTTSFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


class PocketTTSFeatureExtractionTester:
    def __init__(
        self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, feature_size=1, sampling_rate=24000
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        return {"feature_size": self.feature_size, "sampling_rate": self.sampling_rate, "padding_value": 0.0}

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        if equal_length:
            audio_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            audio_inputs = [
                list(itertools.chain(*floats_list((length, self.feature_size))))
                for length in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            audio_inputs = [np.asarray(audio) for audio in audio_inputs]
        return audio_inputs


@require_torch
class PocketTTSFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = PocketTTSFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = PocketTTSFeatureExtractionTester(self)

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = feature_extractor.sampling_rate
        audio_inputs = [floats_list((1, length))[0] for length in range(800, 1400, 200)]
        np_inputs = [np.asarray(audio, dtype=np.float32) for audio in audio_inputs]
        torch_inputs = [torch.tensor(audio, dtype=torch.float32) for audio in audio_inputs]

        # numpy and torch inputs, single and batched, give the same values
        single_np = feature_extractor(np_inputs[0], sampling_rate=sampling_rate)
        single_torch = feature_extractor(torch_inputs[0], sampling_rate=sampling_rate)
        torch.testing.assert_close(single_np["input_values"], single_torch["input_values"])
        batched = feature_extractor(np_inputs, sampling_rate=sampling_rate)
        self.assertEqual(batched["input_values"].shape[-1] % feature_extractor.pad_to_multiple_of, 0)
        self.assertListEqual(
            batched["padding_mask"].sum(-1).tolist(),
            [len(feature_extractor._end_on_pause(audio)) for audio in torch_inputs],
        )

    def test_end_on_pause(self):
        feature_extractor = PocketTTSFeatureExtractor()
        sampling_rate = feature_extractor.sampling_rate
        # 0.5 s of signal followed by 1 s of silence: the silence is cut, 80 ms of pause appended
        audio = torch.cat([torch.randn(sampling_rate // 2) * 0.1, torch.zeros(sampling_rate)])
        trimmed = feature_extractor(audio, sampling_rate=sampling_rate)
        kept = feature_extractor(audio, sampling_rate=sampling_rate, end_on_pause=False)
        expected_length = sampling_rate // 2 + int(feature_extractor.pause_duration * sampling_rate)
        self.assertEqual(trimmed["padding_mask"].sum().item(), expected_length)
        self.assertEqual(kept["padding_mask"].sum().item(), audio.shape[0])
        # the end of the kept signal is faded out, then the pause is exact zeros
        self.assertEqual(trimmed["input_values"][0, 0, sampling_rate // 2 - 1].item(), 0.0)
        self.assertTrue((trimmed["input_values"][0, 0, sampling_rate // 2 :] == 0).all())
        self.assertEqual(trimmed["input_values"].shape[-1] % feature_extractor.pad_to_multiple_of, 0)

    def test_sampling_rate_mismatch(self):
        with self.assertRaises(ValueError):
            PocketTTSFeatureExtractor()(torch.zeros(1000), sampling_rate=16000)
