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
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


class PocketTTSFeatureExtractionTester:
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

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        if equal_length:
            audio_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            audio_inputs = [
                list(itertools.chain(*floats_list((x, self.feature_size))))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            audio_inputs = [np.asarray(x) for x in audio_inputs]
        return audio_inputs


@require_torch
class PocketTTSFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = PocketTTSFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = PocketTTSFeatureExtractionTester(self)

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 4000, 1000)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]

        # not batched: a list, a numpy array and a tensor give the same waveform
        expected = feature_extractor(np_audio_inputs[0], sampling_rate=24000)
        for audio in (audio_inputs[0], torch.tensor(np_audio_inputs[0])):
            torch.testing.assert_close(
                feature_extractor(audio, sampling_rate=24000).input_values, expected.input_values
            )

        # batched: (batch_size, audio_channels, num_samples), right-padded to a multiple of the frame size
        inputs = feature_extractor(np_audio_inputs, sampling_rate=24000)
        num_samples = inputs.input_values.shape[-1]
        self.assertEqual(inputs.input_values.shape, (len(audio_inputs), 1, num_samples))
        self.assertEqual(inputs.padding_mask.shape, (len(audio_inputs), num_samples))
        self.assertEqual(num_samples % feature_extractor.frame_size, 0)

    def test_end_on_pause(self):
        feature_extractor = self.feature_extraction_class()
        sampling_rate = feature_extractor.sampling_rate
        speech = 0.5 * torch.randn(int(0.2 * sampling_rate))
        waveform = torch.cat([speech, torch.zeros(int(0.1 * sampling_rate))])
        fade_length = int(feature_extractor.fade_out_duration * sampling_rate)
        pause_length = int(feature_extractor.pause_duration * sampling_rate)

        inputs = feature_extractor(waveform, sampling_rate=sampling_rate)
        num_samples = int(inputs.padding_mask.sum())
        output = inputs.input_values[0, 0, :num_samples]
        # the trailing silence is cut, the end of the speech faded out, and exactly `pause_duration` of zeros appended
        self.assertEqual(num_samples, len(speech) + pause_length)
        torch.testing.assert_close(output[: len(speech) - fade_length], speech[: len(speech) - fade_length])
        self.assertTrue(output[len(speech) - fade_length : len(speech)].abs().le(speech[-fade_length:].abs()).all())
        self.assertTrue(output[len(speech) :].eq(0).all())

        inputs = feature_extractor(waveform, sampling_rate=sampling_rate, end_on_pause=False)
        self.assertEqual(int(inputs.padding_mask.sum()), len(waveform))
        torch.testing.assert_close(inputs.input_values[0, 0, : len(waveform)], waveform)

    def test_get_num_frames(self):
        feature_extractor = self.feature_extraction_class()
        frame_size = feature_extractor.frame_size
        waveforms = [torch.randn(length) for length in (frame_size, frame_size + 1, 3 * frame_size - 1)]
        inputs = feature_extractor(waveforms, sampling_rate=24000, end_on_pause=False)
        self.assertEqual(feature_extractor.get_num_frames(inputs.padding_mask).tolist(), [1, 2, 3])

    def test_invalid_inputs(self):
        feature_extractor = self.feature_extraction_class()
        with self.assertRaises(ValueError):
            feature_extractor(np.zeros(1000), sampling_rate=16000)
        with self.assertRaises(ValueError):
            feature_extractor(np.zeros(1000), sampling_rate=24000, return_tensors="np")
        with self.assertRaises(ValueError):
            feature_extractor(np.zeros((2, 1000)), sampling_rate=24000)
