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

from transformers import OmniASRFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


@require_torch
class OmniASRFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=16000,
        return_attention_mask=True,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
        }

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
class OmniASRFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = OmniASRFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = OmniASRFeatureExtractionTester(self)

    def _get_feat_extract(self, **kwargs):
        feat_extract_dict = self.feat_extract_tester.prepare_feat_extract_dict()
        feat_extract_dict.update(kwargs)
        return self.feature_extraction_class(**feat_extract_dict)

    def test_call(self):
        TOL = 1e-5
        feat_extract = self._get_feat_extract()
        sampling_rate = self.feat_extract_tester.sampling_rate

        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]
        torch_audio_inputs = [torch.tensor(audio_input) for audio_input in audio_inputs]

        # Not batched: a list of floats, a numpy array and a torch tensor all give the same features
        encoded_list = feat_extract(list(audio_inputs[0]), sampling_rate=sampling_rate)
        encoded_np = feat_extract(np_audio_inputs[0], sampling_rate=sampling_rate)
        encoded_pt = feat_extract(torch_audio_inputs[0], sampling_rate=sampling_rate)
        self.assertEqual(encoded_pt.input_values.shape, (1, len(audio_inputs[0])))
        self.assertEqual(encoded_pt.input_values.dtype, torch.float32)
        torch.testing.assert_close(encoded_list.input_values, encoded_np.input_values, atol=TOL, rtol=TOL)
        torch.testing.assert_close(encoded_pt.input_values, encoded_np.input_values, atol=TOL, rtol=TOL)

        # Batched, padded to the longest input
        encoded_np = feat_extract(list(np_audio_inputs), sampling_rate=sampling_rate)
        encoded_pt = feat_extract(list(torch_audio_inputs), sampling_rate=sampling_rate)
        self.assertEqual(encoded_pt.input_values.shape, (3, len(audio_inputs[-1])))
        torch.testing.assert_close(encoded_pt.input_values, encoded_np.input_values, atol=TOL, rtol=TOL)

        # The mask marks the padding, so the model knows which frames are real
        self.assertListEqual(encoded_pt.padding_mask.sum(-1).tolist(), [len(x) for x in audio_inputs])
        self.assertTrue(torch.all(encoded_pt.input_values[0, len(audio_inputs[0]) :] == 0.0))
