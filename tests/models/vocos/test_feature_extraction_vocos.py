# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
from unittest.mock import patch

import numpy as np
from datasets import Audio, load_dataset

from transformers import VocosFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_torchaudio
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

global_rng = random.Random()


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
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
class VocosFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=10,
        n_fft=512,
        hop_length=128,
        padding_value=0.0,
        sampling_rate=4000,
        padding="center",
        return_attention_mask=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.return_attention_mask = return_attention_mask

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "num_mel_bins": self.feature_size,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "padding": self.padding,
            "padding_value": self.padding_value,
            "return_attention_mask": self.return_attention_mask,
        }

    # Copied from transformers.tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTester.prepare_inputs_for_common
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


@require_torchaudio
@require_torch
class VocosFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = VocosFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = VocosFeatureExtractionTester(self)
        self.EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                -1.7733538150787354,
                -2.6277995109558105,
                -4.34464168548584,
                -3.113943099975586,
                -2.854297637939453,
                -2.973206043243408,
                -2.4957544803619385,
                -2.252488851547241,
                -2.255256175994873,
                -2.6200737953186035,
                -2.978632926940918,
                -2.915842056274414,
                -3.6588683128356934,
                -2.4617226123809814,
                -2.5469284057617188,
                -2.164539098739624,
                -2.7375283241271973,
                -2.8342182636260986,
                -3.9451725482940674,
                -3.013962745666504,
                -3.843592643737793,
                -3.688326358795166,
                -3.601658821105957,
                -3.2761828899383545,
                -3.0132853984832764,
                -2.340486526489258,
                -2.8324050903320312,
                -2.718107223510742,
                -2.8961753845214844,
                -2.846886396408081,
            ]
        )

    # Copied from transformers.tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_feat_extract_from_and_save_pretrained
    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    # Copied from transformers.tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_feat_extract_to_json_file
    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
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

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    # Copied from transformers.tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_double_precision_pad
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

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @require_torch
    def test_integration_torch_backend(self):
        speech = self._load_datasamples(1)
        feature_extractor = VocosFeatureExtractor.from_pretrained("Manel/vocos-mel-24khz")
        input_features = feature_extractor(speech, return_tensors="pt").input_features
        self.assertEqual(input_features.shape, (1, 100, 549))
        torch.testing.assert_close(input_features[0, 0, :30], self.EXPECTED_INPUT_FEATURES, rtol=1e-6, atol=1e-6)

    @patch("transformers.models.vocos.feature_extraction_vocos.is_torch_available", return_value=False)
    def test_integration_numpy_backend(self, _mock_torch_avail):
        speech = self._load_datasamples(1)
        feature_extractor = VocosFeatureExtractor.from_pretrained("Manel/vocos-mel-24khz")
        input_features = feature_extractor(speech, return_tensors="pt").input_features
        self.assertEqual(input_features.shape, (1, 100, 549))

        torch.testing.assert_close(input_features[0, 0, :30], self.EXPECTED_INPUT_FEATURES, rtol=1e-6, atol=1e-6)
