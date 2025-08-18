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

import numpy as np
from datasets import load_dataset

from transformers import Phi4MultimodalFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

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


class Phi4MultimodalFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=80,
        hop_length=160,
        win_length=400,
        padding_value=0.0,
        sampling_rate=16_000,
        return_attention_mask=False,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.feature_size = feature_size
        self.win_length = win_length
        self.hop_length = hop_length

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
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


class Phi4MultimodalFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Phi4MultimodalFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Phi4MultimodalFeatureExtractionTester(self)

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

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]
        pt_speech_inputs = [torch.tensor(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, return_tensors="np").audio_input_features
        max_audio_len = (1200 - feature_extractor.win_length) // feature_extractor.hop_length + 1
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(input_features.shape[-2] == max_audio_len)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(pt_speech_inputs[0], return_tensors="np").audio_input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors="np").audio_input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feature_extractor(pt_speech_inputs, return_tensors="np").audio_input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").audio_input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        pt_speech_inputs = torch.tensor(speech_inputs)
        encoded_sequences_1 = feature_extractor(pt_speech_inputs, return_tensors="np").audio_input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").audio_input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    @require_torch
    def test_double_precision_pad(self):
        import torch

        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"audio_input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.audio_input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"audio_input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.audio_input_features.dtype == torch.float32)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @require_torch
    def test_torch_integration(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                6.5243,  7.2267,  8.0917,  8.0041,  6.8247,  6.3216,  5.9599,  5.6770,
                5.7441,  5.6138,  6.6793,  6.8597,  5.5375,  6.5330,  5.4880,  7.3280,
                9.0736,  9.7665,  9.8773, 10.0828, 10.0518, 10.1736, 10.0145,  9.2545,
                11.0495, 11.6518, 10.8654, 10.2293,  9.1045,  9.4819,
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = Phi4MultimodalFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").audio_input_features

        self.assertEqual(input_features.shape, (1, 584, 80))
        torch.testing.assert_close(input_features[0, 0, :30], EXPECTED_INPUT_FEATURES, rtol=1e-4, atol=1e-4)

    @unittest.mock.patch(
        "transformers.models.phi4_multimodal.feature_extraction_phi4_multimodal.is_torch_available", lambda: False
    )
    def test_numpy_integration(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = np.array(
            [
                6.5242944,  7.226712,   8.091721,   8.004097,   6.824679,   6.3216243,
                5.959894,   5.676975,   5.744051,   5.61384,    6.6793485,  6.8597484,
                5.5374746,  6.532976,   5.4879804,  7.3279905,  9.073576,   9.766463,
                9.877262,  10.082759,  10.051792,  10.173581,  10.0144825,  9.254548,
                11.049487,  11.651841,  10.865354,  10.229329,   9.104464,   9.481946,
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = Phi4MultimodalFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="np").audio_input_features
        self.assertEqual(input_features.shape, (1, 584, 80))
        self.assertTrue(np.allclose(input_features[0, 0, :30], EXPECTED_INPUT_FEATURES, atol=1e-4))

    @require_torch
    def test_torch_integration_batch(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    6.5243,  7.2267,  8.0917,  8.0041,  6.8247,  6.3216,  5.9599,  5.6770,
                    5.7441,  5.6138,  6.6793,  6.8597,  5.5375,  6.5330,  5.4880,  7.3280,
                    9.0736,  9.7665,  9.8773, 10.0828, 10.0518, 10.1736, 10.0145,  9.2545,
                    11.0495, 11.6518, 10.8654, 10.2293,  9.1045,  9.4819
                ],
                [
                    7.5105,  7.9453,  8.6161,  7.7666,  7.2572,  6.8823,  6.3242,  6.1899,
                    6.9706,  8.0810,  7.3227,  5.8580,  5.4990,  7.7373,  8.5447,  7.7203,
                    6.3230,  7.1995,  7.1463,  7.3153,  7.4054,  7.2855,  6.9396,  7.0255,
                    7.3285,  7.2748,  8.0742,  7.3998,  6.4813,  6.7509
                ],
                [
                    7.7932,  8.1604,  8.7653,  8.2080,  7.2630,  6.4537,  4.8394,  6.3153,
                    8.0207,  8.3379,  6.0896,  5.7369,  5.8601,  4.7598,  4.8850,  6.2529,
                    3.9354,  6.1577,  7.9921,  9.6577, 10.1449,  9.1414,  9.3361,  9.0022,
                    9.2533, 10.0548, 10.4372,  8.8550,  9.1266,  9.9013
                ]
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(3)
        feature_extractor = Phi4MultimodalFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").audio_input_features
        self.assertEqual(input_features.shape, (3, 1247, 80))
        print(input_features[:, 0, :30])
        torch.testing.assert_close(input_features[:, 0, :30], EXPECTED_INPUT_FEATURES, rtol=1e-4, atol=1e-4)
