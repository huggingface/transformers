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


import itertools
import random
import unittest

import numpy as np

from transformers import ClapFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio
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
@require_torchaudio
# Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTester with Whisper->Clap
class ClapFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=10,
        hop_length=160,
        chunk_length=8,
        padding_value=0.0,
        sampling_rate=4_000,
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
        self.chunk_length = chunk_length
        self.hop_length = hop_length

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "chunk_length": self.chunk_length,
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
@require_torchaudio
# Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest with Whisper->Clap
class ClapFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = ClapFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = ClapFeatureExtractionTester(self)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding="max_length", return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 4)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

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
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def integration_test_fusion(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -30.2194, -22.4424, -18.6442, -17.2452, -22.7392, -32.2576, -36.1404,
                    -35.6120, -29.6229, -29.0454, -32.2157, -36.7664, -29.4436, -26.7825,
                    -31.1811, -38.3918, -38.8749, -43.4485, -47.6236, -38.7528, -31.8574,
                    -39.0591, -41.3190, -32.3319, -31.4699, -33.4502, -36.7412, -34.5265,
                    -35.1091, -40.4518, -42.7346, -44.5909, -44.9747, -45.8328, -47.0772,
                    -46.2723, -44.3613, -48.6253, -44.9551, -43.8700, -44.6104, -48.0146,
                    -42.7614, -47.3587, -47.4369, -45.5018, -47.0198, -42.8759, -47.5056,
                    -47.1567, -49.2621, -49.5643, -48.4330, -48.8495, -47.2512, -40.8439,
                    -48.1234, -49.1218, -48.7222, -50.2399, -46.8487, -41.9921, -50.4015,
                    -50.7827
                ],
                [
                    -89.0141, -89.1411, -88.8096, -88.5480, -88.3481, -88.2038,
                    -88.1105, -88.0647, -88.0636, -88.1051, -88.1877, -88.1110,
                    -87.8613, -88.6679, -88.2685, -88.9684, -88.7977, -89.6264,
                    -89.9299, -90.3184, -91.1446, -91.9265, -92.7267, -93.6099,
                    -94.6395, -95.3243, -95.5923, -95.5773, -95.0889, -94.3354,
                    -93.5746, -92.9287, -92.4525, -91.9798, -91.8852, -91.7500,
                    -91.7259, -91.7561, -91.7959, -91.7070, -91.6914, -91.5019,
                    -91.0640, -90.0807, -88.7102, -87.0826, -85.5956, -84.4441,
                    -83.8461, -83.8605, -84.6702, -86.3900, -89.3073, -93.2926,
                    -96.3813, -97.3529, -100.0000, -99.6942, -92.2851, -87.9588,
                    -85.7214, -84.6807, -84.1940, -84.2021
                ],
                [
                    -51.6882, -50.6852, -50.8198, -51.7428, -53.0325, -54.1619, -56.4903,
                    -59.0314, -60.7996, -60.5164, -59.9680, -60.5393, -62.5796, -65.4166,
                    -65.6149, -65.1409, -65.7226, -67.9057, -72.5089, -82.3530, -86.3189,
                    -83.4241, -79.1279, -79.3384, -82.7335, -79.8316, -80.2167, -74.3638,
                    -71.3930, -75.3849, -74.5381, -71.4504, -70.3791, -71.4547, -71.8820,
                    -67.3885, -69.5686, -71.9852, -71.0307, -73.0053, -80.8802, -72.9227,
                    -63.8526, -60.3260, -59.6012, -57.8316, -61.0603, -67.3403, -67.1709,
                    -60.4967, -60.5079, -68.3345, -67.5213, -70.6416, -79.6219, -78.2198,
                    -74.6851, -69.5718, -69.4968, -70.6882, -66.8175, -73.8558, -74.3855,
                    -72.9405
                ]
            ]
        )
        # fmt: on
        MEL_BIN = [963, 963, 161]
        input_speech = self._load_datasamples(1)
        feaure_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, idx_in_mel in zip(
            ["repeat", "repeatpad", None], EXPECTED_INPUT_FEATURES, MEL_BIN
        ):
            input_features = feaure_extractor(input_speech, return_tensors="pt", padding=padding).input_features
            self.assertTrue(torch.allclose(input_features[0, idx_in_mel], EXPECTED_VALUES, atol=1e-4))

    def integration_test_rand_trunc(self):
        # TODO in this case we should set the seed and use a longer audio to properly see the random truncation
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -42.3330, -36.2735, -35.9231, -43.5947, -48.4525, -46.5227, -42.6477,
                    -47.2740, -51.4336, -50.0846, -51.8711, -50.4232, -47.4736, -54.2275,
                    -53.3947, -55.4904, -54.8750, -54.5510, -55.4156, -57.4395, -51.7385,
                    -55.9118, -57.7800, -63.2064, -67.0651, -61.4379, -56.4268, -54.8667,
                    -52.3487, -56.4418, -57.1842, -55.1005, -55.6366, -59.4395, -56.8604,
                    -56.4949, -61.6573, -61.0826, -60.3250, -63.7876, -67.4882, -60.2323,
                    -54.6886, -50.5369, -47.7656, -45.8909, -49.1273, -57.4141, -58.3201,
                    -51.9862, -51.4897, -59.2561, -60.4730, -61.2203, -69.3174, -69.7464,
                    -65.5861, -58.9921, -59.5610, -61.0584, -58.1149, -64.4045, -66.2622,
                    -64.4610
                ],
                [
                    -41.2298, -38.4211, -39.8834, -45.9950, -47.3839, -43.9849, -46.0371,
                    -52.5490, -56.6912, -51.8794, -50.1284, -49.7506, -53.9422, -63.2854,
                    -56.5754, -55.0469, -55.3181, -55.8115, -56.0058, -57.9215, -58.7597,
                    -59.1994, -59.2141, -64.4198, -73.5138, -64.4647, -59.3351, -54.5626,
                    -54.7508, -65.0230, -60.0270, -54.7644, -56.0108, -60.1531, -57.6879,
                    -56.3766, -63.3395, -65.3032, -61.5202, -63.0677, -68.4217, -60.6868,
                    -54.4619, -50.8533, -47.7200, -45.9197, -49.0961, -57.7621, -59.0750,
                    -51.9122, -51.4332, -59.4132, -60.3415, -61.6558, -70.7049, -69.7905,
                    -66.9104, -59.0324, -59.6138, -61.2023, -58.2169, -65.3837, -66.4425,
                    -64.4142
                ],
                [
                    -51.6882, -50.6852, -50.8198, -51.7428, -53.0325, -54.1619, -56.4903,
                    -59.0314, -60.7996, -60.5164, -59.9680, -60.5393, -62.5796, -65.4166,
                    -65.6149, -65.1409, -65.7226, -67.9057, -72.5089, -82.3530, -86.3189,
                    -83.4241, -79.1279, -79.3384, -82.7335, -79.8316, -80.2167, -74.3638,
                    -71.3930, -75.3849, -74.5381, -71.4504, -70.3791, -71.4547, -71.8820,
                    -67.3885, -69.5686, -71.9852, -71.0307, -73.0053, -80.8802, -72.9227,
                    -63.8526, -60.3260, -59.6012, -57.8316, -61.0603, -67.3403, -67.1709,
                    -60.4967, -60.5079, -68.3345, -67.5213, -70.6416, -79.6219, -78.2198,
                    -74.6851, -69.5718, -69.4968, -70.6882, -66.8175, -73.8558, -74.3855,
                    -72.9405
                ]
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feaure_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES in zip(["repeat", "repeatpad", None], EXPECTED_INPUT_FEATURES):
            input_features = feaure_extractor(
                input_speech, return_tensors="pt", truncation="rand_trunc", padding=padding
            ).input_features
            self.assertTrue(torch.allclose(input_features[0, 0, :30], EXPECTED_VALUES, atol=1e-4))
