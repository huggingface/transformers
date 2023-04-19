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
from transformers.trainer_utils import set_seed
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

    def test_integration_fusion_short_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -31.4897, -31.8836, -28.5246, -20.3667, -15.7675, -20.8252, -24.9852,
                    -24.3609, -26.7176, -35.1011, -29.9410, -28.8808, -32.9967, -27.1163,
                    -23.4414, -27.2408, -28.1140, -27.0278, -35.3932, -31.2363, -28.6563,
                    -30.5986, -27.0273, -25.5642, -26.5161, -27.3810, -21.4945, -23.1438,
                    -21.9563, -21.1530, -21.8027, -21.6166, -23.1705, -23.5455, -24.7405,
                    -25.0442, -28.8903, -37.4691, -42.7592, -37.9336, -37.4475, -36.7911,
                    -38.4222, -39.2670, -40.5971, -44.0499, -44.4566, -41.1936, -39.3091,
                    -33.0541, -26.2162, -20.3374, -16.0830, -19.7818, -27.3550, -31.5544,
                    -39.3470, -35.8299, -30.7186, -24.9793, -16.9644, -13.9974, -28.9275,
                    -32.8179
                ],
            ]
        )
        # fmt: on
        MEL_BIN = [963, 963, 963]
        input_speech = self._load_datasamples(1)
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, idx_in_mel in zip(
            ["repeat", "repeatpad", None], EXPECTED_INPUT_FEATURES, MEL_BIN
        ):
            input_features = feature_extractor(input_speech, return_tensors="pt", padding=padding).input_features
            self.assertEquals(input_features.shape, (1, 4, 1001, 64))
            self.assertTrue(torch.allclose(input_features[0, 0, idx_in_mel], EXPECTED_VALUES, atol=1e-4))
            self.assertTrue(torch.all(input_features[0, 0] == input_features[0, 1]))
            self.assertTrue(torch.all(input_features[0, 0] == input_features[0, 2]))
            self.assertTrue(torch.all(input_features[0, 0] == input_features[0, 3]))

    def test_integration_rand_trunc_short_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -46.6997, -40.5216, -33.9434, -34.9428, -41.0734, -40.8453, -46.1212,
                    -52.1113, -46.3411, -48.0314, -50.1488, -44.4313, -41.4034, -44.2689,
                    -49.8047, -44.9534, -45.7702, -54.6047, -53.9873, -49.5832, -47.8417,
                    -50.7548, -47.8888, -46.5488, -45.7829, -46.1124, -48.8562, -44.6606,
                    -41.1969, -46.9161, -41.7349, -42.9788, -43.2269, -43.1812, -45.2678,
                    -45.1584, -46.8845, -47.9311, -49.6091, -59.6006, -66.2816, -61.5206,
                    -61.2616, -60.2390, -62.2890, -63.2951, -65.0391, -68.6216, -69.1934,
                    -65.3913, -62.2194, -56.3727, -47.9670, -43.3525, -43.7095, -52.3037,
                    -57.8282, -65.8889, -62.2096, -55.2755, -49.6361, -41.6378, -48.2226,
                    -61.1353
                ],
            ]
        )
        # fmt: on
        MEL_BIN = [963, 963, 963]
        input_speech = self._load_datasamples(1)
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, idx_in_mel in zip(
            ["repeat", "repeatpad", None], EXPECTED_INPUT_FEATURES, MEL_BIN
        ):
            input_features = feature_extractor(
                input_speech, return_tensors="pt", truncation="rand_trunc", padding=padding
            ).input_features
            self.assertEquals(input_features.shape, (1, 1, 1001, 64))
            self.assertTrue(torch.allclose(input_features[0, 0, idx_in_mel], EXPECTED_VALUES, atol=1e-4))

    def test_integration_fusion_long_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -11.1830, -10.1894,  -8.6051,  -4.8578,  -1.3268,  -8.4606, -14.5453,
                     -9.2017,   0.5781,  16.2129,  14.8289,   3.6326,  -3.8794,  -6.5544,
                     -2.4408,   1.9531,   6.0967,   1.7590,  -7.6730,  -6.1571,   2.0052,
                     16.6694,  20.6447,  21.2145,  13.4972,  15.9043,  16.8987,   4.1766,
                     11.9428,  21.2372,  12.3016,   4.8604,   6.7241,   1.8543,   4.9235,
                      5.3188,  -0.9897,  -1.2416,  -6.5864,   2.9529,   2.9274,   6.4753,
                     10.2300,  11.2127,   3.4042,  -1.0055,  -6.0475,  -6.7524,  -3.9801,
                     -1.4434,   0.4740,  -0.1584,  -4.5457,  -8.5746,  -8.8428, -13.1475,
                     -9.6079,  -8.5798,  -4.1143,  -3.7966,  -7.1651,  -6.1517,  -8.0258,
                    -12.1486
                ],
                [
                    -10.2017,  -7.9924,  -5.9517,  -3.9372,  -1.9735,  -4.3130,  16.1647,
                     25.0592,  23.5532,  14.4974,  -7.0778, -10.2262,   6.4782,  20.3454,
                     19.4269,   1.7976, -16.5070,   4.9380,  12.3390,   6.9285, -13.6325,
                     -8.5298,   1.0839,  -5.9629,  -8.4812,   3.1331,  -2.0963, -16.6046,
                    -14.0070, -17.5707, -13.2080, -17.2168, -17.7770, -12.1111, -18.6184,
                    -17.1897, -13.9801, -12.0426, -23.5400, -25.6823, -23.5813, -18.7847,
                    -20.5473, -25.6458, -19.7585, -27.6007, -28.9276, -24.8948, -25.4458,
                    -22.2807, -19.6613, -19.2669, -15.7813, -19.6821, -24.3439, -22.2598,
                    -28.2631, -30.1017, -32.7646, -33.6525, -27.5639, -22.0548, -27.8054,
                    -29.6947
                ],
                [
                     -9.2083,  -7.2966,  -6.2097,  -7.9957,  -2.9279, -11.1844,  -6.1487,
                      5.0738,  19.2957,  21.4577,  14.6803,  -3.3148,  -6.3328,  -2.3537,
                      6.9511,  15.2963,  14.6618,   5.2078,  -0.0868,   1.1920,  18.1982,
                     20.8467,  10.8038,   2.2521,   7.6906,   7.7427,  -1.2541,  -5.0018,
                      0.9809,  -2.1582,  -5.4576,  -5.4758, -11.8883,  -9.0605,  -8.4639,
                     -9.9899,  -0.0543,  -5.1628,   0.0481,  -4.1505,  -4.8141,  -7.8235,
                     -9.0621, -10.1742,  -8.9596, -11.5377, -16.5596, -17.1852, -17.5027,
                    -20.9322, -23.9538, -25.2600, -25.3426, -27.4534, -26.8857, -22.7851,
                    -25.8286, -24.8395, -23.8889, -24.2093, -26.5415, -23.7280, -25.6849,
                    -22.3628
                ]
            ]
        )
        # fmt: on
        MEL_BIN = 963
        input_speech = torch.cat([torch.tensor(x) for x in self._load_datasamples(5)])
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, block_idx in zip(
            ["repeat", "repeatpad", None], EXPECTED_INPUT_FEATURES, [0, 1, 3]
        ):
            set_seed(987654321)
            input_features = feature_extractor(input_speech, return_tensors="pt", padding=padding).input_features
            self.assertEquals(input_features.shape, (1, 4, 1001, 64))
            self.assertTrue(torch.allclose(input_features[0, block_idx, MEL_BIN], EXPECTED_VALUES, atol=1e-4))

    def test_integration_rand_trunc_long_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -35.4022, -32.7555, -31.2004, -32.7764, -42.5770, -41.6339, -43.1630,
                    -44.5080, -44.3029, -48.9628, -39.5022, -39.2105, -43.1350, -43.2195,
                    -48.4894, -52.2344, -57.6891, -52.2228, -45.5155, -44.2893, -43.4697,
                    -46.6702, -43.7490, -40.4819, -42.7275, -46.3434, -46.8412, -41.2003,
                    -43.1681, -46.2948, -46.1925, -47.8333, -45.6812, -44.9182, -41.7786,
                    -43.3809, -44.3199, -42.8814, -45.4771, -46.7114, -46.9746, -42.7090,
                    -41.6057, -38.3965, -40.1980, -41.0263, -34.1256, -28.3289, -29.0201,
                    -30.4453, -29.5561, -30.1734, -25.9406, -19.0897, -15.8452, -20.1351,
                    -23.6515, -23.1194, -17.1845, -19.4399, -23.6527, -22.8768, -20.7279,
                    -22.7864
                ],
                [
                    -35.7719, -27.2566, -23.6964, -27.5521,   0.2510,   7.4391,   1.3917,
                    -13.3417, -28.1758, -17.0856,  -5.7723,  -0.8000,  -7.8832, -15.5548,
                    -30.5935, -24.7571, -13.7009, -10.3432, -21.2464, -24.8118, -19.4080,
                    -14.9779, -11.7991, -18.4485, -20.1982, -17.3652, -20.6328, -28.2967,
                    -25.7819, -21.8962, -28.5083, -29.5719, -30.2120, -35.7033, -31.8218,
                    -34.0408, -37.7744, -33.9653, -31.3009, -30.9063, -28.6153, -32.2202,
                    -28.5456, -28.8579, -32.5170, -37.9152, -43.0052, -46.4849, -44.0786,
                    -39.1933, -33.2757, -31.6313, -42.6386, -52.3679, -53.5785, -55.6444,
                    -47.0050, -47.6459, -56.6361, -60.6781, -61.5244, -55.8272, -60.4832,
                    -58.1897
                ],
                [
                    -38.2686, -36.6285, -32.5835, -35.1693, -37.7938, -37.4035, -35.3132,
                    -35.6083, -36.3609, -40.9472, -36.7846, -36.1544, -38.9076, -39.3618,
                    -35.4953, -34.2809, -39.9466, -39.7433, -34.8347, -37.5674, -41.5689,
                    -38.9161, -34.3947, -30.2924, -30.4841, -34.5831, -28.9261, -24.8849,
                    -31.2324, -27.1622, -27.2107, -25.9385, -30.1691, -30.9223, -23.9495,
                    -25.6047, -26.7119, -28.5523, -27.7481, -32.8427, -35.4650, -31.0399,
                    -31.2073, -30.5163, -22.9819, -20.8892, -19.2510, -24.7905, -28.9426,
                    -28.1998, -26.7386, -25.0140, -27.9223, -32.9913, -33.1864, -34.9742,
                    -38.5995, -39.6990, -29.3203, -22.4697, -25.6415, -33.5608, -33.0945,
                    -27.1716
                ]
            ]
        )
        # fmt: on
        MEL_BIN = 963
        SEEDS = [987654321, 1234, 666]
        input_speech = torch.cat([torch.tensor(x) for x in self._load_datasamples(5)])
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, seed in zip(
            ["repeat", "repeatpad", None], EXPECTED_INPUT_FEATURES, SEEDS
        ):
            set_seed(seed)
            input_features = feature_extractor(
                input_speech, return_tensors="pt", truncation="rand_trunc", padding=padding
            ).input_features
            self.assertEquals(input_features.shape, (1, 1, 1001, 64))
            self.assertTrue(torch.allclose(input_features[0, 0, MEL_BIN], EXPECTED_VALUES, atol=1e-4))
