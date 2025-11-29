# Copyright 2024 HuggingFace Inc.
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
"""Tests for the Xcodec2 feature extractor."""

import itertools
import json
import random
import unittest
from pathlib import Path

import numpy as np

from transformers import Xcodec2FeatureExtractor
from transformers.testing_utils import require_torch
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
class Xcodec2FeatureExtractionTester:
    # Ignore copy
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=80,  # number of mel bins
        n_channels=1,
        sampling_rate=16000,
        spec_hop_length=160,
        hop_length=320,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.spec_hop_length = spec_hop_length
        self.hop_length = hop_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate

    # Ignore copy
    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "hop_length": self.hop_length,
            "spec_hop_length": self.spec_hop_length,
            "n_channels": self.n_channels,
            # Model usses 1.0, but 0.0 needed to pass `test_padding_from_array`
            # -- SeamlessM4TFeatureExtractionTester does the same
            "padding_value": 0.0,
        }

    # Copied from transformers.tests.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTester.prepare_inputs_for_common
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
class Xcodec2FeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Xcodec2FeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Xcodec2FeatureExtractionTester(self)

    def test_call(self):
        TOL = 1e-6

        # Tests that all call wrap to encode_plus and batch_encode_plus
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]
        torch_audio_inputs = [torch.tensor(audio_input) for audio_input in audio_inputs]

        # Test not batched input
        sampling_rate = self.feat_extract_tester.sampling_rate
        encoded_sequences_1 = feat_extract(torch_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np")
        encoded_sequences_2 = feat_extract(np_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np")
        # -- use torch for mel features
        encoded_sequences_3 = feat_extract(
            torch_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np", use_torch=True, device="cpu"
        )
        self.assertTrue(np.allclose(encoded_sequences_1.audio, encoded_sequences_2.audio, atol=TOL))
        self.assertTrue(
            np.allclose(encoded_sequences_1.audio_spectrogram, encoded_sequences_2.audio_spectrogram, atol=TOL)
        )
        self.assertTrue(
            np.allclose(encoded_sequences_1.audio_spectrogram, encoded_sequences_3.audio_spectrogram, atol=TOL)
        )

        # Test using GPU
        if torch.cuda.is_available():
            encoded_sequences_3 = feat_extract(
                torch_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np", use_torch=True, device="cuda"
            )
            self.assertTrue(np.allclose(encoded_sequences_1.audio, encoded_sequences_3.audio, atol=TOL))
            self.assertTrue(
                np.allclose(encoded_sequences_1.audio_spectrogram, encoded_sequences_3.audio_spectrogram, atol=TOL)
            )

        # Test batched
        encoded_sequences_1 = feat_extract(
            torch_audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="np"
        )
        encoded_sequences_2 = feat_extract(
            np_audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="np"
        )
        encoded_sequences_3 = feat_extract(
            torch_audio_inputs,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="np",
            use_torch=True,
            device="cpu",
        )
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1.audio, encoded_sequences_2.audio):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=TOL))
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1.audio_spectrogram, encoded_sequences_2.audio_spectrogram):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=TOL))
        for enc_seq_1, enc_seq_3 in zip(encoded_sequences_1.audio_spectrogram, encoded_sequences_3.audio_spectrogram):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_3, atol=TOL))

    def test_double_precision_pad(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_audio_inputs = np.random.rand(100, 32).astype(np.float64)
        py_audio_inputs = np_audio_inputs.tolist()

        for inputs in [py_audio_inputs, np_audio_inputs]:
            np_processed = feature_extractor.pad([{"audio_spectrogram": inputs}], return_tensors="np")
            self.assertTrue(np_processed.audio_spectrogram.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"audio_spectrogram": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.audio_spectrogram.dtype == torch.float32)

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        audio_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in audio_samples]

    def test_integration(self):
        # Reproducer: https://gist.github.com/ebezzam/909243aa00934cab1c1aad3d6a161580#file-xcodec2_feature_extraction-py
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/xcodec2/expected_results_feature_extractor.json"
        with open(RESULTS_PATH, "r") as f:
            expected_results = json.load(f)
        # expected_results = expected_results[0]
        EXPECTED_INPUT_VALUES = torch.tensor(expected_results["audio_spectrogram"])
        EXPECTED_SHAPE = tuple(expected_results["shape"])

        input_audio = self._load_datasamples(1)
        feature_extractor = Xcodec2FeatureExtractor()
        input_values = feature_extractor(input_audio, return_tensors="pt")["audio_spectrogram"]
        self.assertEqual(input_values.shape, EXPECTED_SHAPE)
        torch.testing.assert_close(input_values[0, 0, : EXPECTED_INPUT_VALUES.shape[-1]], EXPECTED_INPUT_VALUES)

    # Ignore copy
    @unittest.skip("Xcodec2 doesn't support stereo input")
    def test_integration_stereo(self):
        pass
