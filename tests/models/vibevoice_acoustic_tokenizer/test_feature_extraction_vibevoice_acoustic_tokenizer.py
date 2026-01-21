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
import random
import unittest

import numpy as np

from transformers import VibeVoiceAcousticTokenizerFeatureExtractor
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
class VibeVoiceAcousticTokenizerFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=24000,
        normalize_audio=True,
        target_dB_FS=-25,
        eps=1e-6,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "normalize_audio": self.normalize_audio,
            "target_dB_FS": self.target_dB_FS,
            "eps": self.eps,
        }

    # Copied from tests.models.encodec.test_feature_extraction_encodec.EnCodecFeatureExtractionTester.prepare_inputs_for_common
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
class VibeVoiceAcousticTokenizerFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = VibeVoiceAcousticTokenizerFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = VibeVoiceAcousticTokenizerFeatureExtractionTester(self)

    def test_call(self):
        TOL = 1e-6

        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = feature_extractor.sampling_rate
        # create three inputs of length 800, 1000, and 1200
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]
        torch_audio_inputs = [torch.tensor(audio_input) for audio_input in audio_inputs]

        # Test not batched input
        encoded_sequences_1 = feature_extractor(
            torch_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np"
        ).input_values
        encoded_sequences_2 = feature_extractor(
            np_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np"
        ).input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=TOL))

        # Test batched
        encoded_sequences_1 = feature_extractor(
            torch_audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="np"
        ).input_values
        encoded_sequences_2 = feature_extractor(
            np_audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="np"
        ).input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=TOL))

    def test_double_precision_pad(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_audio_inputs = np.random.rand(100).astype(np.float64)
        py_audio_inputs = np_audio_inputs.tolist()

        for inputs in [py_audio_inputs, np_audio_inputs]:
            np_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_values.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_values.dtype == torch.float32)

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        audio_samples = ds.sort("id")[:num_samples]["input_values"]

        return [x["array"] for x in audio_samples]

    def test_normalize_audio(self):
        """Test audio normalization functionality specific to VibeVoice."""
        # Test with normalization enabled (default)
        feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor(normalize_audio=True, target_dB_FS=-25)

        # Test with very low amplitude audio (should increase amplitude)
        low_amplitude_audio = np.random.randn(1000).astype(np.float32) * 0.01
        result = feature_extractor([low_amplitude_audio], return_tensors="pt")
        normalized_audio = result.input_values.squeeze()
        self.assertGreater(
            torch.abs(normalized_audio).max().item(), torch.abs(torch.tensor(low_amplitude_audio)).max().item()
        )

        # Test with normalization disabled (should be close to original)
        feature_extractor_no_norm = VibeVoiceAcousticTokenizerFeatureExtractor(normalize_audio=False)
        result_no_norm = feature_extractor_no_norm([low_amplitude_audio], return_tensors="pt")
        torch.testing.assert_close(
            result_no_norm.input_values.squeeze(), torch.tensor(low_amplitude_audio), rtol=1e-5, atol=1e-5
        )

    def test_sampling_rate_validation(self):
        """Test that sampling rate validation works correctly."""
        feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor(sampling_rate=24000)
        input_audio = np.random.randn(1000).astype(np.float32)

        # Should work with correct sampling rate
        result = feature_extractor([input_audio], sampling_rate=24000, return_tensors="pt")
        self.assertIsInstance(result.input_values, torch.Tensor)

        # Should raise error with incorrect sampling rate
        with self.assertRaises(ValueError):
            feature_extractor([input_audio], sampling_rate=16000, return_tensors="pt")

    def test_padding_mask_generation(self):
        """Test that padding masks are generated correctly."""
        feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor()

        # Create audio samples of different lengths
        audio1 = np.random.randn(100).astype(np.float32)
        audio2 = np.random.randn(200).astype(np.float32)

        result = feature_extractor([audio1, audio2], padding=True, return_tensors="pt", return_attention_mask=True)

        # Should have padding_mask
        self.assertIn("padding_mask", result)
        self.assertEqual(result.padding_mask.shape, result.input_values.squeeze(1).shape)

        # First sample should have some padding (False values at the end)
        self.assertTrue(torch.any(~result.padding_mask[0]))
        # Second sample should have no padding (all True values)
        self.assertTrue(torch.all(result.padding_mask[1]))