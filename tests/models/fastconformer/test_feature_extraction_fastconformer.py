# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the FastConformer feature extraction."""

import itertools
import random
import tempfile
import unittest

import numpy as np

from transformers.models.fastconformer import FastConformerFeatureExtractor
from transformers.testing_utils import require_torch, slow
from transformers.utils import is_torch_available


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


class FastConformerFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=128,
        hop_length=160,
        win_length=400,
        n_fft=512,
        sampling_rate=16000,
        padding_value=0.0,
        normalize="per_feature",
        return_attention_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.normalize = normalize
        self.return_attention_mask = return_attention_mask

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "normalize": self.normalize,
            "return_attention_mask": self.return_attention_mask,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = [_flatten(floats_list((1, self.max_seq_length))) for _ in range(self.batch_size)]
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                _flatten(floats_list((1, x))) for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]

        return speech_inputs


class FastConformerFeatureExtractionTest(unittest.TestCase):
    feature_extraction_class = FastConformerFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = FastConformerFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_common_properties(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feat_extract, "feature_size"))
        self.assertTrue(hasattr(feat_extract, "sampling_rate"))
        self.assertTrue(hasattr(feat_extract, "hop_length"))

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_to_json_string(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        json_string = feat_extract_first.to_json_string()
        # Test that json_string is valid JSON and contains expected keys
        import json
        parsed_dict = json.loads(json_string)
        self.assertIn("feature_size", parsed_dict)
        self.assertIn("sampling_rate", parsed_dict)
        self.assertEqual(parsed_dict["feature_size"], feat_extract_first.feature_size)
        self.assertEqual(parsed_dict["sampling_rate"], feat_extract_first.sampling_rate)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = f"{tmpdirname}/feat_extract.json"
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_init_without_params(self):
        feat_extract = self.feature_extraction_class()
        self.assertIsNotNone(feat_extract)

    @require_torch
    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [list(itertools.chain(*floats_list((1, x)))) for x in range(800, 1400, 200)]

        # Test batched input with torch tensors
        if is_torch_available():
            # Convert to torch tensors
            torch_speech_inputs = [torch.tensor(speech_input, dtype=torch.float32) for speech_input in speech_inputs]
            
            # Test single input
            encoded_sequence = feat_extract(torch_speech_inputs[0].unsqueeze(0), return_tensors="pt")
            self.assertTrue("input_features" in encoded_sequence)
            self.assertTrue("attention_mask" in encoded_sequence)
            self.assertTrue("input_lengths" in encoded_sequence)
            
            # Test batched - pad to same length and create batch tensor
            max_length = max(len(inp) for inp in torch_speech_inputs)
            padded_inputs = []
            audio_lengths = []
            
            for inp in torch_speech_inputs:
                audio_lengths.append(len(inp))
                if len(inp) < max_length:
                    padded = torch.cat([inp, torch.zeros(max_length - len(inp))])
                else:
                    padded = inp
                padded_inputs.append(padded)
            
            batch_tensor = torch.stack(padded_inputs)
            lengths_tensor = torch.tensor(audio_lengths, dtype=torch.long)
            
            encoded_sequences = feat_extract(batch_tensor, audio_lengths=lengths_tensor, return_tensors="pt")
            
            self.assertTrue("input_features" in encoded_sequences)
            self.assertTrue("attention_mask" in encoded_sequences)
            self.assertTrue("input_lengths" in encoded_sequences)
            self.assertEqual(encoded_sequences.input_features.shape[0], len(torch_speech_inputs))

    @require_torch
    def test_torch_extraction(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        
        # Create test audio
        speech_inputs = [list(itertools.chain(*floats_list((1, x)))) for x in range(800, 1400, 200)]
        torch_speech_inputs = [torch.tensor(speech_input, dtype=torch.float32) for speech_input in speech_inputs]
        
        # Test single input
        batch_audio = torch_speech_inputs[0].unsqueeze(0)
        audio_lengths = torch.tensor([len(torch_speech_inputs[0])], dtype=torch.long)
        
        features = feat_extract(batch_audio, audio_lengths=audio_lengths, return_tensors="pt")
        
        # Check output format
        self.assertEqual(features.input_features.dim(), 3)  # (batch, time, features)
        self.assertEqual(features.input_features.shape[2], feat_extract.feature_size)
        self.assertEqual(features.attention_mask.dim(), 2)  # (batch, time)
        self.assertEqual(features.input_lengths.dim(), 1)  # (batch,)

    @require_torch
    def test_mel_spectrogram_properties(self):
        feat_extract = self.feature_extraction_class(
            feature_size=80,
            sampling_rate=16000,
            hop_length=160,
            win_length=400,
            n_fft=512,
        )
        
        # Create test signal - 1 second of audio
        duration = 1.0
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        
        # Create a simple sine wave
        t = torch.linspace(0, duration, num_samples)
        frequency = 440  # A4 note
        audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        audio_lengths = torch.tensor([num_samples], dtype=torch.long)
        
        features = feat_extract(audio, audio_lengths=audio_lengths, return_tensors="pt")
        
        # Check that we get expected number of mel bins
        self.assertEqual(features.input_features.shape[2], 80)
        
        # Check that we get reasonable number of time frames
        expected_frames = feat_extract.get_seq_len(num_samples, feat_extract.n_fft, feat_extract.hop_length)
        self.assertEqual(features.input_features.shape[1], expected_frames)

    @require_torch
    def test_normalization_types(self):
        # Test per_feature normalization
        feat_extract_per_feature = self.feature_extraction_class(
            normalize="per_feature",
            feature_size=80,
        )
        
        # Test all_features normalization
        feat_extract_all_features = self.feature_extraction_class(
            normalize="all_features",
            feature_size=80,
        )
        
        # Create test audio
        audio = torch.randn(1, 8000)  # 0.5 seconds at 16kHz
        audio_lengths = torch.tensor([8000], dtype=torch.long)
        
        features_per = feat_extract_per_feature(audio, audio_lengths=audio_lengths, return_tensors="pt")
        features_all = feat_extract_all_features(audio, audio_lengths=audio_lengths, return_tensors="pt")
        
        # Both should have the same shape
        self.assertEqual(features_per.input_features.shape, features_all.input_features.shape)
        
        # But different values due to different normalization
        self.assertFalse(torch.allclose(features_per.input_features, features_all.input_features))

    @require_torch
    def test_attention_mask_computation(self):
        feat_extract = self.feature_extraction_class()
        
        # Create inputs of different lengths
        short_audio = torch.randn(1, 4000)  # 0.25 seconds
        long_audio = torch.randn(1, 12000)  # 0.75 seconds
        
        # Pad to same length
        max_length = 12000
        padded_short = torch.cat([short_audio, torch.zeros(1, max_length - 4000)], dim=1)
        
        batch_audio = torch.cat([padded_short, long_audio], dim=0)
        audio_lengths = torch.tensor([4000, 12000], dtype=torch.long)
        
        features = feat_extract(batch_audio, audio_lengths=audio_lengths, return_tensors="pt")
        
        # Check attention mask
        attention_mask = features.attention_mask
        
        # First sequence should have fewer valid frames
        valid_frames_0 = attention_mask[0].sum().item()
        valid_frames_1 = attention_mask[1].sum().item()
        
        self.assertLess(valid_frames_0, valid_frames_1)
        
        # Check that input_lengths match attention mask sums
        self.assertEqual(features.input_lengths[0].item(), valid_frames_0)
        self.assertEqual(features.input_lengths[1].item(), valid_frames_1)

    @require_torch
    def test_preemphasis(self):
        # Test with and without preemphasis
        feat_extract_with_preemph = self.feature_extraction_class(preemph=0.97)
        feat_extract_without_preemph = self.feature_extraction_class(preemph=0.0)
        
        audio = torch.randn(1, 8000)
        audio_lengths = torch.tensor([8000], dtype=torch.long)
        
        features_with = feat_extract_with_preemph(audio, audio_lengths=audio_lengths, return_tensors="pt")
        features_without = feat_extract_without_preemph(audio, audio_lengths=audio_lengths, return_tensors="pt")
        
        # Should produce different results
        self.assertFalse(torch.allclose(features_with.input_features, features_without.input_features))

    @require_torch
    def test_device_placement(self):
        feat_extract = self.feature_extraction_class()
        
        audio = torch.randn(1, 8000)
        audio_lengths = torch.tensor([8000], dtype=torch.long)
        
        # Test default (CPU)
        features_cpu = feat_extract(audio, audio_lengths=audio_lengths, return_tensors="pt")
        self.assertEqual(features_cpu.input_features.device.type, "cpu")
        
        # Test explicit device parameter
        features_cpu_explicit = feat_extract(audio, audio_lengths=audio_lengths, return_tensors="pt", device="cpu")
        self.assertEqual(features_cpu_explicit.input_features.device.type, "cpu")

    @require_torch
    def test_different_window_functions(self):
        # Test different window functions
        window_types = ["hann", "hamming", "blackman", "bartlett"]
        
        audio = torch.randn(1, 8000)
        audio_lengths = torch.tensor([8000], dtype=torch.long)
        
        features_list = []
        for window in window_types:
            feat_extract = self.feature_extraction_class(window=window)
            features = feat_extract(audio, audio_lengths=audio_lengths, return_tensors="pt")
            features_list.append(features.input_features)
        
        # All should have the same shape
        for features in features_list[1:]:
            self.assertEqual(features.shape, features_list[0].shape)
        
        # But different values due to different windows
        for i, features in enumerate(features_list[1:], 1):
            self.assertFalse(torch.allclose(features, features_list[0], atol=1e-5))

    def test_invalid_normalize_parameter(self):
        """Test that invalid normalize parameter raises ValueError."""
        with self.assertRaises(ValueError):
            FastConformerFeatureExtractor(normalize="invalid_type")

    @require_torch
    def test_batch_equivalence(self):
        """Test that batched processing produces reasonable results."""
        feat_extract = self.feature_extraction_class(sampling_rate=16000)
        
        # Create two different length inputs
        audio1 = torch.randn(8000)  # 0.5 seconds
        audio2 = torch.randn(12000)  # 0.75 seconds
        
        # Process individually
        features1 = feat_extract(audio1.unsqueeze(0), audio_lengths=torch.tensor([8000]), return_tensors="pt")
        features2 = feat_extract(audio2.unsqueeze(0), audio_lengths=torch.tensor([12000]), return_tensors="pt")
        
        # Process as batch
        max_length = 12000
        padded_audio1 = torch.cat([audio1, torch.zeros(max_length - 8000)])
        batch_audio = torch.stack([padded_audio1, audio2])
        batch_lengths = torch.tensor([8000, 12000])
        
        batch_features = feat_extract(batch_audio, audio_lengths=batch_lengths, return_tensors="pt")
        
        # Test that batch processing produces expected shapes and properties
        self.assertEqual(batch_features.input_features.shape[0], 2)  # batch size
        self.assertEqual(batch_features.input_features.shape[2], feat_extract.feature_size)  # feature dimension
        self.assertEqual(batch_features.attention_mask.shape[0], 2)  # batch size
        self.assertEqual(batch_features.input_lengths.shape[0], 2)  # batch size
        
        # Test that input lengths are reasonable
        self.assertGreater(batch_features.input_lengths[0].item(), 0)
        self.assertGreater(batch_features.input_lengths[1].item(), 0)
        self.assertLess(batch_features.input_lengths[0].item(), batch_features.input_lengths[1].item())
        
        # Test that attention masks have correct number of valid frames
        self.assertEqual(batch_features.attention_mask[0].sum().item(), batch_features.input_lengths[0].item())
        self.assertEqual(batch_features.attention_mask[1].sum().item(), batch_features.input_lengths[1].item())

    def test_feature_extractor_without_torch(self):
        """Test that appropriate error is raised when torch is not available."""
        import unittest.mock
        
        with unittest.mock.patch("transformers.models.fastconformer.feature_extraction_fastconformer.is_torch_available", lambda: False):
            with self.assertRaises(ImportError):
                feat_extract = FastConformerFeatureExtractor()
                audio = [[1, 2, 3, 4, 5]]
                feat_extract(audio)


if __name__ == "__main__":
    unittest.main() 