# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

import numpy as np

from transformers import AutoFeatureExtractor, Lfm2AudioFeatureExtractor, ParakeetFeatureExtractor
from transformers.testing_utils import require_librosa, require_torch, require_torch_gpu
from transformers.utils import is_torch_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin


if is_torch_available():
    import torch


@require_torch
@require_librosa
class Lfm2AudioFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = Lfm2AudioFeatureExtractor
    feat_extract_dict = {"feature_size": 8, "sampling_rate": 16000}

    def test_terminal_frame_and_normalization(self):
        audio = np.random.default_rng(0).standard_normal(12800).astype(np.float32)
        extractor = Lfm2AudioFeatureExtractor(feature_size=8)
        outputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
        parakeet = ParakeetFeatureExtractor(feature_size=8)(audio, sampling_rate=16000, return_tensors="pt")
        torch.testing.assert_close(outputs.input_features, parakeet.input_features, atol=0, rtol=0)
        self.assertEqual(outputs.attention_mask.sum().item(), 81)
        self.assertEqual(parakeet.attention_mask.sum().item(), 80)
        self.assertTrue((outputs.input_features[:, -1] == 0).all())

    def test_batch_masks(self):
        audio = [np.zeros(12800, dtype=np.float32), np.zeros(1600, dtype=np.float32)]
        outputs = Lfm2AudioFeatureExtractor()(audio, sampling_rate=16000, return_tensors="pt")
        self.assertEqual(outputs.attention_mask.sum(-1).tolist(), [81, 11])
        self.assertTrue((outputs.input_features[1, 11:] == 0).all())
        self.assertTrue(torch.isfinite(outputs.input_features).all())

    def test_lists_and_noncontiguous_arrays(self):
        extractor = Lfm2AudioFeatureExtractor()
        audio = np.linspace(-1, 1, 1600, dtype=np.float32)[::-1]
        expected = extractor(audio.copy(), sampling_rate=16000, return_tensors="pt")
        for waveform in (audio, audio.tolist(), [audio.tolist()], torch.from_numpy(audio.copy())):
            actual = extractor(waveform, sampling_rate=16000, return_tensors="pt")
            torch.testing.assert_close(actual.input_features, expected.input_features)

    def test_truncation(self):
        extractor = Lfm2AudioFeatureExtractor()
        audio = np.linspace(-1, 1, 3200, dtype=np.float32)
        expected = extractor(audio[:1600], sampling_rate=16000, return_tensors="pt")
        actual = extractor(audio, sampling_rate=16000, max_length=1600, truncation=True, return_tensors="pt")
        torch.testing.assert_close(actual.input_features, expected.input_features)
        torch.testing.assert_close(actual.attention_mask, expected.attention_mask)

    def test_invalid_inputs(self):
        extractor = Lfm2AudioFeatureExtractor()
        for audio in ([], np.zeros(100), [np.zeros((2, 1600))]):
            with self.assertRaises(ValueError):
                extractor(audio, sampling_rate=16000)
        with self.assertRaises(ValueError):
            extractor(np.zeros(1600), sampling_rate=8000)

    def test_save_and_load(self):
        extractor = Lfm2AudioFeatureExtractor(feature_size=8)
        with tempfile.TemporaryDirectory() as directory:
            extractor.save_pretrained(directory)
            reloaded = AutoFeatureExtractor.from_pretrained(directory)
        self.assertIsInstance(reloaded, Lfm2AudioFeatureExtractor)
        self.assertEqual(extractor.to_dict(), reloaded.to_dict())

    @require_torch_gpu
    def test_torch_integration_cuda(self):
        feature_extractor = Lfm2AudioFeatureExtractor(
            feature_size=8,
            sampling_rate=16_000,
            hop_length=160,
            n_fft=512,
            win_length=400,
        )
        inputs = feature_extractor(
            np.zeros(1600, dtype=np.float32),
            sampling_rate=16_000,
            return_tensors="pt",
            device="cuda",
        )

        self.assertEqual(inputs.input_features.device.type, "cuda")
        self.assertEqual(inputs.input_features.dtype, torch.float32)
        self.assertEqual(inputs.attention_mask.device.type, "cuda")

        window_pointer = feature_extractor.window.data_ptr()
        mel_filters_pointer = feature_extractor.mel_filters.data_ptr()
        repeated_inputs = feature_extractor(
            np.zeros(1600, dtype=np.float32),
            sampling_rate=16_000,
            return_tensors="pt",
            device="cuda",
        )

        self.assertEqual(feature_extractor.window.data_ptr(), window_pointer)
        self.assertEqual(feature_extractor.mel_filters.data_ptr(), mel_filters_pointer)
        torch.testing.assert_close(repeated_inputs.input_features, inputs.input_features, rtol=0, atol=0)

        audio_batch = [
            np.linspace(-1.0, 1.0, 1600, dtype=np.float32),
            np.linspace(-0.5, 0.5, 1200, dtype=np.float32),
        ]
        fast_inputs = feature_extractor(audio_batch, sampling_rate=16_000, return_tensors="pt", device="cuda")
        fallback_inputs = feature_extractor(
            audio_batch,
            sampling_rate=16_000,
            pad_to_multiple_of=1,
            return_tensors="pt",
            device="cuda",
        )
        torch.testing.assert_close(fast_inputs.input_features, fallback_inputs.input_features, rtol=0, atol=0)
        self.assertTrue(torch.equal(fast_inputs.attention_mask, fallback_inputs.attention_mask))
