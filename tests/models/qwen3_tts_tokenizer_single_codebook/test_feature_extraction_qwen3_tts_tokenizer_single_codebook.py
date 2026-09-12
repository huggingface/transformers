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

from transformers import Qwen3TTSTokenizerSingleCodebookFeatureExtractor, WhisperFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch


@require_torch
class Qwen3TTSTokenizerSingleCodebookFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=128,
        sampling_rate=16000,
        audio_vq_ds_rate=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.audio_vq_ds_rate = audio_vq_ds_rate

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "audio_vq_ds_rate": self.audio_vq_ds_rate,
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
class Qwen3TTSTokenizerSingleCodebookFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Qwen3TTSTokenizerSingleCodebookFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Qwen3TTSTokenizerSingleCodebookFeatureExtractionTester(self)

    def _audio_inputs(self, lengths=(800, 1000, 1200)):
        return [floats_list((1, length))[0] for length in lengths]

    def test_call(self):
        TOL = 1e-6
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = self.feat_extract_tester.sampling_rate
        audio_inputs = self._audio_inputs()
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]
        torch_audio_inputs = [torch.tensor(audio_input) for audio_input in audio_inputs]

        # single input: list, numpy and torch agree
        encoded_1 = feat_extract(audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np")
        encoded_2 = feat_extract(np_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np")
        encoded_3 = feat_extract(torch_audio_inputs[0], sampling_rate=sampling_rate, return_tensors="np")
        for key in ("input_features", "ref_mels"):
            self.assertTrue(np.allclose(encoded_1[key], encoded_2[key], atol=TOL))
            self.assertTrue(np.allclose(encoded_1[key], encoded_3[key], atol=TOL))

        # batched: numpy and torch agree, and every item equals its single-input features
        batched_np = feat_extract(np_audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="np")
        batched_pt = feat_extract(torch_audio_inputs, sampling_rate=sampling_rate, padding=True, return_tensors="np")
        for key in ("input_features", "ref_mels"):
            self.assertTrue(np.allclose(batched_np[key], batched_pt[key], atol=TOL))
        for index, audio_input in enumerate(np_audio_inputs):
            single = feat_extract(audio_input, sampling_rate=sampling_rate, return_tensors="np")
            num_frames = single["input_features"].shape[-1]
            self.assertTrue(
                np.allclose(batched_np["input_features"][index, :, :num_frames], single["input_features"][0], atol=TOL)
            )
            self.assertEqual(int(batched_np["input_features_mask"][index].sum()), num_frames)
            num_ref_frames = single["ref_mels"].shape[1]
            self.assertTrue(
                np.allclose(batched_np["ref_mels"][index, :num_ref_frames], single["ref_mels"][0], atol=TOL)
            )

    def test_output_shapes(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio_inputs = [np.asarray(audio) for audio in self._audio_inputs((321, 640))]
        outputs = feat_extract(audio_inputs, sampling_rate=self.feat_extract_tester.sampling_rate, return_tensors="pt")

        self.assertEqual(set(outputs.keys()), {"input_features", "input_features_mask", "ref_mels"})
        # waveforms are padded to a multiple of `hop_length * 2 * audio_vq_ds_rate` = 640 samples = 4 frames
        self.assertEqual(outputs["input_features"].shape, (2, feat_extract.feature_size, 4))
        self.assertEqual(outputs["input_features_mask"].sum(dim=-1).tolist(), [4, 4])
        # the reference mel covers the unpadded waveform only
        self.assertEqual(outputs["ref_mels"].shape, (2, 4, feat_extract.ref_num_mel_bins))
        self.assertEqual(outputs["ref_mels"].dtype, torch.float32)

    def test_encoder_features_match_whisper(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        whisper_feat_extract = WhisperFeatureExtractor(
            feature_size=feat_extract.feature_size, sampling_rate=feat_extract.sampling_rate
        )
        audio = np.asarray(self._audio_inputs((1280,))[0])
        sampling_rate = self.feat_extract_tester.sampling_rate

        input_features = feat_extract(audio, sampling_rate=sampling_rate, return_tensors="np")["input_features"][0]
        whisper_features = whisper_feat_extract(
            audio, sampling_rate=sampling_rate, padding="longest", return_tensors="np"
        )
        self.assertTrue(np.allclose(input_features, whisper_features["input_features"][0], atol=1e-5))

    def test_ref_mels_are_peak_normalized(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = np.asarray(self._audio_inputs((1280,))[0])
        sampling_rate = self.feat_extract_tester.sampling_rate

        quiet = feat_extract(audio, sampling_rate=sampling_rate, return_tensors="np")
        loud = feat_extract(4 * audio, sampling_rate=sampling_rate, return_tensors="np")
        # the reference mel is computed on the peak-normalised waveform, the encoder features are not
        self.assertTrue(np.allclose(quiet["ref_mels"], loud["ref_mels"], atol=1e-5))
        self.assertFalse(np.allclose(quiet["input_features"], loud["input_features"], atol=1e-3))

    def test_padding_and_truncation(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        sampling_rate = self.feat_extract_tester.sampling_rate
        audio_inputs = [np.asarray(audio) for audio in self._audio_inputs((800, 1000, 1200))]

        padded = feat_extract(audio_inputs, sampling_rate=sampling_rate, padding="max_length", max_length=1920)
        self.assertEqual(padded["input_features"].shape[-1], 1920 // feat_extract.hop_length)
        self.assertEqual(padded["ref_mels"].shape[1], 1920 // feat_extract.ref_hop_length)
        self.assertEqual(padded["input_features_mask"].sum(-1).tolist(), [8, 8, 8])

        truncated = feat_extract(
            audio_inputs, sampling_rate=sampling_rate, max_length=640, truncation=True, return_tensors="np"
        )
        self.assertEqual(truncated["input_features"].shape, (3, feat_extract.feature_size, 4))
        self.assertEqual(truncated["ref_mels"].shape, (3, 4, feat_extract.ref_num_mel_bins))

        # unpadded ragged batches cannot be turned into an array
        with self.assertRaises(ValueError):
            feat_extract(audio_inputs, sampling_rate=sampling_rate, padding=False, return_tensors="np")

    def test_return_attention_mask(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio_inputs = [np.asarray(audio) for audio in self._audio_inputs((800, 1200))]
        outputs = feat_extract(
            audio_inputs, sampling_rate=self.feat_extract_tester.sampling_rate, return_attention_mask=False
        )
        self.assertNotIn("input_features_mask", outputs)

    def test_sampling_rate_validation(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = np.asarray(self._audio_inputs((800,))[0])
        with self.assertRaises(ValueError):
            feat_extract(audio, sampling_rate=24000)

    def test_double_precision_pad(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_audio_inputs = np.random.rand(100, 128).astype(np.float64)
        py_audio_inputs = np_audio_inputs.tolist()

        for inputs in [py_audio_inputs, np_audio_inputs]:
            np_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)
