# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

from transformers import TransformerTransducerFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_torchaudio
from transformers.utils import is_torchaudio_available


if is_torchaudio_available():
    import torch
    from torchaudio.transforms import MelSpectrogram

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


sys.path.append(str(Path(__file__).parent.parent / "utils"))

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


class TransformerTransducerFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        n_fft: int = 512,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 128,
        stack: int = 4,
        stride: int = 3,
        power: int = 2.0,
        center: bool = True,
        mel_scale: str = "slaney",
        filter_norm: Optional[str] = None,
        min_frequency: float = 0.0,
        max_frequency: Optional[float] = None,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,  # pad inputs to max length with silence token (zero) and no attention mask
    ):
        self.parent = parent
        self.n_fft = n_fft
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.mel_scale = mel_scale
        self.center = center
        self.power = power
        self.filter_norm = filter_norm
        self.max_frequency = max_frequency if max_frequency else float(sampling_rate // 2)
        self.min_frequency = min_frequency

        self.stack = stack
        self.stride = stride

        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask

    def prepare_feat_extract_dict(self):
        return {
            "n_fft": self.n_fft,
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "hop_length": self.hop_length,
            "stack": self.stack,
            "stride": self.stride,
            "power": self.power,
            "center": self.center,
            "mel_scale": self.mel_scale,
            "filter_norm": self.filter_norm,
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "return_attention_mask": self.return_attention_mask,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                _flatten(floats_list((x, self.feature_size)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]

        return speech_inputs


class TransformerTransducerFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = TransformerTransducerFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = TransformerTransducerFeatureExtractionTester(self)
        self.feat_extract_dict = self.feat_extract_tester.prepare_feat_extract_dict()  # for debugging
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.feat_extract_tester.sampling_rate,
            n_fft=self.feat_extract_tester.n_fft,
            hop_length=self.feat_extract_tester.hop_length,
            f_min=self.feat_extract_tester.min_frequency,
            f_max=self.feat_extract_tester.max_frequency,
            power=self.feat_extract_tester.power,
            center=self.feat_extract_tester.center,
            mel_scale=self.feat_extract_tester.mel_scale,
            norm=self.feat_extract_tester.filter_norm,
        )

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        sampling_rate = self.feat_extract_dict["sampling_rate"]
        audio_duration = 10  # unit is second

        audio_length_generator = range(sampling_rate, (sampling_rate * audio_duration), sampling_rate)
        raw_audios = [floats_list((1, x))[0] for x in audio_length_generator]
        numpy_raw_audios = [np.asarray(audio) for audio in raw_audios]

        extractor_outputs = feature_extractor(numpy_raw_audios, return_tensors="np")
        input_features = extractor_outputs.input_features

        self.assertTrue(input_features.ndim == 3)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(raw_audios[0], return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(numpy_raw_audios[0], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        encoded_sequences_1 = feature_extractor(raw_audios, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(numpy_raw_audios, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test truncation required ???

    @require_torch
    @require_torchaudio
    def test_mel_compressor(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        sampling_rate = self.feat_extract_dict["sampling_rate"]
        audio_duration = 10  # unit is second

        audio_length_generator = range(sampling_rate, (sampling_rate * audio_duration), sampling_rate)
        raw_audios = [floats_list((1, x))[0] for x in audio_length_generator]
        raw_audios = [np.asarray(audio) for audio in raw_audios]

        torch_raw_audios = [torch.tensor(audio, dtype=torch.float32) for audio in raw_audios]
        torch_mel_spectrogram = [self.mel_spectrogram(audio) for audio in torch_raw_audios]
        numpy_mel_spectrogram = [feature_extractor.get_mel_spectrogram(audio) for audio in raw_audios]

        audio_zip = zip(torch_mel_spectrogram, numpy_mel_spectrogram)
        for torch_audio, numpy_audio in audio_zip:
            torch_result = feature_extractor.mel_compressor(torch_audio.numpy())
            numpy_result = feature_extractor.mel_compressor(numpy_audio)

            self.assertTrue(np.allclose(torch_result, numpy_result, rtol=1, atol=1))

    @require_torch
    @require_torchaudio
    def test_mel_spectrogram(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        sampling_rate = self.feat_extract_dict["sampling_rate"]
        audio_duration = 10  # unit is second

        audio_length_generator = range(sampling_rate, (sampling_rate * audio_duration), sampling_rate)
        raw_audios = [floats_list((1, x))[0] for x in audio_length_generator]
        raw_audios = [np.asarray(audio) for audio in raw_audios]

        torch_raw_audios = [torch.tensor(audio, dtype=torch.float32) for audio in raw_audios]
        torch_mel_spectrogram = [self.mel_spectrogram(audio) for audio in torch_raw_audios]
        numpy_mel_spectrogram = [feature_extractor.get_mel_spectrogram(audio) for audio in raw_audios]

        audio_zip = zip(torch_mel_spectrogram, numpy_mel_spectrogram)
        for torch_audio, numpy_audio in audio_zip:
            self.assertTrue(np.allclose(torch_audio.numpy(), numpy_audio, rtol=1, atol=1))  # folerence must be check,
            # for check
            # [[(x_, y_) for x_, y_ in zip(x, y)] for x, y in zip(torch_audio.numpy(), numpy_audio)]

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()

        mel_1 = dict_first.pop("mel_filter")
        mel_2 = dict_second.pop("mel_filter")
        self.assertTrue(np.allclose(mel_1, mel_2))

        window_1 = dict_first.pop("window_fn")
        window_2 = dict_second.pop("window_fn")
        self.assertTrue(np.allclose(window_1, window_2))

        self.assertEqual(dict_first, dict_second)

    @require_torch
    def test_double_precision_pad(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_features.dtype == np.float32)

            pt_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = dict_first.pop("mel_filter")
        mel_2 = dict_second.pop("mel_filter")
        self.assertTrue(np.allclose(mel_1, mel_2))

        window_1 = dict_first.pop("window_fn")
        window_2 = dict_second.pop("window_fn")
        self.assertTrue(np.allclose(window_1, window_2))

        self.assertEqual(dict_first, dict_second)
