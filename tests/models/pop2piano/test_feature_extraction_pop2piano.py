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
import os
import random
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from transformers import is_speech_available
from transformers.testing_utils import (check_json_file_has_correct_format, require_torch,
                                        require_essentia, require_librosa, require_scipy,
                                        require_pretty_midi, require_soundfile)
from transformers.utils.import_utils import (is_torch_available, is_essentia_available,
                                             is_scipy_available, is_librosa_available,
                                             is_soundfile_availble, )

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin

requirements = is_speech_available() and is_torch_available() and is_essentia_available() and is_scipy_available() and \
        is_librosa_available() and is_soundfile_availble()

if requirements:
    from transformers import Pop2PianoFeatureExtractor
if is_torch_available():
    import torch

@require_torch
@require_essentia
@require_librosa
@require_soundfile
@require_scipy
@require_pretty_midi
class Pop2PianoFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        n_bars=2,
        sample_rate=22050,
        use_mel=True,
        padding_value=0,
        vocab_size_special=4,
        vocab_size_note=128,
        vocab_size_velocity=2,
        vocab_size_time=100,
    ):
        self.parent = parent
        self.n_bars = n_bars
        self.sample_rate = sample_rate
        self.use_mel = use_mel
        self.padding_value = padding_value
        self.vocab_size_special = vocab_size_special
        self.vocab_size_note = vocab_size_note
        self.vocab_size_velocity = vocab_size_velocity
        self.vocab_size_time = vocab_size_time

    def prepare_feat_extract_dict(self):
        return {
            "n_bars": self.n_bars,
            "sample_rate": self.sample_rate,
            "use_mel": self.use_mel,
            "padding_value": self.padding_value,
            "vocab_size_special": self.vocab_size_special,
            "vocab_size_note": self.vocab_size_note,
            "vocab_size_velocity": self.vocab_size_velocity,
            "vocab_size_time":self.vocab_size_time,
        }

@require_torch
@require_essentia
@require_librosa
@require_soundfile
@require_scipy
@require_pretty_midi
class Pop2PianoFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Pop2PianoFeatureExtractor if requirements else None

    def setUp(self):
        self.feat_extract_tester = Pop2PianoFeatureExtractionTester(self)

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = feat_extract_first.use_mel
        mel_2 = feat_extract_second.use_mel
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
        mel_1 = feat_extract_first.use_mel
        mel_2 = feat_extract_second.use_mel
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input = np.zeros([1000000, ])

        input_features = feature_extractor(speech_input, audio_sr=16_000, return_tensors="np")
        self.assertTrue(input_features.input_features.ndim == 2)
        self.assertTrue(input_features.beatsteps.ndim == 1)
        self.assertTrue(input_features.ext_beatstep.ndim == 1)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples], [x["sampling_rate"] for x in speech_samples]

    def test_integration(self):
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [-4.5434e-05, -1.8900e-04, -2.2150e-04, -2.1844e-04, -2.7647e-04,
             -2.1334e-04, -1.5305e-04, -2.6124e-04, -2.6863e-04, -1.5969e-04,
             -1.6224e-04, -1.2900e-04, -9.9139e-06, 1.5336e-05, 4.7507e-05,
             9.3454e-05, -2.3652e-05, -1.2942e-04, -1.0804e-04, -1.4267e-04,
             -1.5102e-04, -6.7488e-05, -9.6527e-05, -9.6909e-05, 8.0032e-05,
             8.1948e-05, -7.3148e-05, 3.4405e-05, 1.5065e-04, -1.0989e-04]
        )

        input_speech, sampling_rate = self._load_datasamples(1)
        feaure_extractor = Pop2PianoFeatureExtractor.from_pretrained("susnato/pop2piano_dev")
        input_features = feaure_extractor(input_speech, audio_sr=sampling_rate[0], return_tensors="pt").input_features
        self.assertTrue(torch.allclose(input_features[0, :30], EXPECTED_INPUT_FEATURES, atol=1e-4))

    @unittest.skip("Pop2PianoFeatureExtractor does not return attention_mask")
    def test_attention_mask(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not return attention_mask")
    def test_attention_mask_with_truncation(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor only takes one raw_audio at a time")
    def test_batch_feature_pt(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor only takes one raw_audio at a time")
    def test_batch_feature_tf(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor only takes one raw_audio at a time")
    def test_batch_feature(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not supports padding")
    def test_padding_accepts_tensors_pt(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not supports padding")
    def test_padding_accepts_tensors_tf(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not supports truncation")
    def test_truncation_from_array(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not supports truncation")
    def test_truncation_from_list(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not supports padding")
    def test_padding_from_list(self):
        pass

    @unittest.skip("Pop2PianoFeatureExtractor does not supports padding")
    def test_padding_from_array(self):
        pass