# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from transformers import ImageBindFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_torchaudio
from transformers.utils.import_utils import is_speech_available, is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


global_rng = random.Random()

if is_torch_available():
    import torch

if is_speech_available():
    import torchaudio


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


class ImageBindFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=16000,
        do_normalize=True,
        return_attention_mask=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
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


@require_torch
@require_torchaudio
class ImageBindFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = ImageBindFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = ImageBindFeatureExtractionTester(self)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_sequences_1 = feat_extract(speech_inputs[0], return_tensors="np").input_features
        encoded_sequences_2 = feat_extract(np_speech_inputs[0], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test 3-D numpy arrays are batched and chunked.
        speech_inputs = [[floats_list((1, x))[0]] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

    def _load_datasamples(self):
        from datasets import load_dataset

        ds = load_dataset("EduardoPacheco/imagebind-example-data", split="train")
        audios = [
            torchaudio.functional.resample(
                torch.from_numpy(audio["array"]),
                orig_freq=audio["sampling_rate"],
                new_freq=self.feat_extract_tester.sampling_rate,
            ).numpy()
            for audio in ds["audio"]
        ]

        return audios

    @require_torch
    def test_integration(self):
        # fmt: off
        expected_input = torch.tensor(
            [[-1.2776, -0.9167, -1.2776],
            [-1.2439, -0.8372, -0.8748],
            [-1.1235, -0.7492, -1.0867]]
        )
        # fmt: on

        input_speech = self._load_datasamples()
        feature_extractor = ImageBindFeatureExtractor()
        input_values = feature_extractor(input_speech, return_tensors="pt").input_features
        expected_shape = (
            len(input_speech),
            feature_extractor.num_chunks,
            1,
            feature_extractor.num_mel_bins,
            feature_extractor.max_length,
        )
        self.assertEqual(input_values.shape, expected_shape)
        self.assertTrue(torch.allclose(input_values[:, :, 0, 0, 0], expected_input, atol=1e-4))

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertDictEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)


# exact same tests than before, except that we simulate that torchaudio is not available
@require_torch
@unittest.mock.patch(
    "transformers.models.imagebind.feature_extraction_imagebind.is_speech_available",
    lambda: False,
)
class ImageBindFeatureExtractionWithoutTorchaudioTest(ImageBindFeatureExtractionTest):
    def test_using_audio_utils(self):
        # Tests that it uses audio_utils instead of torchaudio
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())

        self.assertTrue(hasattr(feat_extract, "window"))
        self.assertTrue(hasattr(feat_extract, "mel_filters"))

        from transformers.models.imagebind.feature_extraction_imagebind import (
            is_speech_available,
        )

        self.assertFalse(is_speech_available())
