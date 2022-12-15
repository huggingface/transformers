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
""" Testing suite for the TVLT feature extractor. """

import random
import unittest

import numpy as np

from test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_video_inputs
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import TvltAudioFeatureExtractor, TvltPixelFeatureExtractor

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


class TvltPixelFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        pixel_size=18,
        num_frames=8,
        num_channels=3,
        do_resize=True,
        do_center_crop=True,
        do_normalize=True,
        min_resolution=32,
        max_resolution=384,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        crop_size=None,
    ):
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}

        self.parent = parent
        self.batch_size = batch_size
        self.do_resize = do_resize
        self.pixel_size = pixel_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.image_mean = image_mean
        self.image_std = image_std
        self.crop_size = crop_size

    def prepare_feat_extract_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "do_center_crop": self.do_center_crop,
            "pixel_size": self.pixel_size,
        }


@require_torch
@require_vision
class TvltPixelFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = TvltPixelFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = TvltPixelFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "do_center_crop"))
        self.assertTrue(hasattr(feature_extractor, "pixel_size"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL videos
        video_inputs = prepare_video_inputs(self.feature_extract_tester, equal_resolution=False)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], Image.Image)

        # Test not batched input
        encoded_pixels = feature_extractor(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_pixels.shape,
            (
                1,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size["height"],
                self.feature_extract_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_pixels = feature_extractor(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_pixels.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size["height"],
                self.feature_extract_tester.crop_size["width"],
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        video_inputs = prepare_video_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)

        # Test not batched input
        encoded_pixels = feature_extractor(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_pixels.shape,
            (
                1,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size["height"],
                self.feature_extract_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_pixels = feature_extractor(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_pixels.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size["height"],
                self.feature_extract_tester.crop_size["width"],
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        video_inputs = prepare_video_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], torch.Tensor)

        # Test not batched input
        encoded_pixels = feature_extractor(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_pixels.shape,
            (
                1,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size["height"],
                self.feature_extract_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_pixels = feature_extractor(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_pixels.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size["height"],
                self.feature_extract_tester.crop_size["width"],
            ),
        )


class TvltAudioFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=128,
        max_seq_length=2048,
        audio_size=1024,
        num_channels=1,
        feature_size=128,
        sampling_rate=44100,
        hop_length=512,
        chunk_length=30,
    ):
        self.parent = parent
        self.batch_size = batch_size

        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.audio_size = audio_size
        self.num_channels = num_channels
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        return {
            "audio_size": self.audio_size,
        }


@require_torch
@require_vision
class TvltAudioFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = TvltAudioFeatureExtractor

    def setUp(self):
        self.feature_extract_tester = TvltAudioFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "audio_size"))

    def test_batch_feature(self):
        pass

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 20000)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_audios = feature_extractor(np_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

        # Test batched
        encoded_audios = feature_extractor(np_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(8000, 14000, 20000)]
        torch_speech_inputs = [torch.tensor(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_audios = feature_extractor(torch_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

        # Test batched
        encoded_audios = feature_extractor(torch_speech_inputs[0], return_tensors="pt").audio_values

        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.audio_size)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)