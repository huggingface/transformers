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


import unittest

import numpy as np

from parameterized import parameterized
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import VideoMAEFeatureExtractor


class VideoMAEFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=10,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=18,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_feat_extract_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }


@require_torch
@require_vision
class VideoMAEFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = VideoMAEFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = VideoMAEFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))

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
        encoded_videos = feature_extractor(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size,
                self.feature_extract_tester.size,
            ),
        )

        # Test batched
        encoded_videos = feature_extractor(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size,
                self.feature_extract_tester.size,
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
        encoded_videos = feature_extractor(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size,
                self.feature_extract_tester.size,
            ),
        )

        # Test batched
        encoded_videos = feature_extractor(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size,
                self.feature_extract_tester.size,
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
        encoded_videos = feature_extractor(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size,
                self.feature_extract_tester.size,
            ),
        )

        # Test batched
        encoded_videos = feature_extractor(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_frames,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size,
                self.feature_extract_tester.size,
            ),
        )

    @parameterized.expand(
        [
            ("do_resize_True_do_center_crop_True_do_normalize_True", True, True, True),
            ("do_resize_True_do_center_crop_True_do_normalize_False", True, True, False),
            ("do_resize_True_do_center_crop_False_do_normalize_True", True, False, True),
            ("do_resize_True_do_center_crop_False_do_normalize_False", True, False, False),
            ("do_resize_False_do_center_crop_True_do_normalize_True", False, True, True),
            ("do_resize_False_do_center_crop_True_do_normalize_False", False, True, False),
            ("do_resize_False_do_center_crop_False_do_normalize_True", False, False, True),
            ("do_resize_False_do_center_crop_False_do_normalize_False", False, False, False),
        ]
    )
    def test_call_flags(self, _, do_resize, do_center_crop, do_normalize):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        feature_extractor.do_center_crop = do_center_crop
        feature_extractor.do_resize = do_resize
        feature_extractor.do_normalize = do_normalize
        # create random PIL images
        video_inputs = prepare_video_inputs(self.feature_extract_tester, equal_resolution=False)

        pixel_values = feature_extractor(video_inputs, return_tensors=None)["pixel_values"]
        self.assertEqual(len(pixel_values), self.feature_extract_tester.batch_size)

        num_channels = self.feature_extract_tester.num_channels
        size = self.feature_extract_tester.size
        num_frames = self.feature_extract_tester.num_frames
        crop_size = self.feature_extract_tester.size

        for video_input, video_output in zip(video_inputs, pixel_values):
            expected_shape = [(3, *video_input[0].size[::-1])]
            if do_resize:
                c, height, width = expected_shape[0]
                short, long = (width, height) if width <= height else (height, width)
                min_size = size
                if short == min_size:
                    resized_shape = (c, height, width)
                else:
                    short, long = min_size, int(long * min_size / short)
                    resized_shape = (c, long, short) if width <= height else (c, short, long)
                expected_shape = [resized_shape]
            if do_center_crop:
                expected_shape = [(num_channels, crop_size, crop_size)]
            expected_shapes = expected_shape * num_frames

            for idx, frame in enumerate(video_output):
                self.assertEqual(frame.shape, expected_shapes[idx])
                self.assertIsInstance(frame, np.ndarray)
