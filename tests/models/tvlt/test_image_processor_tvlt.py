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
""" Testing suite for the TVLT image processor. """

import json
import os
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingSavingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import TvltImageProcessor


def prepare_video(feature_extract_tester, width=10, height=10, numpify=False, torchify=False):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for i in range(feature_extract_tester.num_frames):
        video.append(np.random.randint(255, size=(feature_extract_tester.num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]

    if torchify:
        video = [torch.from_numpy(frame) for frame in video]

    return video


def prepare_video_inputs(feature_extract_tester, equal_resolution=False, numpify=False, torchify=False):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.
    One can specify whether the videos are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    video_inputs = []
    for i in range(feature_extract_tester.batch_size):
        if equal_resolution:
            width = height = feature_extract_tester.max_resolution
        else:
            width, height = np.random.choice(
                np.arange(feature_extract_tester.min_resolution, feature_extract_tester.max_resolution), 2
            )
            video = prepare_video(
                feature_extract_tester=feature_extract_tester,
                width=width,
                height=height,
                numpify=numpify,
                torchify=torchify,
            )
        video_inputs.append(video)

    return video_inputs


class TvltImageProcessorTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=4,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        crop_size=None,
    ):
        size = size if size is not None else {"shortest_edge": 18}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}

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
        self.crop_size = crop_size

    def prepare_feat_extract_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "crop_size": self.crop_size,
        }


@require_torch
@require_vision
class TvltImageProcessorTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = TvltImageProcessor if is_vision_available() else None

    def setUp(self):
        self.feature_extraction_tester = TvltImageProcessorTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extraction_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "do_center_crop"))
        self.assertTrue(hasattr(feature_extractor, "size"))

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        delattr(feat_extract_first, "random_generator")

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname, random_generator=None)
            delattr(feat_extract_second, "random_generator")

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        delattr(feat_extract_first, "random_generator")

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)
            delattr(feat_extract_second, "random_generator")

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_to_json_string(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        delattr(feat_extract, "random_generator")
        obj = json.loads(feat_extract.to_json_string())
        for key, value in self.feat_extract_dict.items():
            self.assertEqual(obj[key], value)

    def test_call_pil(self):
        # Initialize feature_extraction
        feature_extraction = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL videos
        video_inputs = prepare_video_inputs(self.feature_extraction_tester, equal_resolution=False)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], Image.Image)

        # Test not batched input
        encoded_videos = feature_extraction(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.feature_extraction_tester.num_frames,
                self.feature_extraction_tester.num_channels,
                self.feature_extraction_tester.crop_size["height"],
                self.feature_extraction_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_videos = feature_extraction(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.feature_extraction_tester.batch_size,
                self.feature_extraction_tester.num_frames,
                self.feature_extraction_tester.num_channels,
                self.feature_extraction_tester.crop_size["height"],
                self.feature_extraction_tester.crop_size["width"],
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extraction
        feature_extraction = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        video_inputs = prepare_video_inputs(self.feature_extraction_tester, equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)

        # Test not batched input
        encoded_videos = feature_extraction(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.feature_extraction_tester.num_frames,
                self.feature_extraction_tester.num_channels,
                self.feature_extraction_tester.crop_size["height"],
                self.feature_extraction_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_videos = feature_extraction(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.feature_extraction_tester.batch_size,
                self.feature_extraction_tester.num_frames,
                self.feature_extraction_tester.num_channels,
                self.feature_extraction_tester.crop_size["height"],
                self.feature_extraction_tester.crop_size["width"],
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extraction
        feature_extraction = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        video_inputs = prepare_video_inputs(self.feature_extraction_tester, equal_resolution=False, torchify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], torch.Tensor)

        # Test not batched input
        encoded_videos = feature_extraction(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.feature_extraction_tester.num_frames,
                self.feature_extraction_tester.num_channels,
                self.feature_extraction_tester.crop_size["height"],
                self.feature_extraction_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_videos = feature_extraction(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.feature_extraction_tester.batch_size,
                self.feature_extraction_tester.num_frames,
                self.feature_extraction_tester.num_channels,
                self.feature_extraction_tester.crop_size["height"],
                self.feature_extraction_tester.crop_size["width"],
            ),
        )
