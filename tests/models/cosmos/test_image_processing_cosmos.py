# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import CosmosVideoProcessor


class CosmosVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=9,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        do_pad=True,
    ):
        super().__init__()
        size = size if size is not None else {"height": 64, "width": 100}
        crop_size = crop_size if crop_size is not None else {"height": 64, "width": 100}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean else [1.0, 1.0, 1.0]
        self.image_std = image_std if image_std else [1.0, 1.0, 1.0]
        self.do_convert_rgb = do_convert_rgb
        self.do_pad = do_pad

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_pad": self.do_pad,
        }

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        video = prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            torchify=torchify,
            numpify=numpify,
        )
        return video


@require_torch
@require_vision
class CosmosVideoProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = CosmosVideoProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = CosmosVideoProcessingTester(self)

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_center_crop"))
            self.assertTrue(hasattr(image_processing, "center_crop"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "do_pad"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 64, "width": 100})
            self.assertEqual(image_processor.crop_size, {"height": 64, "width": 100})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            video_inputs = self.image_processor_tester.prepare_video_inputs()
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_images = image_processing(video_inputs[0], return_tensors="pt").pixel_values_videos
            expected_output_image_shape = (1, 33, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(video_inputs, return_tensors="pt").pixel_values_videos
            expected_output_image_shape = (7, 33, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, numpify=True)
            for video in video_inputs:
                self.assertIsInstance(video[0], np.ndarray)

            # Test not batched input
            encoded_images = image_processing(video_inputs[0], return_tensors="pt").pixel_values_videos
            expected_output_image_shape = (1, 33, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(video_inputs, return_tensors="pt").pixel_values_videos
            expected_output_image_shape = (7, 33, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, torchify=True)
            for video in video_inputs:
                self.assertIsInstance(video[0], torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(video_inputs[0], return_tensors="pt").pixel_values_videos
            expected_output_image_shape = (1, 33, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(video_inputs, return_tensors="pt").pixel_values_videos
            expected_output_image_shape = (7, 33, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_no_pad(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, torchify=True)
            for video in video_inputs:
                self.assertIsInstance(video[0], torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(video_inputs[0], do_pad=False, return_tensors="np").pixel_values_videos
            expected_output_image_shape = (1, 9, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(video_inputs, do_pad=False, return_tensors="np").pixel_values_videos
            expected_output_image_shape = (7, 9, 3, 64, 100)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    @unittest.skip(reason="CosmosVideoProcessor doesn't support 4 channel videos")
    def test_call_numpy_4_channels(self):
        pass
