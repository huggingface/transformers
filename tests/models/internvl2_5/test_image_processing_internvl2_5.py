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

from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers.models.internvl2_5.image_processing_internvl2_5 import dynamic_preprocess, find_closest_aspect_ratio
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import InternVL2_5ImageProcessor


class InternVL2_5ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        dynamic_size=None,
        do_normalize=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        min_tiles=1,
        max_tiles=12,
        use_thumbnail=True,
    ):
        size = size if size is not None else {"height": 20, "width": 20}
        dynamic_size = dynamic_size if dynamic_size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.do_resize = do_resize
        self.size = size
        self.dynamic_size = dynamic_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "dynamic_size": self.dynamic_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class InternVL2_5ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = InternVL2_5ImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = InternVL2_5ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 20, "width": 20})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"shortest_edge": 42})

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)

        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (10, 3, 20, 20)  # 9 patches + 1 thumbnail
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7 * 10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)

        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7 * 10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7 * 10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    @unittest.skip(reason="InternVL2_5ImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_input(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

        # Test batched as a list of images
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7 * 10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched as a nested list of images, where each sublist is one batch
        image_inputs_nested = [image_inputs[:3], image_inputs[3:]]
        encoded_images_nested = image_processing(image_inputs_nested, return_tensors="pt").pixel_values
        expected_output_image_shape = (7 * 10, 3, 20, 20)
        self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)

        # Image processor should return same pixel values, independently of input format
        self.assertTrue((encoded_images_nested == encoded_images).all())

    def test_dynamic_tiling(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)

        # Create a test image with known dimensions
        test_image = Image.new("RGB", (100, 50))  # 2:1 aspect ratio

        tiles = dynamic_preprocess(
            test_image,
            self.image_processor_tester.min_tiles,
            self.image_processor_tester.max_tiles,
            self.image_processor_tester.dynamic_size["height"],
            self.image_processor_tester.use_thumbnail,
        )

        # Test tiling with default settings
        encoded_images = image_processing(test_image, return_tensors="pt").pixel_values

        self.assertEqual(len(tiles), encoded_images.shape[0])
        for tile in tiles:
            self.assertEqual(tile.size, encoded_images.shape[-2:])

    def test_find_closest_aspect_ratio(self):
        min_num = self.image_processor_tester.min_tiles
        max_num = self.image_processor_tester.max_tiles
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )

        # Test with various aspect ratios
        test_cases = [
            (1.0, 100, 100, 20),  # Square
            (2.0, 200, 100, 20),  # 2:1 landscape
            (0.5, 100, 200, 20),  # 1:2 portrait
        ]

        for aspect_ratio, width, height, image_size in test_cases:
            result = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertTrue(result[0] * result[1] <= max_num)
            self.assertTrue(result[0] * result[1] >= min_num)

    def test_thumbnail_generation(self):
        # Test with thumbnail enabled
        image_processing_with_thumbnail = self.image_processing_class(**self.image_processor_dict, use_thumbnail=True)

        # Create test image that will generate multiple tiles
        test_image = Image.new("RGB", (200, 100))  # 2:1 aspect ratio

        tiles_with_thumb = image_processing_with_thumbnail(test_image, return_tensors="pt").pixel_values

        # Test with thumbnail disabled
        image_processing_without_thumbnail = self.image_processing_class(
            **self.image_processor_dict, use_thumbnail=False
        )
        tiles_no_thumb = image_processing_without_thumbnail(test_image, return_tensors="pt").pixel_values
        self.assertEqual(tiles_with_thumb.shape[0], tiles_no_thumb.shape[0] + 1)

    def test_tile_count_limits(self):
        # Test with custom min/max tile settings
        image_processing = self.image_processing_class(**self.image_processor_dict, min_tiles=2, max_tiles=4)

        # Test images with different dimensions
        test_cases = [
            Image.new("RGB", (100, 100)),  # Square
            Image.new("RGB", (200, 100)),  # Landscape
            Image.new("RGB", (100, 200)),  # Portrait
        ]

        for test_image in test_cases:
            tiles = image_processing(test_image).pixel_values
            # Number of tiles should be within limits (accounting for thumbnail)
            self.assertGreaterEqual(len(tiles), image_processing.min_tiles)
            self.assertLessEqual(len(tiles), image_processing.max_tiles + 1)  # +1 for possible thumbnail
