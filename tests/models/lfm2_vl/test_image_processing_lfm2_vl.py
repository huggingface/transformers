# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image


if is_torch_available():
    import torch

    if is_torchvision_available():
        from transformers import Lfm2VlImageProcessorFast
        from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import find_closest_aspect_ratio


class Lfm2VlImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_images=1,
        min_resolution=256,
        max_resolution=1024,
        downsample_factor=2,
        do_image_splitting=False,
        min_tiles=2,
        max_tiles=10,
        use_thumbnail=True,
        min_image_tokens=64,
        max_image_tokens=256,
        encoder_patch_size=16,
        tile_size=512,
        max_pixels_tolerance=2.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_images = num_images
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

        self.downsample_factor = downsample_factor
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance

    def prepare_image_processor_dict(self):
        return {
            "downsample_factor": self.downsample_factor,
            "do_image_splitting": self.do_image_splitting,
            "min_tiles": self.min_tiles,
            "max_tiles": self.max_tiles,
            "use_thumbnail": self.use_thumbnail,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "encoder_patch_size": self.encoder_patch_size,
            "tile_size": self.tile_size,
            "max_pixels_tolerance": self.max_pixels_tolerance,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]


@require_torch
@require_vision
class Lfm2VlImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    test_slow_image_processor = False
    fast_image_processing_class = Lfm2VlImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Lfm2VlImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "downsample_factor"))
            self.assertTrue(hasattr(image_processing, "min_tiles"))
            self.assertTrue(hasattr(image_processing, "max_tiles"))
            self.assertTrue(hasattr(image_processing, "use_thumbnail"))
            self.assertTrue(hasattr(image_processing, "min_image_tokens"))
            self.assertTrue(hasattr(image_processing, "max_image_tokens"))
            self.assertTrue(hasattr(image_processing, "encoder_patch_size"))
            self.assertTrue(hasattr(image_processing, "tile_size"))
            self.assertTrue(hasattr(image_processing, "max_pixels_tolerance"))

    @require_vision
    def test_smart_resize(self):
        # verify that smart resize output dims are divisible by encoder_patch_size * downsample_factor
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        width, height = image_processing.smart_resize(
            height=500,
            width=300,
            downsample_factor=image_processing.downsample_factor,
            min_image_tokens=image_processing.min_image_tokens,
            max_image_tokens=image_processing.max_image_tokens,
            encoder_patch_size=image_processing.encoder_patch_size,
        )
        mod = image_processing.encoder_patch_size * image_processing.downsample_factor
        self.assertEqual(width % mod, 0)
        self.assertEqual(height % mod, 0)

    @require_vision
    def test_get_grid_layout(self):
        # splitting a 512×512 image into tiles of size processor.image_processor.tile_size
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        rows, cols, _, _, num_patches = image_processing._get_grid_layout(
            height=1024,
            width=1024,
            min_tiles=image_processing.min_tiles,
            max_tiles=image_processing.max_tiles,
            tile_size=image_processing.tile_size,
        )
        self.assertEqual(num_patches, 4)
        self.assertEqual(num_patches, rows * cols)

        rows, cols, _, _, num_patches = image_processing._get_grid_layout(
            height=1024,
            width=1024,
            min_tiles=8,
            max_tiles=8,
            tile_size=image_processing.tile_size,
        )
        self.assertEqual(num_patches, 8)
        self.assertEqual(num_patches, rows * cols)

    def test_find_closest_aspect_ratio(self):
        # should pick (1,1) over (2,1) for a square image
        result = find_closest_aspect_ratio(1.0, [(1, 1), (2, 1)], width=100, height=100, image_size=100)
        self.assertEqual(result, (1, 1))

        result = find_closest_aspect_ratio(0.5, [(1, 1), (1, 2)], width=100, height=200, image_size=200)
        self.assertEqual(result, (1, 2))

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for sample_images in image_inputs:
            for image in sample_images:
                self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_call_numpy_4_channels(self):
        # Lfm2Vl always processes images as RGB, so it always returns images with 3 channels
        # Initialize image_processing
        image_processor_dict = self.image_processor_dict
        image_processing = self.fast_image_processing_class(**image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        for sample_images in image_inputs:
            for image in sample_images:
                self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for images in image_inputs:
            for image in images:
                self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for images in image_inputs:
            for image in images:
                self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )
