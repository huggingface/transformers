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

from ...test_image_processing_common import ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image

    from transformers import MllamaImageProcessor


if is_torch_available():
    import torch


class MllamaImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        num_images=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
        do_pad=True,
        max_image_tiles=4,
    ):
        size = size if size is not None else {"height": 224, "width": 224}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.max_image_tiles = max_image_tiles
        self.image_size = image_size
        self.num_images = num_images
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_convert_rgb = do_convert_rgb
        self.do_pad = do_pad

    def prepare_image_processor_dict(self):
        return {
            "do_convert_rgb": self.do_convert_rgb,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
            "max_image_tiles": self.max_image_tiles,
        }

    def prepare_image_inputs(
        self,
        batch_size=None,
        min_resolution=None,
        max_resolution=None,
        num_channels=None,
        num_images=None,
        size_divisor=None,
        equal_resolution=False,
        numpify=False,
        torchify=False,
    ):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.

        One can specify whether the images are of the same resolution or not.
        """
        assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

        batch_size = batch_size if batch_size is not None else self.batch_size
        min_resolution = min_resolution if min_resolution is not None else self.min_resolution
        max_resolution = max_resolution if max_resolution is not None else self.max_resolution
        num_channels = num_channels if num_channels is not None else self.num_channels
        num_images = num_images if num_images is not None else self.num_images

        images_list = []
        for i in range(batch_size):
            images = []
            for j in range(num_images):
                if equal_resolution:
                    width = height = max_resolution
                else:
                    # To avoid getting image width/height 0
                    if size_divisor is not None:
                        # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                        min_resolution = max(size_divisor, min_resolution)
                    width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
                images.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))
            images_list.append(images)

        if not numpify and not torchify:
            # PIL expects the channel dimension as last dimension
            images_list = [[Image.fromarray(np.moveaxis(image, 0, -1)) for image in images] for images in images_list]

        if torchify:
            images_list = [[torch.from_numpy(image) for image in images] for images in images_list]

        return images_list

    def expected_output_image_shape(self, images):
        expected_output_image_shape = (
            max(len(images) for images in images),
            self.max_image_tiles,
            self.num_channels,
            self.size["height"],
            self.size["width"],
        )
        return expected_output_image_shape


@require_torch
@require_vision
class MllamaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = MllamaImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = MllamaImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_pad"))
        self.assertTrue(hasattr(image_processing, "max_image_tiles"))

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for sample_images in image_inputs:
            for image in sample_images:
                self.assertIsInstance(image, np.ndarray)

        expected_output_image_shape = (
            max(len(images) for images in image_inputs),
            self.image_processor_tester.max_image_tiles,
            self.image_processor_tester.num_channels,
            self.image_processor_tester.size["height"],
            self.image_processor_tester.size["width"],
        )

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(
            tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
        )

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for images in image_inputs:
            for image in images:
                self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(
            tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
        )

    def test_call_channels_last(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)

        # a white 1x1 pixel RGB image
        image_inputs = [[np.ndarray(shape=(1, 1, 3), dtype=float, buffer=np.array([1.0, 1.0, 1.0]))]]
        encoded_images = image_processing(
            image_inputs, return_tensors="pt", input_data_format="channels_last"
        ).pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    def test_ambiguous_channel_pil_image(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)

        image_inputs = [[Image.new("RGB", (100, 1))]]
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for images in image_inputs:
            for image in images:
                self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (self.image_processor_tester.batch_size, *expected_output_image_shape),
        )

    def test_call_numpy_4_channels(self):
        self.skipTest("4 channels input is not supported yet")

    def test_image_correctly_tiled(self):
        def get_empty_tiles(pixel_values):
            # image has shape batch_size, max_num_images, max_image_tiles, num_channels, height, width
            # we want to get a binary mask of shape batch_size, max_num_images, max_image_tiles
            # of empty tiles, i.e. tiles that are completely zero
            return np.all(pixel_values == 0, axis=(3, 4, 5))

        image_processor_dict = {**self.image_processor_dict, "size": {"height": 50, "width": 50}, "max_image_tiles": 4}
        image_processor = self.image_processing_class(**image_processor_dict)

        # image fits 2x2 tiles grid (width x height)
        image = Image.new("RGB", (80, 95))
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs.pixel_values
        empty_tiles = get_empty_tiles(pixel_values)[0, 0].tolist()
        self.assertEqual(empty_tiles, [False, False, False, False])
        aspect_ratio_ids = inputs.aspect_ratio_ids[0, 0]
        self.assertEqual(aspect_ratio_ids, 6)
        aspect_ratio_mask = inputs.aspect_ratio_mask[0, 0].tolist()
        self.assertEqual(aspect_ratio_mask, [1, 1, 1, 1])

        # image fits 3x1 grid (width x height)
        image = Image.new("RGB", (101, 50))
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs.pixel_values
        empty_tiles = get_empty_tiles(pixel_values)[0, 0].tolist()
        self.assertEqual(empty_tiles, [False, False, False, True])
        aspect_ratio_ids = inputs.aspect_ratio_ids[0, 0]
        self.assertEqual(aspect_ratio_ids, 3)
        num_tiles = inputs.aspect_ratio_mask[0, 0].sum()
        self.assertEqual(num_tiles, 3)
        aspect_ratio_mask = inputs.aspect_ratio_mask[0, 0].tolist()
        self.assertEqual(aspect_ratio_mask, [1, 1, 1, 0])

        # image fits 1x1 grid (width x height)
        image = Image.new("RGB", (20, 39))
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs.pixel_values
        empty_tiles = get_empty_tiles(pixel_values)[0, 0].tolist()
        self.assertEqual(empty_tiles, [False, True, True, True])
        aspect_ratio_ids = inputs.aspect_ratio_ids[0, 0]
        self.assertEqual(aspect_ratio_ids, 1)
        aspect_ratio_mask = inputs.aspect_ratio_mask[0, 0].tolist()
        self.assertEqual(aspect_ratio_mask, [1, 0, 0, 0])

        # image fits 2x1 grid (width x height)
        image = Image.new("RGB", (51, 20))
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs.pixel_values
        empty_tiles = get_empty_tiles(pixel_values)[0, 0].tolist()
        self.assertEqual(empty_tiles, [False, False, True, True])
        aspect_ratio_ids = inputs.aspect_ratio_ids[0, 0]
        self.assertEqual(aspect_ratio_ids, 2)
        aspect_ratio_mask = inputs.aspect_ratio_mask[0, 0].tolist()
        self.assertEqual(aspect_ratio_mask, [1, 1, 0, 0])

        # image is greater than 2x2 tiles grid (width x height)
        image = Image.new("RGB", (150, 150))
        inputs = image_processor(image, return_tensors="np")
        pixel_values = inputs.pixel_values
        empty_tiles = get_empty_tiles(pixel_values)[0, 0].tolist()
        self.assertEqual(empty_tiles, [False, False, False, False])
        aspect_ratio_ids = inputs.aspect_ratio_ids[0, 0]
        self.assertEqual(aspect_ratio_ids, 6)  # (2 - 1) * 4 + 2 = 6
        aspect_ratio_mask = inputs.aspect_ratio_mask[0, 0].tolist()
        self.assertEqual(aspect_ratio_mask, [1, 1, 1, 1])

        # batch of images
        image1 = Image.new("RGB", (80, 95))
        image2 = Image.new("RGB", (101, 50))
        image3 = Image.new("RGB", (23, 49))
        inputs = image_processor([[image1], [image2, image3]], return_tensors="np")
        pixel_values = inputs.pixel_values
        empty_tiles = get_empty_tiles(pixel_values).tolist()
        expected_empty_tiles = [
            # sample 1 with 1 image 2x2 grid
            [
                [False, False, False, False],
                [True, True, True, True],  # padding
            ],
            # sample 2
            [
                [False, False, False, True],  # 3x1
                [False, True, True, True],  # 1x1
            ],
        ]
        self.assertEqual(empty_tiles, expected_empty_tiles)
        aspect_ratio_ids = inputs.aspect_ratio_ids.tolist()
        expected_aspect_ratio_ids = [[6, 0], [3, 1]]
        self.assertEqual(aspect_ratio_ids, expected_aspect_ratio_ids)
        aspect_ratio_mask = inputs.aspect_ratio_mask.tolist()
        expected_aspect_ratio_mask = [
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
            ],
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
            ],
        ]
        self.assertEqual(aspect_ratio_mask, expected_aspect_ratio_mask)
