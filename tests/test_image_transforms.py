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
from transformers.testing_utils import require_flax, require_tf, require_torch, require_vision
from transformers.utils.import_utils import is_flax_available, is_tf_available, is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

if is_flax_available():
    import jax

if is_vision_available():
    import PIL.Image

    from transformers.image_transforms import (
        get_resize_output_image_size,
        resize,
        to_channel_dimension_format,
        to_pil_image,
    )


def get_random_image(height, width, num_channels=3, channels_first=True):
    shape = (num_channels, height, width) if channels_first else (height, width, num_channels)
    random_array = np.random.randint(0, 256, shape, dtype=np.uint8)
    return random_array


@require_vision
class ImageTransformsTester(unittest.TestCase):
    @parameterized.expand(
        [
            ("numpy_float_channels_first", (3, 4, 5), np.float32),
            ("numpy_float_channels_last", (4, 5, 3), np.float32),
            ("numpy_int_channels_first", (3, 4, 5), np.int32),
            ("numpy_uint_channels_first", (3, 4, 5), np.uint8),
        ]
    )
    @require_vision
    def test_to_pil_image(self, name, image_shape, dtype):
        image = np.random.randint(0, 256, image_shape).astype(dtype)
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    @require_tf
    def test_to_pil_image_from_tensorflow(self):
        # channels_first
        image = tf.random.uniform((3, 4, 5))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

        # channels_last
        image = tf.random.uniform((4, 5, 3))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    @require_torch
    def test_to_pil_image_from_torch(self):
        # channels first
        image = torch.rand((3, 4, 5))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

        # channels last
        image = torch.rand((4, 5, 3))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    @require_flax
    def test_to_pil_image_from_jax(self):
        key = jax.random.PRNGKey(0)
        # channel first
        image = jax.random.uniform(key, (3, 4, 5))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

        # channel last
        image = jax.random.uniform(key, (4, 5, 3))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    def test_to_channel_dimension_format(self):
        # Test that function doesn't reorder if channel dim matches the input.
        image = np.random.rand(3, 4, 5)
        image = to_channel_dimension_format(image, "channels_first")
        self.assertEqual(image.shape, (3, 4, 5))

        image = np.random.rand(4, 5, 3)
        image = to_channel_dimension_format(image, "channels_last")
        self.assertEqual(image.shape, (4, 5, 3))

        # Test that function reorders if channel dim doesn't match the input.
        image = np.random.rand(3, 4, 5)
        image = to_channel_dimension_format(image, "channels_last")
        self.assertEqual(image.shape, (4, 5, 3))

        image = np.random.rand(4, 5, 3)
        image = to_channel_dimension_format(image, "channels_first")
        self.assertEqual(image.shape, (3, 4, 5))

    def test_get_resize_output_image_size(self):
        image = np.random.randint(0, 256, (3, 224, 224))

        # Test the output size defaults to (x, x) if an int is given.
        self.assertEqual(get_resize_output_image_size(image, 10), (10, 10))
        self.assertEqual(get_resize_output_image_size(image, [10]), (10, 10))
        self.assertEqual(get_resize_output_image_size(image, (10,)), (10, 10))

        # Test the output size is the same as the input if a two element tuple/list is given.
        self.assertEqual(get_resize_output_image_size(image, (10, 20)), (10, 20))
        self.assertEqual(get_resize_output_image_size(image, [10, 20]), (10, 20))
        self.assertEqual(get_resize_output_image_size(image, (10, 20), default_to_square=True), (10, 20))
        # To match pytorch behaviour, max_size is only relevant if size is an int
        self.assertEqual(get_resize_output_image_size(image, (10, 20), max_size=5), (10, 20))

        # Test output size = (int(size * height / width), size) if size is an int and height > width
        image = np.random.randint(0, 256, (3, 50, 40))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False), (25, 20))

        # Test output size = (size, int(size * width / height)) if size is an int and width <= height
        image = np.random.randint(0, 256, (3, 40, 50))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False), (20, 25))

        # Test size is resized if longer size > max_size
        image = np.random.randint(0, 256, (3, 50, 40))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False, max_size=22), (22, 17))

    def test_resize(self):
        image = np.random.randint(0, 256, (3, 224, 224))

        # Check the channel order is the same by default
        resized_image = resize(image, (30, 40))
        self.assertIsInstance(resized_image, np.ndarray)
        self.assertEqual(resized_image.shape, (3, 30, 40))

        # Check channel order is changed if specified
        resized_image = resize(image, (30, 40), data_format="channels_last")
        self.assertIsInstance(resized_image, np.ndarray)
        self.assertEqual(resized_image.shape, (30, 40, 3))

        # Check PIL.Image.Image is return if return_numpy=False
        resized_image = resize(image, (30, 40), return_numpy=False)
        self.assertIsInstance(resized_image, PIL.Image.Image)
        # PIL size is in (width, height) order
        self.assertEqual(resized_image.size, (40, 30))
