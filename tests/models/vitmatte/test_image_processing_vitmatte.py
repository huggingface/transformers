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


import unittest
import warnings

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image

    from transformers import VitMatteImageProcessor


class VitMatteImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_rescale=True,
        rescale_factor=0.5,
        do_pad=True,
        size_divisibility=10,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        super().__init__()
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.size_divisibility = size_divisibility
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_pad": self.do_pad,
            "size_divisibility": self.size_divisibility,
        }

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
class VitMatteImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = VitMatteImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = VitMatteImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "do_pad"))
        self.assertTrue(hasattr(image_processing, "size_divisibility"))

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

        # Verify that width and height can be divided by size_divisibility
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

        # Verify that width and height can be divided by size_divisibility
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.size[::-1])
        encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

        # Verify that width and height can be divided by size_divisibility
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_call_numpy_4_channels(self):
        # Test that can process images which have an arbitrary number of channels
        # Initialize image_processing
        image_processor = self.image_processing_class(**self.image_processor_dict)

        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        encoded_images = image_processor(
            images=image,
            trimaps=trimap,
            input_data_format="channels_first",
            image_mean=0,
            image_std=1,
            return_tensors="pt",
        ).pixel_values

        # Verify that width and height can be divided by size_divisibility
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_padding(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image = np.random.randn(3, 249, 491)
        images = image_processing.pad_image(image)
        assert images.shape == (3, 256, 512)

        image = np.random.randn(3, 249, 512)
        images = image_processing.pad_image(image)
        assert images.shape == (3, 256, 512)

    def test_image_processor_preprocess_arguments(self):
        # vitmatte require additional trimap input for image_processor
        # that is why we override original common test

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            image = self.image_processor_tester.prepare_image_inputs()[0]
            trimap = np.random.randint(0, 3, size=image.size[::-1])

            with warnings.catch_warnings(record=True) as raised_warnings:
                warnings.simplefilter("always")
                image_processor(image, trimaps=trimap, extra_argument=True)

            messages = " ".join([str(w.message) for w in raised_warnings])
            self.assertGreaterEqual(len(raised_warnings), 1)
            self.assertIn("extra_argument", messages)
