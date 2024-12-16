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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import Swin2SRImageProcessor
    from transformers.image_transforms import get_image_size


class Swin2SRImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_pad=True,
        pad_size=8,
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
        self.pad_size = pad_size

    def prepare_image_processor_dict(self):
        return {
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_pad": self.do_pad,
            "pad_size": self.pad_size,
        }

    def expected_output_image_shape(self, images):
        img = images[0]

        if isinstance(img, Image.Image):
            input_width, input_height = img.size
        elif isinstance(img, np.ndarray):
            input_height, input_width = img.shape[-3:-1]
        else:
            input_height, input_width = img.shape[-2:]

        pad_height = (input_height // self.pad_size + 1) * self.pad_size - input_height
        pad_width = (input_width // self.pad_size + 1) * self.pad_size - input_width

        return self.num_channels, input_height + pad_height, input_width + pad_width

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
class Swin2SRImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Swin2SRImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Swin2SRImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "do_rescale"))
        self.assertTrue(hasattr(image_processor, "rescale_factor"))
        self.assertTrue(hasattr(image_processor, "do_pad"))
        self.assertTrue(hasattr(image_processor, "pad_size"))

    def calculate_expected_size(self, image):
        old_height, old_width = get_image_size(image)
        size = self.image_processor_tester.pad_size

        pad_height = (old_height // size + 1) * size - old_height
        pad_width = (old_width // size + 1) * size - old_width
        return old_height + pad_height, old_width + pad_width

    # Swin2SRImageProcessor does not support batched input
    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    # Swin2SRImageProcessor does not support batched input
    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    # Swin2SRImageProcessor does not support batched input
    def test_call_numpy_4_channels(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(
            image_inputs[0], return_tensors="pt", input_data_format="channels_last"
        ).pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
        self.image_processor_tester.num_channels = 3

    # Swin2SRImageProcessor does not support batched input
    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
