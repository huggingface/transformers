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

from ...test_image_processing_common import ImageProcessingSavingTestMixin


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

    def prepare_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

        if equal_resolution:
            image_inputs = []
            for i in range(self.batch_size):
                image_inputs.append(
                    np.random.randint(
                        255, size=(self.num_channels, self.max_resolution, self.max_resolution), dtype=np.uint8
                    )
                )
        else:
            image_inputs = []
            for i in range(self.batch_size):
                width, height = np.random.choice(np.arange(self.min_resolution, self.max_resolution), 2)
                image_inputs.append(np.random.randint(255, size=(self.num_channels, width, height), dtype=np.uint8))

        if not numpify and not torchify:
            # PIL expects the channel dimension as last dimension
            image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        if torchify:
            image_inputs = [torch.from_numpy(x) for x in image_inputs]

        return image_inputs


@require_torch
@require_vision
class Swin2SRImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = Swin2SRImageProcessor if is_vision_available() else None

    def setUp(self):
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

    def test_batch_feature(self):
        pass

    def calculate_expected_size(self, image):
        old_height, old_width = get_image_size(image)
        size = self.image_processor_tester.pad_size

        pad_height = (old_height // size + 1) * size - old_height
        pad_width = (old_width // size + 1) * size - old_width
        return old_height + pad_height, old_width + pad_width

    def test_call_pil(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
        expected_height, expected_width = self.calculate_expected_size(np.array(image_inputs[0]))
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
        expected_height, expected_width = self.calculate_expected_size(image_inputs[0])
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_pytorch(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
        expected_height, expected_width = self.calculate_expected_size(image_inputs[0])
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )
