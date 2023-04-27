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

    from transformers import BlipImageProcessor


class BlipImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        do_pad=False,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_pad": self.do_pad,
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
class BlipImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = BlipImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = BlipImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
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
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
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
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )


@require_torch
@require_vision
class BlipImageProcessingTestFourChannels(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = BlipImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = BlipImageProcessingTester(self, num_channels=4)
        self.expected_encoded_image_num_channels = 3

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    def test_batch_feature(self):
        pass

    def test_call_pil_four_channels(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.expected_encoded_image_num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.expected_encoded_image_num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
