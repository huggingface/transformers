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

from transformers import DonutImageProcessor, DonutImageProcessorPil, set_seed
from transformers.testing_utils import is_flaky, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class DonutImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 18, "width": 20})
        kwargs.setdefault("do_thumbnail", True)
        kwargs.setdefault("do_align_long_axis", False)
        kwargs.setdefault("do_pad", True)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        super().__init__(parent, **kwargs)


@require_torch
@require_vision
class DonutImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.image_processing_classes = {"torchvision": DonutImageProcessor, "pil": DonutImageProcessorPil}
        self.image_processor_tester = DonutImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_thumbnail"))
            self.assertTrue(hasattr(image_processing, "do_align_long_axis"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 20})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

            # Previous config had dimensions in (width, height) order
            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=(42, 84))
            self.assertEqual(image_processor.size, {"height": 84, "width": 42})

    def test_image_processor_preprocess_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            height = 84
            width = 42
            # Previous config had dimensions in (width, height) order
            encoded_images = image_processing(image_inputs[0], size=(width, height), return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    height,
                    width,
                ),
            )

    @is_flaky()
    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Set seed for deterministic test - ensures reproducible image generation
            set_seed(42)
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )

    @is_flaky()
    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            # Set seed for deterministic test - ensures reproducible image generation
            set_seed(42)
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )

    @is_flaky()
    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            # Set seed for deterministic test - ensures reproducible image generation
            set_seed(42)
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
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
class DonutImageProcessingAlignAxisTest(DonutImageProcessingTest):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = DonutImageProcessingTester(self, do_align_long_axis=True)
