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

from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class ChameleonImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("max_resolution", 200)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 18})
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_size", {"height": 18, "width": 18})
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [1.0, 1.0, 1.0])
        kwargs.setdefault("image_std", [1.0, 1.0, 1.0])
        kwargs.setdefault("do_convert_rgb", True)
        kwargs.setdefault("resample", PILImageResampling.BILINEAR)
        super().__init__(parent, **kwargs)


@require_torch
@require_vision
class ChameleonImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.setUp with CLIP->Chameleon
    def setUp(self):
        super().setUp()
        self.image_processor_tester = ChameleonImageProcessingTester(self)

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_center_crop"))
            self.assertTrue(hasattr(image_processing, "crop_size"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 18})
            self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_nested_input(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched as a nested list of images, where each sublist is one batch
            image_inputs_nested = [image_inputs[:3], image_inputs[3:]]
            encoded_images_nested = image_processing(image_inputs_nested, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)

            # Image processor should return same pixel values, independently of input format
            self.assertTrue((encoded_images_nested == encoded_images).all())
