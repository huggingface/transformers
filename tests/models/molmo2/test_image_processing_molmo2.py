# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available() and is_torchvision_available():
    from PIL import Image

    from transformers import Molmo2ImageProcessor


class Molmo2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)


@require_torch
@require_vision
@require_torchvision
class Molmo2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Molmo2ImageProcessor if (is_vision_available() and is_torchvision_available()) else None
    image_processor_tester_class = Molmo2ImageProcessingTester

    def test_image_processor_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_convert_rgb"))
        self.assertTrue(hasattr(image_processor, "max_crops"))
        self.assertTrue(hasattr(image_processor, "overlap_margins"))
        self.assertTrue(hasattr(image_processor, "patch_size"))
        self.assertTrue(hasattr(image_processor, "pooling_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 378, "width": 378})
        self.assertEqual(image_processor.do_normalize, True)

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size={"height": 400, "width": 400}, do_normalize=False
        )
        self.assertEqual(image_processor.size, {"height": 400, "width": 400})
        self.assertEqual(image_processor.do_normalize, False)

    def _assert_patchified_output(self, outputs, expected_num_images):
        pixel_values = outputs.pixel_values
        self.assertEqual(pixel_values.ndim, 3)
        pixels_per_patch = self.image_processor_tester.patch_size**2 * self.image_processor_tester.num_channels
        self.assertEqual(pixel_values.shape[-1], pixels_per_patch)
        image_num_crops = outputs.image_num_crops
        self.assertEqual(image_num_crops.shape[0], expected_num_images)
        self.assertEqual(pixel_values.shape[0], int(image_num_crops.sum().item()))

    def test_call_pil(self):
        for image_processing_class in [self.image_processing_class]:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            outputs = image_processing(image_inputs[0], return_tensors="pt")
            self._assert_patchified_output(outputs, 1)

            outputs = image_processing([[image] for image in image_inputs], return_tensors="pt")
            self._assert_patchified_output(outputs, self.image_processor_tester.batch_size)

    def test_call_numpy(self):
        for image_processing_class in [self.image_processing_class]:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            outputs = image_processing(image_inputs[0], return_tensors="pt")
            self._assert_patchified_output(outputs, 1)

            outputs = image_processing([[image] for image in image_inputs], return_tensors="pt")
            self._assert_patchified_output(outputs, self.image_processor_tester.batch_size)

    def test_call_nested_images(self):
        for image_processing_class in [self.image_processing_class]:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            nested_images = [[], [image_inputs[0]], [image_inputs[1], image_inputs[2]]]

            outputs = image_processing(nested_images, return_tensors="pt")

            self._assert_patchified_output(outputs, 3)
            for crops in outputs.image_num_crops.tolist():
                self.assertGreater(crops, 0)

    def test_call_pytorch(self):
        for image_processing_class in [self.image_processing_class]:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            outputs = image_processing(image_inputs[0], return_tensors="pt")
            self._assert_patchified_output(outputs, 1)

            outputs = image_processing([[image] for image in image_inputs], return_tensors="pt")
            self._assert_patchified_output(outputs, self.image_processor_tester.batch_size)

    @unittest.skip(
        reason="Molmo2ImageProcessor always converts to RGB before processing; 4-channel images are not supported."
    )
    def test_call_numpy_4_channels(self):
        pass
