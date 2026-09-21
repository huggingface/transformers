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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Gemma3ImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    do_resize = True
    size = {"height": 18, "width": 18}
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    do_pan_and_scan = True
    pan_and_scan_min_crop_size = 10
    pan_and_scan_max_num_crops = 2
    pan_and_scan_min_ratio_to_activate = 1.2


@require_torch
@require_vision
class Gemma3ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = Gemma3ImageProcessingTester

    def test_without_pan_and_scan(self):
        """
        Disable do_pan_and_scan parameter.
        """
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processor = image_processing_class.from_dict(self.image_processor_dict, do_pan_and_scan=False)

            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_pan_and_scan(self):
        """
        Enables Pan and Scan path by choosing the correct input image resolution. If you are changing
        image processor attributes for PaS, please update this test.
        """
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            """This function prepares a list of PIL images"""
            image_inputs = [np.random.randint(255, size=(3, 300, 600), dtype=np.uint8)] * 3
            image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

            # Test not batched input, 3 images because we have base image + 2 crops
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (3, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched, 9 images because we have base image + 2 crops per each item
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (9, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched unbalanced, 9 images because we have base image + 2 crops per each item
            encoded_images = image_processing(
                [[image_inputs[0], image_inputs[1]], [image_inputs[2]]], return_tensors="pt"
            ).pixel_values
            expected_output_image_shape = (9, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

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

    @unittest.skip("Gemma3 doesn't work with 4 channels due to pan and scan method")
    def test_call_numpy_4_channels(self):
        pass

    @require_vision
    @require_torch
    def test_backends_equivalence_batched_pas(self):
        """Test pan and scan equivalence across backends."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        crop_config = {
            "do_pan_and_scan": True,
            "pan_and_scan_max_num_crops": 448,
            "pan_and_scan_min_crop_size": 32,
            "pan_and_scan_min_ratio_to_activate": 0.3,
        }
        image_processor_dict = self.image_processor_dict
        image_processor_dict.update(crop_config)
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_encoding = encodings[backend_names[0]]
        for backend_name in backend_names[1:]:
            torch.testing.assert_close(reference_encoding.num_crops, encodings[backend_name].num_crops)
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
